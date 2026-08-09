"""Stage 2: job-level accumulators to the model feature matrix.

Stage 1 stores additive statistics in three elapsed buckets, so the early
observation window is chosen here rather than in the expensive raw pass. All
inputs are metadata known when the job starts plus telemetry from inside that
window; nothing describing how the job ended is read.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    CATEGORICAL_FEATURES,
    HOST_DISPERSION_METRICS,
    LABEL_COLUMN,
    METRICS,
    PART_BOUNDS,
    PART_NAMES,
    POSITIVE_OUTCOMES,
)

LOGGER = logging.getLogger(__name__)

EPSILON = 1e-6
_TRAILING_DIGITS = re.compile(r"(\d+)")
META_COLUMNS = ("row_id", "jid", "start_epoch", "month", "target")


def parts_for_window(window_seconds: float) -> list[str]:
    return [name for name, (_, high) in zip(PART_NAMES, PART_BOUNDS) if high <= window_seconds]


def accumulator_paths(root: Path, cluster: str) -> list[Path]:
    paths = sorted((root / cluster).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Stage 1 accumulators for {cluster} under {root}")
    return paths


def load_accumulators(
    root: Path,
    cluster: str,
    min_month: str | None = None,
    max_month: str | None = None,
    max_rows: int | None = None,
    seed: int = 2026,
) -> pd.DataFrame:
    """Read accumulators, optionally restricted to a month range and row budget.

    Sampling keeps every row from the most recent months and thins older ones,
    because behaviour close to the scored period matters most.
    """
    frames = []
    for path in accumulator_paths(root, cluster):
        month = path.stem.split("_")[-1]
        if month == "unknown":
            continue
        if min_month is not None and month < min_month:
            continue
        if max_month is not None and month > max_month:
            continue
        frame = pq.read_table(path).to_pandas()
        for column in frame.columns:
            if frame[column].dtype == np.float64:
                frame[column] = frame[column].astype(np.float32)
        # Earliest-available records only stand in for jobs the strict early
        # window missed, so a proper record always wins.
        frame["source_rank"] = 1 if path.stem.startswith("fallback") else 0
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No accumulator months selected for cluster {cluster}")
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("source_rank", kind="stable").drop_duplicates(subset="jid", keep="first")
    data = data.drop(columns="source_rank").reset_index(drop=True)
    if max_rows is not None and len(data) > max_rows:
        data = _recency_sample(data, max_rows, seed)
    LOGGER.info("%s: %s job records loaded", cluster, f"{len(data):,}")
    return data.reset_index(drop=True)


def observation_delay(data: pd.DataFrame) -> pd.Series:
    """Seconds between the job anchor and its first observed telemetry tick."""
    columns = [f"tmin_{part}" for part in PART_NAMES if f"tmin_{part}" in data.columns]
    if not columns:
        return pd.Series(np.inf, index=data.index)
    return data[columns].bfill(axis=1).iloc[:, 0].fillna(np.inf)


def deduplicate_jobs(data: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per job, preferring the least truncated observation.

    A job running when a release chunk opened appears in two bundles: once
    anchored on its real start time and once anchored on the chunk boundary. The
    record that starts closest to the job's own start is the informative one.
    """
    order = np.argsort(observation_delay(data).to_numpy(dtype=np.float64), kind="stable")
    return data.iloc[order].drop_duplicates(subset="jid", keep="first")


def _recency_sample(data: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    order = np.argsort(-data["start_epoch"].fillna(-1.0).to_numpy(dtype=np.float64), kind="stable")
    ranked = data.iloc[order]
    keep_recent = min(len(ranked), max_rows // 2)
    recent = ranked.iloc[:keep_recent]
    older = ranked.iloc[keep_recent:]
    remaining = max_rows - len(recent)
    if remaining > 0 and len(older) > remaining:
        older = older.sample(n=remaining, random_state=seed)
    return pd.concat([recent, older], ignore_index=True)


def _coalesce(frame: pd.DataFrame, columns: list[str], reverse: bool) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    block = frame[present[::-1] if reverse else present].apply(pd.to_numeric, errors="coerce")
    return block.bfill(axis=1).iloc[:, 0]


def identifier_number(values: pd.Series) -> pd.Series:
    """The number inside an anonymised identifier, e.g. `USER12717_S` -> 12717.

    These are issued roughly in order of first appearance, so the number acts as
    a proxy for how recently the user, workload or account arrived — a signal
    that survives for identities the model has never seen, where the category
    itself is worthless.
    """
    extracted = values.astype("string").str.extract(_TRAILING_DIGITS, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def build_matrix(data: pd.DataFrame, window_seconds: float = 1800.0) -> pd.DataFrame:
    """Feature matrix for one cluster. Column order is stable across calls."""
    parts = parts_for_window(window_seconds)
    if not parts:
        raise ValueError(f"window_seconds={window_seconds} selects no telemetry bucket")
    index = data.index
    # Columns are collected in a dict and assembled once: inserting ~180 columns
    # into a live DataFrame reallocates its blocks on every assignment.
    out: dict[str, pd.Series] = {}

    numeric = {
        column: pd.to_numeric(data[column], errors="coerce")
        if column in data.columns
        else pd.Series(np.nan, index=index)
        for column in set(data.columns) - {"jid", "host", LABEL_COLUMN, "account", "queue", "unit", "jobname", "username", "host_list"}
    }

    def column(name: str) -> pd.Series:
        return numeric.get(name, pd.Series(np.nan, index=index))

    # ---- observation shape -------------------------------------------------
    rows_total = sum(column(f"rows_{part}").fillna(0.0) for part in parts)
    hosts_seen = pd.concat([column(f"hosts_{part}") for part in parts], axis=1).max(axis=1)
    tmin = pd.concat([column(f"tmin_{part}") for part in parts], axis=1).min(axis=1)
    tmax = pd.concat([column(f"tmax_{part}") for part in parts], axis=1).max(axis=1)
    out["obs_rows"] = rows_total
    out["obs_hosts"] = hosts_seen
    out["obs_first_offset"] = tmin
    out["obs_last_offset"] = tmax
    out["obs_span"] = tmax - tmin
    out["obs_rows_per_host"] = _safe_ratio(rows_total, hosts_seen)
    out["obs_delayed"] = (tmin > 60.0).astype(np.float32)
    for part in parts:
        out[f"obs_active_{part}"] = (column(f"rows_{part}").fillna(0.0) > 0).astype(np.float32)
    # A job still emitting telemetry at the end of the window has survived it;
    # one that stopped early has already left the queue.
    out["obs_reached_end"] = (tmax >= window_seconds - 120.0).astype(np.float32)

    # ---- requested resources and queue context -----------------------------
    timelimit = column("timelimit")
    nhosts = column("nhosts")
    ncores = column("ncores")
    out["timelimit"] = timelimit
    out["timelimit_log"] = np.log1p(timelimit.clip(lower=0))
    out["nhosts"] = nhosts
    out["nhosts_log"] = np.log1p(nhosts.clip(lower=0))
    out["ncores"] = ncores
    out["ncores_log"] = np.log1p(ncores.clip(lower=0))
    out["cores_per_host"] = _safe_ratio(ncores, nhosts)
    out["hosts_seen_ratio"] = _safe_ratio(hosts_seen, nhosts)
    out["hosts_missing"] = nhosts - hosts_seen

    start = column("start_epoch")
    submit = column("submit_epoch")
    queue_wait = (start - submit).clip(lower=0.0)
    out["queue_wait"] = queue_wait
    out["queue_wait_log"] = np.log1p(queue_wait)
    out["queue_wait_over_timelimit"] = _safe_ratio(queue_wait, timelimit)

    start_time = pd.to_datetime(start, unit="s", utc=True, errors="coerce")
    hour = start_time.dt.hour.astype(np.float32)
    out["start_hour"] = hour
    out["start_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["start_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["start_dayofweek"] = start_time.dt.dayofweek.astype(np.float32)
    out["start_is_weekend"] = (start_time.dt.dayofweek >= 5).astype(np.float32)
    out["start_day"] = start_time.dt.day.astype(np.float32)
    submit_time = pd.to_datetime(submit, unit="s", utc=True, errors="coerce")
    out["submit_hour"] = submit_time.dt.hour.astype(np.float32)
    out["submit_dayofweek"] = submit_time.dt.dayofweek.astype(np.float32)

    host_list = data["host_list"].astype("string") if "host_list" in data.columns else pd.Series(pd.NA, index=index, dtype="string")
    listed = host_list.fillna("").str.strip("{}").str.count(",") + 1.0
    listed = listed.where(host_list.fillna("").str.len() > 0)
    out["host_list_count"] = listed
    out["host_list_ratio"] = _safe_ratio(listed, nhosts)

    # ---- telemetry ---------------------------------------------------------
    metric_means: dict[str, pd.Series] = {}
    for metric in METRICS:
        count = sum(column(f"{metric}_{part}_cnt").fillna(0.0) for part in parts)
        total = sum(column(f"{metric}_{part}_sum").fillna(0.0) for part in parts)
        squares = sum(column(f"{metric}_{part}_sq").fillna(0.0) for part in parts)
        zeros = sum(column(f"{metric}_{part}_zero").fillna(0.0) for part in parts)
        mean = _safe_ratio(total, count)
        variance = (_safe_ratio(squares, count) - mean * mean).clip(lower=0.0)
        minimum = pd.concat([column(f"{metric}_{part}_min") for part in parts], axis=1).min(axis=1)
        maximum = pd.concat([column(f"{metric}_{part}_max") for part in parts], axis=1).max(axis=1)
        first = _coalesce(data, [f"{metric}_{part}_first" for part in parts], reverse=False)
        last = _coalesce(data, [f"{metric}_{part}_last" for part in parts], reverse=True)
        metric_means[metric] = mean
        out[f"{metric}_cnt"] = count
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = np.sqrt(variance)
        out[f"{metric}_cv"] = _safe_ratio(np.sqrt(variance), mean.abs() + EPSILON)
        out[f"{metric}_min"] = minimum
        out[f"{metric}_max"] = maximum
        out[f"{metric}_range"] = maximum - minimum
        out[f"{metric}_first"] = first
        out[f"{metric}_last"] = last
        out[f"{metric}_delta"] = last - first
        out[f"{metric}_zero_frac"] = _safe_ratio(zeros, count)
        out[f"{metric}_coverage"] = _safe_ratio(count, rows_total)
        # Direction of travel inside the window: the opening bucket against
        # everything after it.
        head_count = column(f"{metric}_{parts[0]}_cnt").fillna(0.0)
        head_sum = column(f"{metric}_{parts[0]}_sum").fillna(0.0)
        tail_count = count - head_count
        tail_sum = total - head_sum
        head_mean = _safe_ratio(head_sum, head_count)
        tail_mean = _safe_ratio(tail_sum, tail_count)
        out[f"{metric}_head_mean"] = head_mean
        out[f"{metric}_tail_mean"] = tail_mean
        out[f"{metric}_trend"] = tail_mean - head_mean
        out[f"{metric}_trend_ratio"] = _safe_ratio(tail_mean, head_mean.abs() + EPSILON)

    for metric in HOST_DISPERSION_METRICS:
        spread = column(f"hostdisp_{metric}_max") - column(f"hostdisp_{metric}_min")
        out[f"hostdisp_{metric}_std"] = column(f"hostdisp_{metric}_std")
        out[f"hostdisp_{metric}_range"] = spread
        out[f"hostdisp_{metric}_rel"] = _safe_ratio(spread, metric_means[metric].abs() + EPSILON)
        out[f"hostdisp_{metric}_cnt"] = column(f"hostdisp_{metric}_cnt")

    out["mem_per_core"] = _safe_ratio(metric_means["memused"], out["cores_per_host"])
    out["mem_diskcache_gap"] = metric_means["memused"] - metric_means["memused_minus_diskcache"]
    out["mem_diskcache_ratio"] = _safe_ratio(
        metric_means["memused_minus_diskcache"], metric_means["memused"]
    )
    out["cpu_per_core"] = _safe_ratio(metric_means["cpuuser"], out["cores_per_host"])
    out["io_ratio"] = _safe_ratio(metric_means["nfs"], metric_means["block"].abs() + EPSILON)

    matrix = pd.DataFrame(out, index=index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    del out

    # ---- categorical and identifier columns --------------------------------
    extra: dict[str, pd.Series] = {}
    for name in ("account", "queue", "unit", "username", "jobname"):
        extra[name] = (
            data[name].astype("string")
            if name in data.columns
            else pd.Series(pd.NA, index=index, dtype="string")
        )
    extra["timelimit_bucket"] = (
        np.log1p(timelimit.clip(lower=0)).round(1).astype("string").fillna("na")
    )
    extra["jid"] = data["jid"].astype("string")
    extra["start_epoch"] = start
    extra["month"] = pd.to_datetime(start, unit="s", utc=True, errors="coerce").dt.strftime("%Y-%m")
    if LABEL_COLUMN in data.columns:
        extra["target"] = (
            data[LABEL_COLUMN]
            .astype("string")
            .fillna("")
            .str.upper()
            .isin(POSITIVE_OUTCOMES)
            .astype(np.int8)
        )
    return pd.concat([matrix, pd.DataFrame(extra, index=index)], axis=1)


def numeric_feature_columns(matrix: pd.DataFrame) -> list[str]:
    excluded = set(META_COLUMNS) | set(CATEGORICAL_FEATURES) | {"timelimit_bucket"}
    return [column for column in matrix.columns if column not in excluded]


def reference_range_guard(
    matrix: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
    max_outside: float = 0.20,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Drop features whose scored values sit largely outside the training range.

    A boosted tree cannot extrapolate: every value above the largest one it was
    trained on falls in the same terminal bin and receives whatever response the
    training tail happened to have. That is harmless for bounded quantities, and
    ruinous for one that tracks calendar time.

    This exists because of a regression it would have prevented. Identifier
    numbers were added as features on the reasoning that they proxy how new a
    user or workload is; they do, but they also encode when it first appeared,
    so 46-63 % of scored rows landed beyond every training value. The holdout,
    sitting one month after training rather than one to six, barely saw the
    effect and reported a gain. The leaderboard fell by 0.018.
    """
    kept: list[str] = []
    dropped: list[tuple[str, float]] = []
    for name in columns:
        if name not in reference.columns:
            kept.append(name)
            continue
        train = matrix[name].to_numpy(dtype=np.float64, copy=False)
        train = train[np.isfinite(train)]
        scored = reference[name].to_numpy(dtype=np.float64, copy=False)
        scored = scored[np.isfinite(scored)]
        if train.size == 0 or scored.size == 0:
            kept.append(name)
            continue
        low, high = float(train.min()), float(train.max())
        outside = float(np.mean((scored < low) | (scored > high)))
        if outside > max_outside:
            dropped.append((name, round(outside, 4)))
        else:
            kept.append(name)
    return kept, dropped


def drop_constant_columns(matrix: pd.DataFrame, columns: list[str]) -> list[str]:
    """Remove features with no usable variation, such as the always-null GPU
    channel, so cluster models stay compact."""
    keep = []
    for name in columns:
        values = matrix[name].to_numpy(dtype=np.float32, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if finite.size < len(values) or float(finite.min()) != float(finite.max()):
            keep.append(name)
    return keep
