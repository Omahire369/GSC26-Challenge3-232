"""Stage 1: raw hourly Parquet telemetry to mergeable job-level accumulators.

Only metadata known at submit/start time and telemetry from the first 30 minutes
of a job are read. `end_time` is never requested, and `exitcode` is requested
only for labelled bundles, where it becomes the target and never a feature.

Every statistic written here is additive or associative across files, so the
first-30-minute window of a job that straddles two hourly files is reconstructed
exactly. Stage 2 turns these accumulators into model features, which means the
early-window length can be re-chosen without touching the raw data again.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    CLUSTERS,
    HOST_DISPERSION_METRICS,
    LABEL_COLUMN,
    MAX_WINDOW_SECONDS,
    META_STRING_COLUMNS,
    METRICS,
    PART_BOUNDS,
    PART_NAMES,
    PRESTART_SECONDS,
    RAW_COLUMNS,
    STATIC_NUMERIC_COLUMNS,
    VALUE_COLUMNS,
    cluster_for_path,
    host_dispersion_columns,
    metric_stat_columns,
    part_shape_columns,
)

LOGGER = logging.getLogger(__name__)

IDENTITY_COLUMNS = ("jid", "start_epoch", "submit_epoch", "chunk_start_epoch")
FIRST_META_COLUMNS = STATIC_NUMERIC_COLUMNS + META_STRING_COLUMNS + ("host",)

_SUM_MERGE = tuple(
    f"{metric}_{part}_{stat}"
    for metric in METRICS
    for part in PART_NAMES
    for stat in ("cnt", "sum", "sq", "zero")
) + tuple(f"rows_{part}" for part in PART_NAMES)
_MIN_MERGE = tuple(
    f"{metric}_{part}_min" for metric in METRICS for part in PART_NAMES
) + tuple(f"tmin_{part}" for part in PART_NAMES)
_MAX_MERGE = (
    tuple(f"{metric}_{part}_max" for metric in METRICS for part in PART_NAMES)
    + tuple(f"tmax_{part}" for part in PART_NAMES)
    + tuple(f"hosts_{part}" for part in PART_NAMES)
)
# `first`, host dispersion and the static metadata are taken from the partial
# with the earliest telemetry, which is why partials are merged in time order.
_FIRST_MERGE = (
    tuple(f"{metric}_{part}_first" for metric in METRICS for part in PART_NAMES)
    + host_dispersion_columns()
    + FIRST_META_COLUMNS
    + ("start_epoch", "submit_epoch", LABEL_COLUMN)
)
_LAST_MERGE = tuple(
    f"{metric}_{part}_last" for metric in METRICS for part in PART_NAMES
)

ACCUMULATOR_COLUMNS = (
    ("jid",)
    + metric_stat_columns()
    + part_shape_columns()
    + host_dispersion_columns()
    + ("start_epoch", "submit_epoch", "n_hosts_seen")
    + FIRST_META_COLUMNS
)


def list_files(
    source: Path, cluster: str, from_month: str | None = None, to_month: str | None = None
) -> list[Path]:
    """Hourly files sorted by name, which for this dataset is time order."""
    paths = [p for p in source.rglob("*.parquet") if cluster_for_path(p) == cluster]
    if from_month is not None:
        paths = [p for p in paths if p.name[:7] >= from_month]
    if to_month is not None:
        paths = [p for p in paths if p.name[:7] <= to_month]
    return sorted(paths, key=lambda p: p.name)


def _read_frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    table = pq.read_table(path, columns=[c for c in columns if c in available])
    arrays, names = [], []
    for index, field in enumerate(table.schema):
        column = table.column(index)
        # Some releases store `exitcode` as a uint32-indexed dictionary, which
        # pandas cannot consume directly.
        arrays.append(column.cast(pa.string()) if pa.types.is_dictionary(field.type) else column)
        names.append(field.name)
    frame = pa.table(arrays, names=names).to_pandas()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _epoch_seconds(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(None)
    seconds = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(np.float64)
    seconds[pd.isna(parsed).to_numpy()] = np.nan
    return seconds / 1e9


def _prepare(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = _read_frame(path, columns)
    frame["jid"] = frame["jid"].astype("string")
    frame = frame.loc[frame["jid"].notna()].copy()
    if frame.empty:
        return frame
    # The coverage pass asks for only `time` and `jid`, so every conversion here
    # is conditional on the column actually having been requested.
    for name, target in (
        ("time", "time_epoch"),
        ("submit_time", "submit_epoch"),
        ("start_time", "start_epoch"),
    ):
        if name in frame.columns:
            frame[target] = _epoch_seconds(frame[name])
    for column in STATIC_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in VALUE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[np.isfinite(frame["time_epoch"].to_numpy())]
    return frame


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Anvil repeats each (job, host, tick) once per `unit`, with near-identical
    values; keep the most complete copy of every measurement."""
    value_columns = [f"value_{metric}" for metric in METRICS]
    completeness = frame[value_columns].notna().sum(axis=1).to_numpy()
    order = np.lexsort((-completeness, frame["time_epoch"].to_numpy()))
    frame = frame.iloc[order]
    return frame.drop_duplicates(subset=["jid", "host", "time_epoch"], keep="first")


def _host_dispersion(selected: pd.DataFrame) -> pd.DataFrame:
    """Spread of a metric across the nodes of one job.

    A single node behaving differently from its peers is the visible signature
    of a straggler or a failing node, which per-row statistics wash out.
    """
    columns = [f"value_{metric}" for metric in HOST_DISPERSION_METRICS]
    per_host = selected.groupby(["jid", "host"], sort=False)[columns].mean()
    block = per_host.groupby(level=0, sort=False).agg(["std", "min", "max", "count"])
    block.columns = [
        f"hostdisp_{column[len('value_'):]}_{aggregation}" for column, aggregation in block.columns
    ]
    return block


def _part_index(elapsed: np.ndarray) -> np.ndarray:
    index = np.full(elapsed.shape, -1, dtype=np.int8)
    for position, (low, high) in enumerate(PART_BOUNDS):
        index[(elapsed >= low) & (elapsed < high)] = position
    index[elapsed == PART_BOUNDS[-1][1]] = len(PART_BOUNDS) - 1
    return index


_VALUE_AGGREGATIONS = ("count", "sum", "min", "max", "first", "last")
_STAT_FOR_AGGREGATION = {
    "count": "cnt",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "first": "first",
    "last": "last",
}


def _aggregate_parts(selected: pd.DataFrame) -> pd.DataFrame:
    """One row per job holding per-metric accumulators for each elapsed bucket.

    Every statistic for a bucket comes out of a single grouped pass, which
    matters because this runs once per hourly file across the whole release.
    """
    specification: dict[str, list[str]] = {"host": ["nunique"], "elapsed": ["min", "max", "size"]}
    for metric in METRICS:
        specification[f"value_{metric}"] = list(_VALUE_AGGREGATIONS)
        specification[f"sq_{metric}"] = ["sum"]
        specification[f"zero_{metric}"] = ["sum"]

    pieces: list[pd.DataFrame] = []
    for position, part in enumerate(PART_NAMES):
        subset = selected.loc[selected["part"].to_numpy() == position]
        if subset.empty:
            continue
        block = subset.groupby("jid", sort=False).agg(specification)
        renamed = {}
        for column, aggregation in block.columns:
            if column == "host":
                renamed[(column, aggregation)] = f"hosts_{part}"
            elif column == "elapsed":
                renamed[(column, aggregation)] = {
                    "min": f"tmin_{part}",
                    "max": f"tmax_{part}",
                    "size": f"rows_{part}",
                }[aggregation]
            elif column.startswith("sq_"):
                renamed[(column, aggregation)] = f"{column[3:]}_{part}_sq"
            elif column.startswith("zero_"):
                renamed[(column, aggregation)] = f"{column[5:]}_{part}_zero"
            else:
                metric = column[len("value_") :]
                renamed[(column, aggregation)] = f"{metric}_{part}_{_STAT_FOR_AGGREGATION[aggregation]}"
        block.columns = [renamed[column] for column in block.columns]
        pieces.append(block)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)


def summarise_file(
    path_text: str,
    chunk_start_epoch: float,
    labeled: bool,
    use_anchors: bool = False,
) -> pd.DataFrame:
    """Partial accumulators for every job whose early window touches this file.

    With `use_anchors`, each job is anchored on the explicit epoch held in the
    process-wide anchor map instead of on its scheduler start time. That is the
    earliest-available pass for scored jobs whose telemetry is missing from the
    strict early window, usually because of a gap in the released hours.
    """
    columns = RAW_COLUMNS + ((LABEL_COLUMN,) if labeled else ())
    frame = _prepare(Path(path_text), columns)
    if frame.empty:
        return pd.DataFrame()

    if use_anchors:
        frame = frame.loc[frame["jid"].isin(_ANCHORS.keys())]
        if frame.empty:
            return pd.DataFrame()
        anchor = frame["jid"].map(_ANCHORS).to_numpy(dtype=np.float64)
    else:
        start = frame["start_epoch"].to_numpy(dtype=np.float64)
        # A job that was already running when the chunk opened is anchored at
        # the chunk boundary: that is the earliest moment it could be observed.
        anchor = np.maximum(start, chunk_start_epoch)
        missing = ~np.isfinite(start)
        if missing.any():
            fallback = frame.groupby("jid", sort=False)["time_epoch"].transform("min")
            anchor = np.where(missing, fallback.to_numpy(dtype=np.float64), anchor)

    elapsed = frame["time_epoch"].to_numpy(dtype=np.float64) - anchor
    keep = (elapsed >= -PRESTART_SECONDS) & (elapsed <= MAX_WINDOW_SECONDS)
    if not keep.any():
        return pd.DataFrame()
    selected = frame.loc[keep].copy()
    selected["elapsed"] = elapsed[keep]
    selected = _deduplicate(selected)
    selected["part"] = _part_index(selected["elapsed"].to_numpy())
    selected = selected.loc[selected["part"].to_numpy() >= 0]
    if selected.empty:
        return pd.DataFrame()
    selected = selected.sort_values(["jid", "elapsed"], kind="stable")
    for metric in METRICS:
        values = selected[f"value_{metric}"]
        selected[f"sq_{metric}"] = values * values
        selected[f"zero_{metric}"] = (values == 0).astype(np.float32).where(values.notna())

    result = _aggregate_parts(selected)
    if result.empty:
        return pd.DataFrame()

    grouped = selected.groupby("jid", sort=False)
    meta = grouped[list(FIRST_META_COLUMNS)].first()
    meta["start_epoch"] = grouped["start_epoch"].first()
    meta["submit_epoch"] = grouped["submit_epoch"].first()
    meta["n_hosts_seen"] = grouped["host"].nunique()
    if labeled:
        meta[LABEL_COLUMN] = grouped[LABEL_COLUMN].first()
    result = result.join(meta, how="left").join(_host_dispersion(selected), how="left")
    return _normalise(result.reset_index().rename(columns={"index": "jid"}))


def merge_partials(partials: pd.DataFrame) -> pd.DataFrame:
    """Combine per-file partials into one record per job.

    Callers pass partials in file order, so `first`/`last` reductions resolve to
    the earliest and latest telemetry seen for the job.
    """
    grouped = partials.groupby("jid", sort=False)
    blocks = []
    for names, how in (
        (_SUM_MERGE, "sum"),
        (_MIN_MERGE, "min"),
        (_MAX_MERGE, "max"),
        (_FIRST_MERGE, "first"),
        (_LAST_MERGE, "last"),
    ):
        present = [name for name in names if name in partials.columns]
        if not present:
            continue
        blocks.append(getattr(grouped[present], how)())
    merged = pd.concat(blocks, axis=1)
    merged["n_hosts_seen"] = grouped["n_hosts_seen"].max()
    return merged.reset_index()


class MonthlyWriter:
    """Buffered Parquet writer that partitions job records by start month."""

    def __init__(self, output: Path, prefix: str, flush_rows: int = 20_000) -> None:
        self.output = output
        self.prefix = prefix
        self.flush_rows = flush_rows
        self.buffers: dict[str, list[pd.DataFrame]] = {}
        self.counts: dict[str, int] = {}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.schema: pa.Schema | None = None
        self.column_order: list[str] | None = None
        self.total = 0

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        epoch = pd.to_numeric(frame["start_epoch"], errors="coerce")
        month = pd.to_datetime(epoch, unit="s", utc=True, errors="coerce").dt.strftime("%Y-%m")
        month = month.fillna("unknown")
        for key, block in frame.groupby(month.to_numpy(), sort=False):
            self.buffers.setdefault(key, []).append(block)
            self.counts[key] = self.counts.get(key, 0) + len(block)
            if self.counts[key] >= self.flush_rows:
                self.flush(key)

    def flush(self, month: str) -> None:
        blocks = self.buffers.get(month)
        if not blocks:
            return
        frame = pd.concat(blocks, ignore_index=True)
        self.buffers[month] = []
        self.counts[month] = 0
        if self.column_order is None:
            self.column_order = list(frame.columns)
        # Every month shares one schema, so column order has to be pinned to
        # whatever the first flush established.
        frame = frame.reindex(columns=self.column_order)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.schema is None:
            self.schema = table.schema
        if month not in self.writers:
            self.output.mkdir(parents=True, exist_ok=True)
            self.writers[month] = pq.ParquetWriter(
                self.output / f"{self.prefix}_{month}.parquet", self.schema, compression="zstd"
            )
        self.writers[month].write_table(table.cast(self.schema))
        self.total += len(frame)

    def close(self) -> int:
        for month in list(self.buffers):
            self.flush(month)
        for writer in self.writers.values():
            writer.close()
        return self.total


_TEXT_COLUMNS = frozenset(META_STRING_COLUMNS) | {"jid", "host", LABEL_COLUMN}


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every partial the same columns and dtypes before concatenation.

    Float32 halves both the pickling cost back from the worker pool and the
    memory held while a batch is reduced.
    """
    if frame.empty:
        return frame
    missing = [column for column in ACCUMULATOR_COLUMNS if column not in frame.columns]
    if missing:
        frame = pd.concat(
            [frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1
        )
    converted = {}
    for column in frame.columns:
        if column in _TEXT_COLUMNS:
            converted[column] = frame[column].astype("string")
        else:
            values = pd.to_numeric(frame[column], errors="coerce")
            # Epoch seconds need more than float32's ~7 significant digits.
            converted[column] = values.astype(
                np.float64 if column.endswith("_epoch") else np.float32
            )
    return pd.DataFrame(converted, index=frame.index)


def build_cluster(
    source: Path,
    output: Path,
    cluster: str,
    labeled: bool,
    workers: int,
    batch_files: int,
    prefix: str = "jobs",
    job_anchors: dict[str, float] | None = None,
    from_month: str | None = None,
    to_month: str | None = None,
    restrict_hours: set[str] | None = None,
) -> int:
    paths = list_files(source, cluster, from_month, to_month)
    if restrict_hours is not None:
        paths = [path for path in paths if path.name[:13] in restrict_hours]
    if not paths:
        raise FileNotFoundError(f"No {cluster} Parquet files under {source}")
    chunk_start_epoch = float(
        pd.Timestamp(
            f"{paths[0].name[:10]} {paths[0].name[11:13]}:00:00", tz="UTC"
        ).timestamp()
    )
    LOGGER.info("%s: %s files, chunk starts %s", cluster, len(paths), paths[0].name)

    writer = MonthlyWriter(output / cluster, prefix)
    pending = pd.DataFrame()
    use_anchors = job_anchors is not None
    pool_options: dict[str, Any] = {"max_workers": workers}
    if use_anchors:
        # Sent once per worker rather than once per task.
        pool_options["initializer"] = _init_anchors
        pool_options["initargs"] = (job_anchors,)
        _init_anchors(job_anchors)

    for offset in range(0, len(paths), batch_files):
        batch = paths[offset : offset + batch_files]
        arguments = [(str(path), chunk_start_epoch, labeled, use_anchors) for path in batch]
        if workers > 1:
            # A fresh pool per batch keeps worker memory flat over tens of
            # thousands of files, and unlike `max_tasks_per_child` it cannot
            # leave the pool waiting on a replacement worker that never starts.
            with ProcessPoolExecutor(**pool_options) as executor:
                results = list(executor.map(summarise_file, *zip(*arguments), chunksize=2))
        else:
            results = [summarise_file(*argument) for argument in arguments]
        frames = [frame for frame in results if not frame.empty]
        del results
        if not frames and pending.empty:
            continue
        partials = pd.concat(([pending] if not pending.empty else []) + frames, ignore_index=True)
        del frames
        merged = merge_partials(partials)
        del partials
        # A job's window can reach into the next hourly file, so anything
        # anchored near the batch edge is carried over rather than written.
        batch_end = float(
            pd.Timestamp(
                f"{batch[-1].name[:10]} {batch[-1].name[11:13]}:00:00", tz="UTC"
            ).timestamp()
        )
        anchor = np.maximum(merged["start_epoch"].to_numpy(dtype=np.float64), chunk_start_epoch)
        carry = ~np.isfinite(anchor) | (anchor > batch_end - MAX_WINDOW_SECONDS)
        writer.add(merged.loc[~carry])
        pending = _normalise(merged.loc[carry].copy())
        del merged
        gc.collect()
        LOGGER.info(
            "%s: %s/%s files, %s jobs written, %s carried",
            cluster,
            min(offset + batch_files, len(paths)),
            len(paths),
            f"{writer.total:,}",
            f"{len(pending):,}",
        )
    writer.add(pending)
    return writer.close()


def existing_job_ids(root: Path, cluster: str) -> set[str]:
    directory = root / cluster
    found: set[str] = set()
    for path in sorted(directory.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["jid"], batch_size=200_000):
            found.update(batch.column(0).to_pylist())
    return found


_WANTED: frozenset[str] = frozenset()
_ANCHORS: dict[str, float] = {}


def _init_wanted(wanted: frozenset[str]) -> None:
    global _WANTED
    _WANTED = wanted


def _init_anchors(anchors: dict[str, float]) -> None:
    global _ANCHORS
    _ANCHORS = anchors


def first_seen_in_file(path_text: str) -> dict[str, float]:
    """Earliest telemetry timestamp in this file for each job of interest."""
    frame = _prepare(Path(path_text), ("time", "jid"))
    if frame.empty:
        return {}
    frame = frame.loc[frame["jid"].isin(_WANTED)]
    if frame.empty:
        return {}
    grouped = frame.groupby("jid", sort=False)["time_epoch"].min()
    return {str(job_id): float(value) for job_id, value in grouped.items()}


def _first_seen(source: Path, cluster: str, wanted: set[str], workers: int) -> dict[str, float]:
    """Earliest released telemetry timestamp for each requested job.

    Only `time` and `jid` are read, so this stays cheap even when it has to look
    at the whole release.
    """
    paths = [str(path) for path in list_files(source, cluster)]
    frozen = frozenset(wanted)
    anchors: dict[str, float] = {}

    def absorb(found: dict[str, float]) -> None:
        for job_id, value in found.items():
            current = anchors.get(job_id)
            if current is None or value < current:
                anchors[job_id] = value

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_wanted, initargs=(frozen,)
        ) as executor:
            for index, found in enumerate(
                executor.map(first_seen_in_file, paths, chunksize=8), start=1
            ):
                absorb(found)
                if index % 400 == 0:
                    LOGGER.info(
                        "%s: first-seen scan %s/%s files, %s/%s jobs located",
                        cluster, index, len(paths), len(anchors), len(frozen),
                    )
    else:
        _init_wanted(frozen)
        for index, path in enumerate(paths, start=1):
            absorb(first_seen_in_file(path))
            if len(anchors) == len(frozen):
                break
    LOGGER.info("%s: located %s/%s uncovered jobs", cluster, len(anchors), len(frozen))
    return anchors


def fill_missing(
    source: Path, output: Path, cluster: str, wanted: set[str], workers: int, batch_files: int
) -> int:
    """Earliest-available features for scored jobs the strict window missed.

    Restricted to job ids that produced no record in the main pass, and the
    resulting records still carry the observation-delay feature so the model can
    tell them apart.
    """
    missing = wanted - existing_job_ids(output, cluster)
    if not missing:
        LOGGER.info("%s: strict early-window coverage is complete", cluster)
        return 0
    LOGGER.info("%s: %s scored jobs need earliest-available features", cluster, len(missing))
    anchors = _first_seen(source, cluster, missing, workers)
    if not anchors:
        raise ValueError(f"{cluster}: no released telemetry for {len(missing)} scored jobs")
    # Each of these jobs needs only the two or three hourly files its own window
    # touches, so the pass reads those instead of the whole release again.
    hours: set[str] = set()
    for anchor in anchors.values():
        start = anchor - PRESTART_SECONDS
        stop = anchor + MAX_WINDOW_SECONDS
        hour = np.floor(start / 3600.0) * 3600.0
        while hour <= stop:
            hours.add(pd.Timestamp(hour, unit="s", tz="UTC").strftime("%Y-%m-%d-%H"))
            hour += 3600.0
    LOGGER.info("%s: coverage pass reads %s hourly files", cluster, len(hours))
    return build_cluster(
        source,
        output,
        cluster,
        False,
        workers,
        batch_files,
        prefix="fallback",
        job_anchors=anchors,
        restrict_hours=hours,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build job-level FRESCO accumulators.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labeled", action="store_true")
    parser.add_argument("--clusters", nargs="+", choices=CLUSTERS, default=list(CLUSTERS))
    parser.add_argument("--sample", type=Path, help="sample_submission.csv; enables coverage fill")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument(
        "--batch-files",
        type=int,
        default=150,
        help="Files reduced together; larger batches trade memory for speed",
    )
    parser.add_argument("--from-month", help="Earliest hourly file month, e.g. 2015-01")
    parser.add_argument("--to-month", help="Latest hourly file month, e.g. 2016-12")
    parser.add_argument(
        "--fill-only",
        action="store_true",
        help="Skip the main pass and only add earliest-available records for uncovered scored jobs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    required: dict[str, set[str]] = {}
    if args.sample is not None:
        sample = pd.read_csv(args.sample, usecols=["row_id"])["row_id"].astype(str)
        for cluster in args.clusters:
            prefix = f"{cluster}_"
            selected = sample.loc[sample.str.startswith(prefix)]
            required[cluster] = set(selected.str[len(prefix) :])

    counts: dict[str, int] = {}
    for cluster in args.clusters:
        counts[cluster] = 0
        if not args.fill_only:
            counts[cluster] = build_cluster(
                args.source,
                args.output,
                cluster,
                args.labeled,
                args.workers,
                args.batch_files,
                from_month=args.from_month,
                to_month=args.to_month,
            )
        if cluster in required:
            counts[cluster] += fill_missing(
                args.source, args.output, cluster, required[cluster], args.workers, args.batch_files
            )
        LOGGER.info("%s: %s job records", cluster, f"{counts[cluster]:,}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate_metadata.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "labeled": args.labeled,
                "prestart_seconds": PRESTART_SECONDS,
                "part_bounds": PART_BOUNDS,
                "job_counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Stage 1 complete: %s", counts)


if __name__ == "__main__":
    main()
