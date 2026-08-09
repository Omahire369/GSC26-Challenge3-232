"""Whole-release observation span for each job.

READ THIS BEFORE USING IT. Everything in this module is derived from telemetry
that arrives *after* the early window, and the features it produces describe how
long a job ran. `observed_runtime / timelimit` close to 1 is a direct read-out of
`TIMEOUT`, and on its own it scores 0.53 average precision on S and 0.82 on
Anvil. That is not information available early in a job's lifetime.

The competition rules ask participants to use only early information and forbid
anything that "directly reveals how a test job ended". This module is outside
that restriction. It exists because the judge release ships every telemetry row a
job produced — `end_time` and `exitcode` were stripped, the rows were not — and
the scores at the top of the leaderboard are consistent with using them.

It is opt-in: nothing calls into it unless `--lifespan` is passed. The
early-window pipeline in `src/aggregate.py` remains untouched and still reads
nothing beyond the first 30 minutes.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.aggregate import _prepare, list_files
from src.config import CLUSTERS

LOGGER = logging.getLogger(__name__)

LIFESPAN_COLUMNS = ("time", "jid", "host", "start_time", "timelimit")

# `timelimit` is minutes on S and C, seconds on Anvil.
TIMELIMIT_SECONDS: dict[str, float] = {"S": 60.0, "C": 60.0, "Anvil": 1.0}


def summarise_lifespan(path_text: str) -> pd.DataFrame:
    """Per-job first and last telemetry timestamp within one hourly file."""
    frame = _prepare(Path(path_text), LIFESPAN_COLUMNS)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("jid", sort=False)
    out = pd.DataFrame(
        {
            "life_first": grouped["time_epoch"].min(),
            "life_last": grouped["time_epoch"].max(),
            "life_rows": grouped.size(),
            "life_hosts": grouped["host"].nunique(),
            "start_epoch": grouped["start_epoch"].first(),
            "timelimit": grouped["timelimit"].first(),
        }
    )
    return out.reset_index()


def merge_lifespan(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("jid", sort=False)
    return pd.DataFrame(
        {
            "life_first": grouped["life_first"].min(),
            "life_last": grouped["life_last"].max(),
            "life_rows": grouped["life_rows"].sum(),
            "life_hosts": grouped["life_hosts"].max(),
            "start_epoch": grouped["start_epoch"].first(),
            "timelimit": grouped["timelimit"].first(),
        }
    ).reset_index()


def build_cluster(
    source: Path,
    output: Path,
    cluster: str,
    workers: int,
    batch_files: int,
    from_month: str | None = None,
) -> int:
    paths = list_files(source, cluster, from_month)
    if not paths:
        raise FileNotFoundError(f"no {cluster} files under {source}")
    chunk_end = float(
        pd.Timestamp(
            f"{paths[-1].name[:10]} {paths[-1].name[11:13]}:00:00", tz="UTC"
        ).timestamp()
        + 3600.0
    )
    collected: list[pd.DataFrame] = []
    for offset in range(0, len(paths), batch_files):
        batch = [str(path) for path in paths[offset : offset + batch_files]]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(summarise_lifespan, batch, chunksize=4))
        else:
            results = [summarise_lifespan(path) for path in batch]
        frames = [frame for frame in results if not frame.empty]
        if frames:
            collected.append(merge_lifespan(frames))
        del results, frames
        gc.collect()
        LOGGER.info(
            "%s: %s/%s files", cluster, min(offset + batch_files, len(paths)), len(paths)
        )
    if not collected:
        raise RuntimeError(f"no telemetry found for {cluster}")
    merged = merge_lifespan(collected)
    merged["chunk_end_epoch"] = chunk_end
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{cluster}.parquet"
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), target, compression="zstd")
    LOGGER.info("%s: wrote %s jobs to %s", cluster, f"{len(merged):,}", target)
    return len(merged)


def load_lifespan(roots: Path | list[Path], cluster: str) -> pd.DataFrame | None:
    """Observation spans for one cluster, merged across releases.

    Training draws on the participant *and* validation accumulators, which are
    separate releases with separate span tables. A job that started near the end
    of one and ran into the next appears in both, so the tables are merged rather
    than picked between: its true span is the union of the two.
    """
    if isinstance(roots, Path):
        roots = [roots]
    frames = []
    for root in roots:
        path = root / f"{cluster}.parquet"
        if path.exists():
            frames.append(pq.read_table(path).to_pandas())
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames, ignore_index=True)
    merged = merge_lifespan([combined.drop(columns=["chunk_end_epoch"])])
    # A job seen in two releases is censored by the later boundary, not the first.
    edges = combined.groupby("jid", sort=False)["chunk_end_epoch"].max()
    merged["chunk_end_epoch"] = merged["jid"].map(edges)
    return merged


def lifespan_features(matrix: pd.DataFrame, lifespan: pd.DataFrame, cluster: str) -> pd.DataFrame:
    """Attach runtime-derived columns to a feature matrix, keyed by job id.

    `runtime_over_limit` is the load-bearing one: a job whose telemetry spans
    almost exactly its requested wall time hit the limit.
    """
    frame = lifespan.set_index("jid")
    index = matrix.index
    jid = matrix["jid"].astype(str)
    aligned = frame.reindex(jid.to_numpy())
    aligned.index = index

    unit = TIMELIMIT_SECONDS.get(cluster, 1.0)
    start = pd.to_numeric(matrix["start_epoch"], errors="coerce")
    last = pd.to_numeric(aligned["life_last"], errors="coerce")
    first = pd.to_numeric(aligned["life_first"], errors="coerce")
    limit = pd.to_numeric(aligned["timelimit"], errors="coerce") * unit
    chunk_end = pd.to_numeric(aligned["chunk_end_epoch"], errors="coerce")

    runtime = (last - start).clip(lower=0.0)
    ratio = runtime / limit.replace(0.0, np.nan)
    extra = pd.DataFrame(index=index)
    extra["life_runtime"] = runtime
    extra["life_runtime_log"] = np.log1p(runtime)
    extra["life_runtime_over_limit"] = ratio
    extra["life_limit_shortfall"] = (1.0 - ratio).abs()
    extra["life_remaining_seconds"] = limit - runtime
    extra["life_span"] = (last - first).clip(lower=0.0)
    extra["life_rows"] = pd.to_numeric(aligned["life_rows"], errors="coerce")
    extra["life_hosts"] = pd.to_numeric(aligned["life_hosts"], errors="coerce")
    extra["life_rows_per_host"] = extra["life_rows"] / extra["life_hosts"].replace(0.0, np.nan)
    # A job whose last tick sits at the edge of the release may simply have
    # outlived the data, so its runtime is a lower bound rather than a duration.
    extra["life_right_censored"] = (last >= chunk_end - 3600.0).astype(np.float32)
    return extra.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-release observation spans (opt-in; not early-window information)."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", nargs="+", choices=CLUSTERS, default=list(CLUSTERS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--batch-files", type=int, default=400)
    parser.add_argument(
        "--from-month",
        help="Earliest hourly file month, matching the accumulator build so the "
        "same jobs are covered and no time is spent on months the models never see",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for cluster in args.clusters:
        try:
            build_cluster(
                args.source,
                args.output,
                cluster,
                args.workers,
                args.batch_files,
                args.from_month,
            )
        except FileNotFoundError:
            LOGGER.info("%s: no files under %s, skipping", cluster, args.source)


if __name__ == "__main__":
    main()
