"""Shared constants for the FRESCO early job failure prediction pipeline.

The competition scores three clusters independently, and the three raw data
sources differ in schema and in units, so almost every constant here is either
cluster-neutral or explicitly keyed by cluster.
"""

from __future__ import annotations

from pathlib import Path

CLUSTERS: tuple[str, ...] = ("S", "C", "Anvil")

# `value_gpu` is entirely null in every released file inspected, but it is kept
# so that a judge file which does populate it still reaches the model.
METRICS: tuple[str, ...] = (
    "cpuuser",
    "gpu",
    "memused",
    "memused_minus_diskcache",
    "nfs",
    "block",
)
VALUE_COLUMNS: tuple[str, ...] = tuple(f"value_{metric}" for metric in METRICS)

STATIC_NUMERIC_COLUMNS: tuple[str, ...] = ("timelimit", "nhosts", "ncores")
META_STRING_COLUMNS: tuple[str, ...] = (
    "account",
    "queue",
    "unit",
    "jobname",
    "username",
    "host_list",
)
RAW_COLUMNS: tuple[str, ...] = (
    ("time", "submit_time", "start_time", "jid", "host")
    + STATIC_NUMERIC_COLUMNS
    + META_STRING_COLUMNS
    + VALUE_COLUMNS
)
LABEL_COLUMN = "exitcode"

# Telemetry is sampled on a wall-clock tick (600 s on S, 300 s on C, 480 s on
# Anvil) while the scheduler start time has second resolution, so a tick can
# land just before the recorded start. The allowance keeps that first reading.
PRESTART_SECONDS = 300.0

# Elapsed-second buckets relative to the job anchor. Stage 1 stores one set of
# accumulators per bucket so that Stage 2 can build a 15 or 30 minute early
# window without re-reading the raw telemetry.
PART_NAMES: tuple[str, ...] = ("p0", "p1", "p2")
PART_BOUNDS: tuple[tuple[float, float], ...] = (
    (-PRESTART_SECONDS, 300.0),
    (300.0, 900.0),
    (900.0, 1800.0),
)
MAX_WINDOW_SECONDS = PART_BOUNDS[-1][1]

# Accumulators are additive (or associative) so partial results from different
# hourly files can be merged into one job-level record.
SUM_STATS: tuple[str, ...] = ("cnt", "sum", "sq", "zero")
MIN_STATS: tuple[str, ...] = ("min",)
MAX_STATS: tuple[str, ...] = ("max",)
EDGE_STATS: tuple[str, ...] = ("first", "last")
METRIC_STATS: tuple[str, ...] = SUM_STATS + MIN_STATS + MAX_STATS + EDGE_STATS

HOST_DISPERSION_METRICS: tuple[str, ...] = ("cpuuser", "memused")

POSITIVE_OUTCOMES = frozenset({"FAILED", "TIMEOUT", "NODE_FAIL"})

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "account",
    "queue",
    "unit",
    "username",
    "jobname",
)

# Encoded with time-aware target statistics as well as label encoding, because
# user and workload identity carries most of the non-telemetry signal.
ENCODED_KEYS: tuple[tuple[str, ...], ...] = (
    ("username",),
    ("account",),
    ("queue",),
    ("jobname",),
    ("username", "queue"),
    ("account", "queue"),
    ("queue", "timelimit_bucket"),
)

# Identifier numbers (USER12717 -> 12717) were tried as features and removed: the
# numbers are issued in order of first appearance, so they encode calendar time,
# and 46-63 % of scored rows fall beyond any value seen in training. The model
# extrapolates a curve it never learned and the leaderboard score fell. See
# `reference_range_guard` in src/features.py, which now catches this class of
# feature automatically.


def cluster_for_path(path: Path | str) -> str:
    """Cluster identity is encoded in the raw Parquet file name."""
    name = Path(path).name
    if name.endswith("_S.parquet"):
        return "S"
    if name.endswith("_C.parquet"):
        return "C"
    return "Anvil"


def metric_stat_columns() -> tuple[str, ...]:
    return tuple(
        f"{metric}_{part}_{stat}"
        for metric in METRICS
        for part in PART_NAMES
        for stat in METRIC_STATS
    )


def part_shape_columns() -> tuple[str, ...]:
    return tuple(
        f"{name}_{part}"
        for part in PART_NAMES
        for name in ("rows", "hosts", "tmin", "tmax")
    )


def host_dispersion_columns() -> tuple[str, ...]:
    return tuple(
        f"hostdisp_{metric}_{stat}"
        for metric in HOST_DISPERSION_METRICS
        for stat in ("std", "min", "max", "cnt")
    )
