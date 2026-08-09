"""Make the models survive identities they have never seen.

User, account and workload names carry most of the non-telemetry signal, and a
gradient-boosted model will happily spend the bulk of its capacity memorising
them. That capacity is worthless on a job whose submitter never appears in the
training months, and the scored chunk is full of those: on Anvil only 67 % of
judge jobs have a `username` seen in training and only 13 % have a known
`jobname`, against 95 % and 44 % on the adjacent validation chunk.

Two corrections, both driven by rates measured from the released *unlabelled*
judge features (no labels are read):

* the holdout is degraded to the same coverage as the scored chunk, so round
  selection and configuration choice optimise for the regime that is actually
  scored rather than the easier adjacent-month one;
* a fraction of training rows have their identity columns blanked, which forces
  the model to keep a usable fallback in telemetry and requested resources.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

# Field -> the encoded columns that become uninformative when it is unseen.
# `username__jobname_stem` belongs to both groups: masking either field kills it.
IDENTITY_FIELDS: tuple[str, ...] = ("username", "jobname", "account")
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "username": ("username",),
    "jobname": ("jobname", "jobname_stem"),
    "account": ("account",),
}
UNSEEN_CATEGORY_CODE = 0


def columns_for_field(field: str, columns: list[str]) -> list[str]:
    aliases = _FIELD_ALIASES[field]
    selected = []
    for column in columns:
        if not column.startswith(("te_", "tc_", "cat_")):
            continue
        body = column.split("_", 1)[1]
        parts = body.split("__")
        if any(alias in parts for alias in aliases):
            selected.append(column)
    return selected


def read_identities(root: Path, cluster: str) -> pd.DataFrame:
    """Just the identity columns of a Stage 1 accumulator directory."""
    directory = root / cluster
    frames = []
    for path in sorted(directory.glob("*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        wanted = [f for f in IDENTITY_FIELDS if f in available]
        if not wanted:
            continue
        frames.append(pq.read_table(path, columns=wanted).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no accumulators for {cluster} under {root}")
    return pd.concat(frames, ignore_index=True)


def measure_unseen_rates(
    training: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, float]:
    """Fraction of reference rows whose identity never occurs in training."""
    rates: dict[str, float] = {}
    for field in IDENTITY_FIELDS:
        if field not in training.columns or field not in reference.columns:
            rates[field] = 0.0
            continue
        seen = set(training[field].dropna().astype(str))
        values = reference[field].astype(str)
        rates[field] = float(1.0 - values.isin(seen).mean())
    return rates


def category_holdout_mask(values: pd.Series, rate: float, seed: int) -> np.ndarray:
    """Rows belonging to whole categories chosen to be 'unseen'.

    Unseen-ness is a property of the category, not of the row, so entire
    categories are withdrawn until the requested share of rows is covered.
    """
    count = len(values)
    mask = np.zeros(count, dtype=bool)
    if rate <= 0 or count == 0:
        return mask
    text = values.astype(str)
    sizes = text.value_counts()
    generator = np.random.default_rng(seed)
    order = generator.permutation(sizes.index.to_numpy())
    budget = rate * count
    chosen: list[str] = []
    running = 0
    for category in order:
        if running >= budget:
            break
        chosen.append(category)
        running += int(sizes[category])
    if not chosen:
        return mask
    return text.isin(set(chosen)).to_numpy()


def row_dropout_mask(count: int, rate: float, seed: int) -> np.ndarray:
    if rate <= 0 or count == 0:
        return np.zeros(count, dtype=bool)
    return np.random.default_rng(seed).random(count) < rate


def blank_identities(
    encoded: pd.DataFrame, masks: dict[str, np.ndarray], prior: float
) -> pd.DataFrame:
    """Give masked rows exactly what an unseen category gets at inference time.

    Target statistics collapse to the global prior, observed frequency to zero
    and the label code to the unknown bucket, which is what the encoders emit
    for a category they have never recorded.
    """
    out = encoded.copy()
    columns = list(out.columns)
    for field, mask in masks.items():
        if not mask.any():
            continue
        for column in columns_for_field(field, columns):
            if column.startswith("te_"):
                out.loc[mask, column] = np.float32(prior)
            elif column.startswith("tc_"):
                out.loc[mask, column] = np.float32(0.0)
            else:
                out.loc[mask, column] = np.int32(UNSEEN_CATEGORY_CODE)
    return out


def holdout_masks(matrix: pd.DataFrame, rates: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    return {
        field: category_holdout_mask(matrix[field], rates.get(field, 0.0), seed + index)
        for index, field in enumerate(IDENTITY_FIELDS)
        if field in matrix.columns
    }


def training_masks(count: int, rates: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    return {
        field: row_dropout_mask(count, rates.get(field, 0.0), seed + 100 + index)
        for index, field in enumerate(IDENTITY_FIELDS)
    }
