"""Time-aware categorical encoding for the cluster models.

Job identity fields (user, account, queue, workload name) carry most of the
non-telemetry signal, but they are high cardinality and their failure rates
drift. Two encodings are produced for each key:

* a smoothed historical failure rate;
* a log frequency, which lets the model discount thinly observed categories.

For training rows the statistics come only from strictly earlier months. That
mirrors inference, where every labelled month precedes the scored period, and it
keeps a job's own outcome out of its own encoded value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, ENCODED_KEYS

MISSING = "__missing__"


def _key_series(matrix: pd.DataFrame, key: tuple[str, ...]) -> pd.Series:
    parts = [matrix[name].astype("string").fillna(MISSING) for name in key]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="\x1f")
    return joined.astype("string")


def _key_name(key: tuple[str, ...]) -> str:
    return "__".join(key)


@dataclass
class ClusterEncoder:
    """Label codes plus historical target statistics for one cluster."""

    smoothing: float = 25.0
    min_frequency: int = 5
    prior: float = 0.0
    categories: dict[str, dict[str, int]] = field(default_factory=dict)
    target_sum: dict[str, pd.Series] = field(default_factory=dict)
    target_count: dict[str, pd.Series] = field(default_factory=dict)

    # ---- label encoding ----------------------------------------------------
    def fit_categories(self, matrix: pd.DataFrame) -> None:
        for name in CATEGORICAL_FEATURES:
            values = matrix[name].astype("string").fillna(MISSING)
            counts = values.value_counts()
            kept = counts.index[counts >= self.min_frequency]
            self.categories[name] = {category: code for code, category in enumerate(kept, start=1)}

    def transform_categories(self, matrix: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=matrix.index)
        for name in CATEGORICAL_FEATURES:
            values = matrix[name].astype("string").fillna(MISSING)
            codes = values.map(self.categories[name]).fillna(0).astype(np.int32)
            out[f"cat_{name}"] = codes
        return out

    # ---- historical target statistics --------------------------------------
    def fit_target_statistics(self, matrix: pd.DataFrame, target: pd.Series) -> None:
        self.prior = float(target.mean())
        for key in ENCODED_KEYS:
            name = _key_name(key)
            keys = _key_series(matrix, key)
            frame = pd.DataFrame({"key": keys.to_numpy(), "target": target.to_numpy()})
            grouped = frame.groupby("key", observed=True)["target"]
            self.target_sum[name] = grouped.sum()
            self.target_count[name] = grouped.count()

    def transform_target_statistics(self, matrix: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=matrix.index)
        for key in ENCODED_KEYS:
            name = _key_name(key)
            keys = _key_series(matrix, key)
            total = keys.map(self.target_sum[name]).astype(np.float64).fillna(0.0)
            count = keys.map(self.target_count[name]).astype(np.float64).fillna(0.0)
            out[f"te_{name}"] = (
                (total + self.smoothing * self.prior) / (count + self.smoothing)
            ).astype(np.float32)
            out[f"tc_{name}"] = np.log1p(count).astype(np.float32)
        return out

    def fit_expanding(self, matrix: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Encode training rows using only months strictly before their own.

        The earliest month has no history and falls back to the global prior,
        which is the same behaviour a genuinely unseen category gets at
        inference time.
        """
        self.prior = float(target.mean())
        months = matrix["month"].astype("string").fillna("unknown")
        ordered_months = sorted(m for m in months.unique() if m != "unknown")
        out = pd.DataFrame(index=matrix.index)
        for key in ENCODED_KEYS:
            name = _key_name(key)
            keys = _key_series(matrix, key)
            frame = pd.DataFrame(
                {"key": keys.to_numpy(), "month": months.to_numpy(), "target": target.to_numpy()}
            )
            monthly = frame.groupby(["month", "key"], observed=True)["target"].agg(["sum", "count"])
            encoded = np.full(len(matrix), self.prior, dtype=np.float32)
            counts = np.zeros(len(matrix), dtype=np.float32)
            history_sum: pd.Series = pd.Series(dtype=np.float64)
            history_count: pd.Series = pd.Series(dtype=np.float64)
            for month in ordered_months:
                # `months` is a nullable string Series, so the comparison is a
                # nullable boolean; numpy needs a plain bool array to index with.
                mask = (months == month).fillna(False).to_numpy(dtype=bool)
                if mask.any() and len(history_count) > 0:
                    subset = keys[mask]
                    total = subset.map(history_sum).astype(np.float64).fillna(0.0).to_numpy()
                    seen = subset.map(history_count).astype(np.float64).fillna(0.0).to_numpy()
                    encoded[mask] = (
                        (total + self.smoothing * self.prior) / (seen + self.smoothing)
                    ).astype(np.float32)
                    counts[mask] = seen.astype(np.float32)
                if month in monthly.index.get_level_values(0):
                    block = monthly.loc[month]
                    history_sum = history_sum.add(block["sum"], fill_value=0.0)
                    history_count = history_count.add(block["count"], fill_value=0.0)
            out[f"te_{name}"] = encoded
            out[f"tc_{name}"] = np.log1p(counts).astype(np.float32)
        # The saved encoder must see every labelled month, since inference rows
        # come after all of them.
        self.fit_target_statistics(matrix, target)
        return out


def encoded_column_names() -> list[str]:
    names: list[str] = []
    for key in ENCODED_KEYS:
        name = _key_name(key)
        names.extend([f"te_{name}", f"tc_{name}"])
    names.extend(f"cat_{name}" for name in CATEGORICAL_FEATURES)
    return names
