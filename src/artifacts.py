"""The trained-model container and the scoring path shared by both entry points.

This lives outside `src/train.py` and `src/predict.py` on purpose: a class
defined in a module executed with `python -m` is pickled as `__main__.<name>`
and cannot be loaded from a different entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.encoders import ClusterEncoder


@dataclass
class ClusterArtifacts:
    """Everything needed to score one cluster: no other cluster's state is here."""

    cluster: str
    encoder: ClusterEncoder
    feature_columns: list[str]
    models: dict[str, Any]
    window_seconds: float
    # "blend", or the name of a single model when it clearly beats the blend on
    # the out-of-time holdout.
    predictor: str = "blend"


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def blend(predictions: list[np.ndarray]) -> np.ndarray:
    """Combine models by averaging ranks, reported on a probability scale.

    Average precision depends only on the ordering within a cluster, so ranks
    combine differently calibrated models without any rescaling. The blended
    ordering is then mapped back onto the ensemble's own probability
    distribution, which is a strictly monotone transform — the ranking, and so
    the score, is unchanged — but the submitted column reads as a failure
    probability rather than a percentile.
    """
    combined = np.mean([rank(p) for p in predictions], axis=0)
    reference = np.sort(np.mean(predictions, axis=0))
    positions = combined * (len(reference) - 1)
    return np.interp(positions, np.arange(len(reference), dtype=np.float64), reference)


def design_matrix(
    matrix: pd.DataFrame, encoded: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    present = [column for column in feature_columns if column in matrix.columns]
    frame = pd.concat([matrix[present], encoded], axis=1)
    return frame.reindex(columns=feature_columns)


BLEND = "blend"


def choose_predictor(scores: dict[str, float], margin: float = 0.01) -> str:
    """Prefer the blend unless one member clearly beats it out of time.

    Blending is the lower-variance default, so a single model has to win by more
    than holdout noise before it is trusted on its own. On C, XGBoost trails
    LightGBM by 0.08 average precision and the equal-weight blend gives away
    0.03; that is well past the margin, and taking the blend anyway would be
    throwing away a measured result.
    """
    singles = {name: value for name, value in scores.items() if name != BLEND}
    if not singles:
        return BLEND
    best = max(singles, key=lambda name: singles[name])
    if BLEND not in scores or singles[best] - scores[BLEND] > margin:
        return best
    return BLEND


def family(name: str) -> str:
    """`lightgbm#2` -> `lightgbm`. Models are stored one entry per seed."""
    return name.split("#", 1)[0]


def selected_models(artifacts: ClusterArtifacts) -> list[Any]:
    choice = getattr(artifacts, "predictor", BLEND)
    if choice == BLEND:
        return list(artifacts.models.values())
    chosen = [m for name, m in artifacts.models.items() if family(name) == choice]
    return chosen or list(artifacts.models.values())


def predict_matrix(artifacts: ClusterArtifacts, matrix: pd.DataFrame) -> np.ndarray:
    encoded = pd.concat(
        [
            artifacts.encoder.transform_target_statistics(matrix),
            artifacts.encoder.transform_categories(matrix),
        ],
        axis=1,
    )
    design = design_matrix(matrix, encoded, artifacts.feature_columns)
    chosen = selected_models(artifacts)
    return blend([model.predict_proba(design)[:, 1] for model in chosen])
