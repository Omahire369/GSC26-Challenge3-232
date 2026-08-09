"""Route judge jobs to the S, C and Anvil models and write the Kaggle files.

Every scored row is handled by the model trained on its own cluster; the routing
key is the `<cluster>_<jid>` prefix published in sample_submission.csv, so no
example ever reaches a model from another cluster.
"""

from __future__ import annotations

import argparse
import gc
import logging
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.artifacts import ClusterArtifacts, predict_matrix
from src.config import CLUSTERS
from src.features import build_matrix, load_accumulators
from src.lifespan import lifespan_features, load_lifespan

LOGGER = logging.getLogger(__name__)


def predict_cluster(
    accumulators: Path, models: Path, cluster: str, lifespan_roots: list[Path] | None = None
) -> pd.DataFrame:
    artifacts: ClusterArtifacts = joblib.load(models / f"{cluster}_model.joblib")
    data = load_accumulators(accumulators, cluster)
    matrix = build_matrix(data, artifacts.window_seconds)
    del data
    gc.collect()
    if lifespan_roots:
        lifespan = load_lifespan(list(lifespan_roots), cluster)
        if lifespan is not None:
            matrix = pd.concat(
                [matrix, lifespan_features(matrix, lifespan, cluster)], axis=1
            )
    if matrix["jid"].duplicated().any():
        raise ValueError(f"{cluster}: duplicate job ids in the judge accumulators")
    scores = predict_matrix(artifacts, matrix)
    return pd.DataFrame(
        {
            "row_id": cluster + "_" + matrix["jid"].astype(str),
            "failure_probability": np.clip(scores.astype(np.float64), 0.0, 1.0),
        }
    )


def create_submission(
    accumulators: Path,
    models: Path,
    sample: Path,
    output: Path,
    lifespan_roots: list[Path] | None = None,
) -> Path:
    required = pd.read_csv(sample, usecols=["row_id"])
    predictions = pd.concat(
        [predict_cluster(accumulators, models, cluster, lifespan_roots) for cluster in CLUSTERS],
        ignore_index=True,
    )
    if predictions["row_id"].duplicated().any():
        raise ValueError("duplicate row_id after cluster routing")
    submission = required.merge(predictions, on="row_id", how="left", validate="one_to_one")
    missing = submission["failure_probability"].isna()
    if missing.any():
        examples = submission.loc[missing, "row_id"].head(10).tolist()
        raise ValueError(f"missing {int(missing.sum())} required predictions, e.g. {examples}")
    if len(submission) != len(required) or not submission["row_id"].equals(required["row_id"]):
        raise ValueError("submission rows must match sample_submission.csv exactly")

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "submission.csv"
    zip_path = output / "submission.zip"
    submission.to_csv(csv_path, index=False, float_format="%.8f")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.csv"]:
            raise ValueError("the Kaggle ZIP must contain exactly submission.csv")
    LOGGER.info("wrote %s with %s predictions", zip_path, f"{len(submission):,}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the FRESCO Kaggle submission.")
    parser.add_argument("--accumulators", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lifespan", type=Path, nargs="+", help="Whole-release observation spans (opt-in)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    create_submission(
        args.accumulators, args.models, args.sample, args.output, args.lifespan
    )


if __name__ == "__main__":
    main()
