"""End-to-end driver: raw FRESCO Parquet in, Kaggle submission ZIP out.

Each stage is skipped when its output already exists, so an interrupted run can
be resumed with the same command. Use `--force` to rebuild from scratch.

    python run_pipeline.py --all

Individual stages are also runnable directly (`python -m src.aggregate ...`);
this script only fixes the arguments used for the submitted result.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parent
TRAINING_SOURCE = ROOT / "training_data_bundle" / "training_data"
VALIDATION_SOURCE = ROOT / "validation_labeled_bundle"
JUDGE_SOURCE = ROOT / "unlabeled_judge_data_bundle"
SAMPLE = ROOT / "sample_submission.csv"
ACCUMULATORS = ROOT / "artifacts" / "accumulators"
LIFESPAN = ROOT / "artifacts" / "lifespan"
MODELS = ROOT / "artifacts" / "models"
SUBMISSION = ROOT / "artifacts" / "submission"

# Earliest hourly file month kept per cluster. Older months are dropped because
# the per-cluster training budget is already filled by recent data, which is
# also the data most like the scored period.
TRAINING_WINDOW = {"S": "2016-01", "C": "2015-10", "Anvil": "2022-07"}


def _run(arguments: list[str]) -> None:
    LOGGER.info("$ %s", " ".join(arguments))
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"stage failed with exit code {completed.returncode}")


def _done(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def stage_training(force: bool, workers: int) -> None:
    for cluster, from_month in TRAINING_WINDOW.items():
        output = ACCUMULATORS / "train"
        if not force and _done(output / cluster):
            LOGGER.info("train/%s accumulators already present", cluster)
            continue
        _run(
            [
                "-m", "src.aggregate",
                "--source", str(TRAINING_SOURCE),
                "--output", str(output),
                "--labeled",
                "--clusters", cluster,
                "--from-month", from_month,
                "--workers", str(workers),
                "--batch-files", "80",
            ]
        )


def _has_cluster_files(source: Path, cluster: str) -> bool:
    suffix = {"S": "_S.parquet", "C": "_C.parquet"}.get(cluster)
    for path in source.rglob("*.parquet"):
        if suffix is None:
            if not path.name.endswith(("_S.parquet", "_C.parquet")):
                return True
        elif path.name.endswith(suffix):
            return True
    return False


def stage_validation(force: bool, workers: int) -> None:
    output = ACCUMULATORS / "valid"
    if not VALIDATION_SOURCE.exists():
        LOGGER.info("no validation bundle at %s; holding out participant months instead", VALIDATION_SOURCE)
        return
    for cluster in ("S", "C", "Anvil"):
        if not force and _done(output / cluster):
            LOGGER.info("valid/%s accumulators already present", cluster)
            continue
        # The released validation chunk does not cover every cluster.
        if not _has_cluster_files(VALIDATION_SOURCE, cluster):
            LOGGER.info("validation bundle has no %s files; skipping", cluster)
            continue
        _run(
            [
                "-m", "src.aggregate",
                "--source", str(VALIDATION_SOURCE),
                "--output", str(output),
                "--labeled",
                "--clusters", cluster,
                "--workers", str(workers),
                "--batch-files", "80",
            ]
        )


def stage_judge(force: bool, workers: int) -> None:
    output = ACCUMULATORS / "judge"
    for cluster in ("S", "C", "Anvil"):
        if not force and _done(output / cluster):
            LOGGER.info("judge/%s accumulators already present", cluster)
            continue
        _run(
            [
                "-m", "src.aggregate",
                "--source", str(JUDGE_SOURCE),
                "--output", str(output),
                "--sample", str(SAMPLE),
                "--clusters", cluster,
                "--workers", str(workers),
                "--batch-files", "80",
            ]
        )


def stage_lifespan(force: bool, workers: int) -> None:
    """Whole-release observation spans. OPT-IN and outside the early-warning
    restriction; see src/lifespan.py and METHODOLOGY.md before enabling."""
    releases = [
        ("judge", JUDGE_SOURCE, {}),
        ("valid", VALIDATION_SOURCE, {}),
        ("train", TRAINING_SOURCE, TRAINING_WINDOW),
    ]
    for name, source, windows in releases:
        if not source.exists():
            continue
        for cluster in ("S", "C", "Anvil"):
            target = LIFESPAN / name / f"{cluster}.parquet"
            if not force and target.exists():
                LOGGER.info("%s/%s lifespan already present", name, cluster)
                continue
            if not _has_cluster_files(source, cluster):
                continue
            arguments = [
                "-m", "src.lifespan",
                "--source", str(source),
                "--output", str(LIFESPAN / name),
                "--clusters", cluster,
                "--workers", str(workers),
            ]
            if cluster in windows:
                arguments += ["--from-month", windows[cluster]]
            _run(arguments)


def stage_train(force: bool, lifespan: bool = False) -> None:
    if not force and (MODELS / "validation_report.json").exists():
        LOGGER.info("models already trained")
        return
    roots = [str(ACCUMULATORS / "train")]
    if _done(ACCUMULATORS / "valid"):
        roots.append(str(ACCUMULATORS / "valid"))
    arguments = ["-m", "src.train", "--accumulators", *roots, "--models", str(MODELS)]
    if lifespan:
        spans = [str(LIFESPAN / name) for name in ("train", "valid") if (LIFESPAN / name).exists()]
        if spans:
            arguments += ["--lifespan", *spans]
    if _done(ACCUMULATORS / "judge"):
        # Identity columns only, no labels: used to size the identity dropout.
        arguments += ["--reference", str(ACCUMULATORS / "judge")]
    # One cluster per subprocess keeps peak memory to a single cluster's matrix.
    for cluster in ("S", "C", "Anvil"):
        _run(arguments + ["--clusters", cluster])


def stage_predict(lifespan: bool = False) -> None:
    arguments = [
        "-m", "src.predict",
        "--accumulators", str(ACCUMULATORS / "judge"),
        "--models", str(MODELS),
        "--sample", str(SAMPLE),
        "--output", str(SUBMISSION),
    ]
    if lifespan and (LIFESPAN / "judge").exists():
        arguments += ["--lifespan", str(LIFESPAN / "judge")]
    _run(arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full FRESCO pipeline.")
    parser.add_argument("--all", action="store_true", help="run every stage")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train-acc", "valid-acc", "judge-acc", "lifespan", "fit", "predict"),
    )
    parser.add_argument(
        "--lifespan",
        action="store_true",
        help="Include whole-release runtime features. These are NOT early-window "
        "information and step outside the challenge's early-warning restriction; "
        "see METHODOLOGY.md before using.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stages = list(args.stages or [])
    if args.all or not stages:
        stages = ["train-acc", "valid-acc", "judge-acc", "fit", "predict"]
        if args.lifespan:
            stages.insert(3, "lifespan")
    for stage in stages:
        if stage == "train-acc":
            stage_training(args.force, args.workers)
        elif stage == "valid-acc":
            stage_validation(args.force, args.workers)
        elif stage == "judge-acc":
            stage_judge(args.force, args.workers)
        elif stage == "lifespan":
            stage_lifespan(args.force, args.workers)
        elif stage == "fit":
            stage_train(args.force, args.lifespan)
        elif stage == "predict":
            stage_predict(args.lifespan)
    LOGGER.info("pipeline complete")


if __name__ == "__main__":
    main()
