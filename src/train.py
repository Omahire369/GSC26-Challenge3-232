"""Train one gradient-boosted model per cluster and score it out of time.

Three independent models are produced, one for S, one for C and one for Anvil.
Nothing is pooled: each cluster has its own encoder, its own feature selection
and its own boosting rounds, and inference routes on the submission row prefix.

Validation always holds out the latest labelled months, because the scored judge
chunk sits after every labelled month the pipeline can see.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.artifacts import BLEND, ClusterArtifacts, blend, choose_predictor, design_matrix
from src.config import CATEGORICAL_FEATURES, CLUSTERS
from src.drift import (
    blank_identities,
    holdout_masks,
    measure_unseen_rates,
    read_identities,
    training_masks,
)
from src.encoders import ClusterEncoder
from src.lifespan import lifespan_features, load_lifespan
from src.features import (
    build_matrix,
    deduplicate_jobs,
    drop_constant_columns,
    load_accumulators,
    numeric_feature_columns,
    reference_range_guard,
)

LOGGER = logging.getLogger(__name__)

# Recency-ordered cap per cluster. Sized for an 8 GB workstation; raise it with
# `--max-rows` when more memory is available.
DEFAULT_MAX_ROWS = {"S": 420_000, "C": 420_000, "Anvil": 420_000}
DEFAULT_MIN_MONTH = {"S": None, "C": None, "Anvil": None}


def _lightgbm_model(rounds: int, seed: int, imbalance: float):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        learning_rate=0.04,
        num_leaves=96,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.1,
        max_bin=127,
        cat_smooth=20.0,
        min_data_per_group=50,
        max_cat_threshold=48,
        scale_pos_weight=imbalance,
        n_jobs=6,
        random_state=seed,
        verbosity=-1,
    )


def _xgboost_model(rounds: int, seed: int, imbalance: float, early_stopping: int | None):
    from xgboost import XGBClassifier

    parameters: dict[str, Any] = {
        "n_estimators": rounds,
        "learning_rate": 0.045,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_lambda": 6.0,
        "reg_alpha": 0.1,
        "max_bin": 128,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "scale_pos_weight": imbalance,
        "n_jobs": 6,
        "random_state": seed,
    }
    if early_stopping is not None:
        parameters["early_stopping_rounds"] = early_stopping
    return XGBClassifier(**parameters)


def _positive_weight(target: pd.Series) -> float:
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if positives <= 0:
        raise ValueError("training split contains no failures")
    return float(min(6.0, max(1.0, np.sqrt(negatives / positives))))


def _fit_pair(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    rounds: dict[str, int] | None,
    seed: int,
    models_wanted: tuple[str, ...] = ("lightgbm", "xgboost"),
) -> tuple[dict[str, Any], dict[str, int], dict[str, float]]:
    """Fit LightGBM and XGBoost, early stopping when a validation split exists."""
    import lightgbm as lgb

    imbalance = _positive_weight(y_train)
    # Generous patience on purpose. With a short fuse these models sometimes
    # stop after 7-17 rounds on a brief plateau and land 0.15 average precision
    # below the same configuration that was allowed to continue.
    patience = 300
    categorical = [f"cat_{name}" for name in CATEGORICAL_FEATURES if f"cat_{name}" in x_train.columns]
    models: dict[str, Any] = {}
    best_rounds: dict[str, int] = {}
    scores: dict[str, float] = {}

    if "lightgbm" not in models_wanted:
        return _fit_xgboost_only(
            x_train, y_train, x_valid, y_valid, rounds, seed, imbalance, models, best_rounds, scores
        )

    lgb_rounds = 4000 if rounds is None else rounds["lightgbm"]
    lightgbm_model = _lightgbm_model(lgb_rounds, seed, imbalance)
    if x_valid is not None:
        lightgbm_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="average_precision",
            categorical_feature=categorical,
            callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)],
        )
        best_rounds["lightgbm"] = int(lightgbm_model.best_iteration_ or lgb_rounds)
        scores["lightgbm"] = float(
            average_precision_score(y_valid, lightgbm_model.predict_proba(x_valid)[:, 1])
        )
    else:
        lightgbm_model.fit(x_train, y_train, categorical_feature=categorical)
        best_rounds["lightgbm"] = lgb_rounds
    models["lightgbm"] = lightgbm_model
    if "xgboost" not in models_wanted:
        return models, best_rounds, scores

    xgb_rounds = 3000 if rounds is None else rounds["xgboost"]
    xgboost_model = _xgboost_model(xgb_rounds, seed, imbalance, patience if x_valid is not None else None)
    if x_valid is not None:
        xgboost_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        best_rounds["xgboost"] = int(getattr(xgboost_model, "best_iteration", xgb_rounds - 1)) + 1
        scores["xgboost"] = float(
            average_precision_score(y_valid, xgboost_model.predict_proba(x_valid)[:, 1])
        )
    else:
        xgboost_model.fit(x_train, y_train, verbose=False)
        best_rounds["xgboost"] = xgb_rounds
    models["xgboost"] = xgboost_model

    if x_valid is not None:
        combined = blend(
            [models[name].predict_proba(x_valid)[:, 1] for name in ("lightgbm", "xgboost")]
        )
        scores["blend"] = float(average_precision_score(y_valid, combined))
    return models, best_rounds, scores


def _fit_xgboost_only(
    x_train, y_train, x_valid, y_valid, rounds, seed, imbalance, models, best_rounds, scores
):
    xgb_rounds = 3000 if rounds is None else rounds["xgboost"]
    model = _xgboost_model(xgb_rounds, seed, imbalance, 300 if x_valid is not None else None)
    if x_valid is not None:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        best_rounds["xgboost"] = int(getattr(model, "best_iteration", xgb_rounds - 1)) + 1
        scores["xgboost"] = float(
            average_precision_score(y_valid, model.predict_proba(x_valid)[:, 1])
        )
    else:
        model.fit(x_train, y_train, verbose=False)
        best_rounds["xgboost"] = xgb_rounds
    models["xgboost"] = model
    return models, best_rounds, scores


def _top_features(model: Any, feature_columns: list[str], top: int = 25) -> list[list[Any]]:
    """Gain-ranked features, recorded so a reviewer can check what drives each
    cluster model and confirm nothing outcome-derived is in there."""
    if model is None or not hasattr(model, "booster_"):
        return []
    gains = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
    total = float(gains.sum()) or 1.0
    order = np.argsort(-gains)[:top]
    return [[feature_columns[i], round(float(gains[i]) / total, 5)] for i in order]


def _identity_design(
    matrix: pd.DataFrame,
    encoded: pd.DataFrame,
    columns: list[str],
    masks: dict[str, np.ndarray] | None,
    prior: float,
) -> pd.DataFrame:
    if masks:
        encoded = blank_identities(encoded, masks, prior)
    return design_matrix(matrix, encoded, columns)


def _select_identity_dropout(
    train_matrix: pd.DataFrame,
    encoded_train: pd.DataFrame,
    feature_columns: list[str],
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    x_valid_natural: pd.DataFrame,
    y_valid: pd.Series,
    unseen_rates: dict[str, float],
    mode: str,
    prior: float,
    seed: int,
    cluster: str,
) -> tuple[bool, dict[str, int], dict[str, float], dict[str, float]]:
    """Decide, per cluster, whether to blank identities on some training rows.

    A single LightGBM fit per option settles it, because that is where the
    difference shows: with identities always present the model spends its
    capacity memorising them and stops improving within a few rounds, and that
    memory is worth nothing for a job whose submitter is new.
    """
    options: list[bool] = []
    if mode == "off" or not unseen_rates:
        options = [False]
    elif mode == "on":
        options = [True]
    else:
        options = [False, True]

    chosen = options[0]
    if len(options) > 1:
        probe: dict[bool, float] = {}
        for use_dropout in options:
            masks = training_masks(len(train_matrix), unseen_rates, seed) if use_dropout else None
            x_train = _identity_design(
                train_matrix, encoded_train, feature_columns, masks, prior
            )
            _, _, single = _fit_pair(
                x_train, y_train, x_valid, y_valid, None, seed, models_wanted=("lightgbm",)
            )
            probe[use_dropout] = single["lightgbm"]
            del x_train
            gc.collect()
        chosen = max(probe, key=lambda key: probe[key])
        LOGGER.info(
            "%s: identity-dropout probe %s -> %s",
            cluster,
            {("on" if k else "off"): round(v, 5) for k, v in probe.items()},
            "on" if chosen else "off",
        )

    masks = training_masks(len(train_matrix), unseen_rates, seed) if chosen else None
    x_train = _identity_design(train_matrix, encoded_train, feature_columns, masks, prior)
    models, best_rounds, scores = _fit_pair(x_train, y_train, x_valid, y_valid, None, seed)
    natural_scores = {
        name: float(average_precision_score(y_valid, model.predict_proba(x_valid_natural)[:, 1]))
        for name, model in models.items()
    }
    natural_scores["blend"] = float(
        average_precision_score(
            y_valid,
            blend([model.predict_proba(x_valid_natural)[:, 1] for model in models.values()]),
        )
    )
    del x_train, models
    gc.collect()
    return chosen, best_rounds, scores, natural_scores


def labelled_matrix(
    accumulators: list[Path],
    cluster: str,
    window_seconds: float,
    max_rows: int | None,
    min_month: str | None,
    lifespan_roots: list[Path] | None = None,
) -> pd.DataFrame:
    """Every labelled job for one cluster, newest first when a cap applies.

    A root without files for this cluster is skipped: the released validation
    chunk covers S and Anvil but not C.
    """
    frames = []
    for root in accumulators:
        try:
            frames.append(load_accumulators(root, cluster, min_month=min_month, max_rows=None))
        except FileNotFoundError:
            LOGGER.info("%s: no accumulators under %s, skipping", cluster, root)
    if not frames:
        raise FileNotFoundError(f"no accumulators for cluster {cluster} in {accumulators}")
    data = deduplicate_jobs(pd.concat(frames, ignore_index=True)).reset_index(drop=True)
    del frames
    gc.collect()
    matrix = build_matrix(data, window_seconds)
    del data
    gc.collect()
    matrix = _attach_lifespan(matrix, lifespan_roots, cluster)
    matrix = matrix.loc[matrix["target"].notna() & matrix["month"].notna()].reset_index(drop=True)
    if max_rows is not None and len(matrix) > max_rows:
        keep = np.argsort(-matrix["start_epoch"].fillna(-1.0).to_numpy(), kind="stable")[:max_rows]
        matrix = matrix.iloc[np.sort(keep)].reset_index(drop=True)
    gc.collect()
    return matrix


def train_cluster(
    accumulators: list[Path],
    cluster: str,
    holdout_months: int,
    window_seconds: float,
    max_rows: int | None,
    min_month: str | None,
    seed: int,
    max_holdout_rows: int | None = 120_000,
    reference_identities: pd.DataFrame | None = None,
    identity_dropout: str = "auto",
    seeds: int = 3,
    reference_matrix: pd.DataFrame | None = None,
    lifespan_roots: list[Path] | None = None,
) -> tuple[ClusterArtifacts, dict[str, Any]]:
    matrix = labelled_matrix(
        accumulators, cluster, window_seconds, max_rows, min_month, lifespan_roots
    )
    # Measured against the whole capped matrix, because that is the vocabulary
    # the *deployed* model is refit on and therefore the coverage it will
    # actually meet on the scored chunk. Measuring against the smaller
    # validation split instead overstates the loss and, on C, pushes early
    # stopping to a far worse model.
    unseen_rates = (
        measure_unseen_rates(matrix, reference_identities)
        if reference_identities is not None
        else {}
    )
    months = sorted(matrix["month"].unique())
    if len(months) <= holdout_months:
        raise ValueError(f"{cluster}: only {len(months)} labelled months available")
    cutoff = months[-holdout_months]
    is_valid = (matrix["month"] >= cutoff).fillna(False).to_numpy(dtype=bool)
    LOGGER.info(
        "%s: %s jobs over %s months, holdout from %s (%s jobs)",
        cluster,
        f"{len(matrix):,}",
        len(months),
        cutoff,
        f"{int(is_valid.sum()):,}",
    )

    candidates = drop_constant_columns(matrix, numeric_feature_columns(matrix))
    if reference_matrix is not None:
        candidates, extrapolating = reference_range_guard(matrix, reference_matrix, candidates)
        if extrapolating:
            LOGGER.warning(
                "%s: dropped %s feature(s) that mostly fall outside the training "
                "range on the scored chunk: %s",
                cluster,
                len(extrapolating),
                extrapolating,
            )
    total_rows = len(matrix)
    train_rows = int((~is_valid).sum())
    positive_rate = float(matrix["target"].mean())
    positive_rate_holdout = float(matrix.loc[is_valid, "target"].mean())
    holdout_rows = int(is_valid.sum())

    # ---- out-of-time validation -------------------------------------------
    # `matrix` is released before fitting and reloaded for the refit: holding
    # the full matrix, both splits and both design matrices at once is what
    # pushes a modest workstation into swap.
    train_matrix = matrix.loc[~is_valid].reset_index(drop=True)
    valid_matrix = matrix.loc[is_valid].reset_index(drop=True)
    del matrix
    gc.collect()
    if max_holdout_rows and len(valid_matrix) > max_holdout_rows:
        # The holdout only has to estimate average precision and pick a round
        # count; a uniform subsample keeps the class balance and the memory down.
        valid_matrix = valid_matrix.sample(
            n=max_holdout_rows, random_state=seed
        ).sort_index().reset_index(drop=True)
    y_train = train_matrix["target"].astype(np.int8)
    y_valid = valid_matrix["target"].astype(np.int8)
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        raise ValueError(f"{cluster}: holdout split needs both classes")

    if unseen_rates:
        LOGGER.info(
            "%s: judge identities unseen by the deployed vocabulary %s",
            cluster,
            {k: round(v, 3) for k, v in unseen_rates.items()},
        )

    scoring_encoder = ClusterEncoder()
    scoring_encoder.fit_categories(train_matrix)
    encoded_train = pd.concat(
        [
            scoring_encoder.fit_expanding(train_matrix, y_train),
            scoring_encoder.transform_categories(train_matrix),
        ],
        axis=1,
    )
    feature_columns = candidates + list(encoded_train.columns)
    encoded_valid = pd.concat(
        [
            scoring_encoder.transform_target_statistics(valid_matrix),
            scoring_encoder.transform_categories(valid_matrix),
        ],
        axis=1,
    )
    prior = scoring_encoder.prior

    # The holdout sits next to the training months and shares most of their
    # identities; the scored chunk does not. Scoring against a copy degraded to
    # the scored chunk's own coverage is what makes the round count and the
    # dropout decision reflect the regime that is actually graded.
    x_valid_natural = design_matrix(valid_matrix, encoded_valid, feature_columns)
    if unseen_rates:
        masks = holdout_masks(valid_matrix, unseen_rates, seed)
        masked_share = {field: round(float(m.mean()), 3) for field, m in masks.items()}
        LOGGER.info("%s: holdout degraded to judge coverage %s", cluster, masked_share)
        x_valid = design_matrix(
            valid_matrix, blank_identities(encoded_valid, masks, prior), feature_columns
        )
    else:
        masked_share = {}
        x_valid = x_valid_natural
    del valid_matrix, encoded_valid
    gc.collect()

    dropout_used, best_rounds, scores, natural_scores = _select_identity_dropout(
        train_matrix,
        encoded_train,
        feature_columns,
        y_train,
        x_valid,
        x_valid_natural,
        y_valid,
        unseen_rates,
        identity_dropout,
        prior,
        seed,
        cluster,
    )
    predictor = choose_predictor(scores)
    LOGGER.info(
        "%s holdout AP (judge-like): %s | natural: %s | dropout=%s predictor=%s",
        cluster,
        {k: round(v, 5) for k, v in scores.items()},
        {k: round(v, 5) for k, v in natural_scores.items()},
        dropout_used,
        predictor,
    )
    del train_matrix, encoded_train, x_valid, x_valid_natural, y_train, y_valid, scoring_encoder
    gc.collect()

    # ---- refit on every labelled month ------------------------------------
    matrix = labelled_matrix(
        accumulators, cluster, window_seconds, max_rows, min_month, lifespan_roots
    )
    y_all = matrix["target"].astype(np.int8)
    final_encoder = ClusterEncoder()
    final_encoder.fit_categories(matrix)
    encoded_all = pd.concat(
        [
            final_encoder.fit_expanding(matrix, y_all),
            final_encoder.transform_categories(matrix),
        ],
        axis=1,
    )
    x_base = design_matrix(matrix, encoded_all, feature_columns)
    rows_all = len(matrix)
    final_prior = final_encoder.prior
    del matrix, encoded_all
    gc.collect()
    # More rows than the validation fit saw, so allow proportionally more trees.
    # The lower bound is deliberately small: when the out-of-time holdout says a
    # cluster peaks after a handful of rounds, overriding that with a larger
    # floor just refits the training period the holdout warned about.
    growth = max(1.0, total_rows / max(train_rows, 1))
    final_rounds = {
        name: int(max(10, min(4000, round(rounds * min(growth, 1.4)))))
        for name, rounds in best_rounds.items()
    }
    # Only the models the chosen predictor actually uses are refit, once per
    # seed. A single fit is a lottery ticket on C, where the same configuration
    # spans 0.41 to 0.52 average precision across seeds; averaging the ranks of
    # several draws is worth more than any single one of them.
    wanted = ("lightgbm", "xgboost") if predictor == BLEND else (predictor,)
    models: dict[str, Any] = {}
    for index in range(max(1, seeds)):
        seed_for_fit = seed + 1000 * index
        if dropout_used:
            # The mask columns live in the design matrix under the same names,
            # so each seed's draw is applied there rather than rebuilding it.
            masks = training_masks(rows_all, unseen_rates, seed_for_fit + 7)
            x_seed = blank_identities(x_base, masks, final_prior)
        else:
            x_seed = x_base
        fitted, _, _ = _fit_pair(
            x_seed, y_all, None, None, final_rounds, seed_for_fit, models_wanted=wanted
        )
        for name, model in fitted.items():
            models[f"{name}#{index}"] = model
        if x_seed is not x_base:
            del x_seed
        gc.collect()
        LOGGER.info("%s: refit %s/%s seeds", cluster, index + 1, max(1, seeds))
    importances = _top_features(models.get("lightgbm#0"), feature_columns)
    del x_base
    gc.collect()

    report = {
        "cluster": cluster,
        "jobs": total_rows,
        "months": len(months),
        "holdout_from": cutoff,
        "holdout_jobs": holdout_rows,
        "positive_rate": positive_rate,
        "positive_rate_holdout": positive_rate_holdout,
        "holdout_average_precision": scores,
        "holdout_average_precision_natural": natural_scores,
        "identity_dropout": dropout_used,
        "predictor": predictor,
        "judge_unseen_rates": {k: round(v, 4) for k, v in unseen_rates.items()},
        "holdout_masked_share": masked_share,
        "validation_rounds": best_rounds,
        "final_rounds": final_rounds,
        "n_features": len(feature_columns),
        "dropped_for_extrapolation": extrapolating if reference_matrix is not None else [],
        "refit_seeds": max(1, seeds),
        "top_features": importances,
    }
    artifacts = ClusterArtifacts(
        cluster, final_encoder, feature_columns, models, window_seconds, predictor
    )
    return artifacts, report


def _reference_identities(reference: Path | None, cluster: str) -> pd.DataFrame | None:
    """Identity columns of the unlabelled judge accumulators.

    No outcome field is touched, so the unseen rates derived from this are a
    property of the released features, not of the hidden labels.
    """
    if reference is None:
        return None
    try:
        return read_identities(reference, cluster)
    except FileNotFoundError:
        LOGGER.info("%s: no reference accumulators under %s", cluster, reference)
        return None


def _attach_lifespan(
    matrix: pd.DataFrame, lifespan_roots: list[Path] | None, cluster: str
) -> pd.DataFrame:
    """Opt-in whole-release runtime columns.

    See src/lifespan.py for what these are and why they sit outside the
    early-warning restriction.
    """
    if not lifespan_roots:
        return matrix
    lifespan = load_lifespan(list(lifespan_roots), cluster)
    if lifespan is None:
        LOGGER.info("%s: no lifespan table under %s", cluster, lifespan_roots)
        return matrix
    extra = lifespan_features(matrix, lifespan, cluster)
    LOGGER.info("%s: attached %s whole-release runtime features", cluster, extra.shape[1])
    return pd.concat([matrix, extra], axis=1)


def _reference_matrix(
    reference: Path | None,
    cluster: str,
    window_seconds: float,
    lifespan_roots: list[Path] | None,
) -> pd.DataFrame | None:
    """Feature matrix of the unlabelled scored chunk, used only to compare
    feature ranges. No label is read; the judge accumulators contain none."""
    if reference is None:
        return None
    try:
        matrix = build_matrix(load_accumulators(reference, cluster), window_seconds)
    except (FileNotFoundError, RuntimeError):
        return None
    return _attach_lifespan(matrix, lifespan_roots, cluster)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three FRESCO cluster models.")
    parser.add_argument("--accumulators", type=Path, nargs="+", required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--clusters", nargs="+", choices=CLUSTERS, default=list(CLUSTERS))
    parser.add_argument("--holdout-months", type=int, default=3)
    parser.add_argument("--window-minutes", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=0, help="0 uses the per-cluster default")
    parser.add_argument("--max-holdout-rows", type=int, default=120_000)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Judge accumulator directory. Only its identity columns are read, never "
        "labels, to measure how often the scored chunk contains a user, account or "
        "workload name the training months never saw.",
    )
    parser.add_argument(
        "--identity-dropout",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto picks per cluster on the judge-like holdout",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Independent refits whose ranks are averaged; blunts the seed-to-seed "
        "swing that makes a single fit unreliable on C",
    )
    parser.add_argument(
        "--lifespan",
        type=Path,
        nargs="+",
        help="Directory of whole-release observation spans. OPT-IN: these describe how "
        "long each job ran and are not early-window information; see src/lifespan.py",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.models.mkdir(parents=True, exist_ok=True)
    report_path = args.models / "validation_report.json"
    # Clusters can be trained one per invocation on a memory-limited machine, so
    # the report accumulates rather than being replaced.
    by_cluster: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        by_cluster = {entry["cluster"]: entry for entry in previous.get("clusters", [])}

    for cluster in args.clusters:
        cap = args.max_rows or DEFAULT_MAX_ROWS[cluster]
        reference_identities = _reference_identities(args.reference, cluster)
        reference_matrix = _reference_matrix(
            args.reference, cluster, args.window_minutes * 60.0, args.lifespan
        )
        artifacts, report = train_cluster(
            args.accumulators,
            cluster,
            args.holdout_months,
            args.window_minutes * 60.0,
            cap,
            DEFAULT_MIN_MONTH.get(cluster),
            args.seed,
            args.max_holdout_rows,
            reference_identities,
            args.identity_dropout,
            args.seeds,
            reference_matrix,
            args.lifespan,
        )
        joblib.dump(artifacts, args.models / f"{cluster}_model.joblib", compress=3)
        by_cluster[cluster] = report
        LOGGER.info("%s: saved %s", cluster, args.models / f"{cluster}_model.joblib")

    reports = [by_cluster[cluster] for cluster in CLUSTERS if cluster in by_cluster]
    macro = float(
        np.mean(
            [r["holdout_average_precision"][r.get("predictor", "blend")] for r in reports]
        )
    )
    payload = {
        "clusters": reports,
        "macro_holdout_auprc": macro,
        "complete": len(reports) == len(CLUSTERS),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "Macro holdout AUPRC %.5f over %s -> %s",
        macro,
        [r["cluster"] for r in reports],
        report_path,
    )


if __name__ == "__main__":
    main()
