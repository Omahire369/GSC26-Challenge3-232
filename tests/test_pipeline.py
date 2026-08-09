"""Unit tests for the invariants that matter, on synthetic data.

These run without the competition bundles, so a reviewer can check the pipeline's
contracts without first downloading 55 000 Parquet files. They cover the things
that would silently corrupt a submission: accumulator merging across hourly file
boundaries, the leakage rules in the encoder, the drift guards, and the
submission format itself.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.artifacts import blend, choose_predictor, rank
from src.config import CLUSTERS, POSITIVE_OUTCOMES, cluster_for_path
from src.drift import (
    blank_identities,
    category_holdout_mask,
    columns_for_field,
    measure_unseen_rates,
)
from src.encoders import ClusterEncoder
from src.features import identifier_number, observation_delay, reference_range_guard
from src.lifespan import TIMELIMIT_SECONDS, merge_lifespan


# --------------------------------------------------------------------------- #
# cluster routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name,expected",
    [
        ("2017-04-01-00_S.parquet", "S"),
        ("2017-04-01-00_C.parquet", "C"),
        ("2023-04-01-00.parquet", "Anvil"),
    ],
)
def test_cluster_is_read_from_the_file_name(name, expected):
    assert cluster_for_path(name) == expected


def test_every_cluster_has_a_timelimit_unit():
    # S and C report minutes, Anvil seconds. A missing entry would silently
    # scale runtime ratios by 60.
    assert set(TIMELIMIT_SECONDS) == set(CLUSTERS)


# --------------------------------------------------------------------------- #
# target definition
# --------------------------------------------------------------------------- #

def test_positive_class_excludes_cancelled():
    assert POSITIVE_OUTCOMES == {"FAILED", "TIMEOUT", "NODE_FAIL"}
    assert "CANCELLED" not in POSITIVE_OUTCOMES
    assert "COMPLETED" not in POSITIVE_OUTCOMES


# --------------------------------------------------------------------------- #
# Stage 1 merging
# --------------------------------------------------------------------------- #

def test_lifespan_merges_a_job_split_across_two_files():
    first = pd.DataFrame(
        {"jid": ["A"], "life_first": [100.0], "life_last": [200.0], "life_rows": [3],
         "life_hosts": [2], "start_epoch": [90.0], "timelimit": [60.0]}
    )
    second = pd.DataFrame(
        {"jid": ["A"], "life_first": [300.0], "life_last": [400.0], "life_rows": [2],
         "life_hosts": [1], "start_epoch": [90.0], "timelimit": [60.0]}
    )
    merged = merge_lifespan([first, second])
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["life_first"] == 100.0 and row["life_last"] == 400.0
    assert row["life_rows"] == 5
    assert row["life_hosts"] == 2


def test_observation_delay_takes_the_earliest_populated_bucket():
    data = pd.DataFrame({"tmin_p0": [np.nan, -30.0], "tmin_p1": [450.0, 400.0], "tmin_p2": [1000.0, np.nan]})
    assert list(observation_delay(data)) == [450.0, -30.0]


# --------------------------------------------------------------------------- #
# encoder: a job's own outcome must not reach its own encoded value
# --------------------------------------------------------------------------- #

def test_expanding_encoding_ignores_the_current_and_later_months():
    # One user, failing in the first month and succeeding afterwards. The first
    # month has no history, so it must fall back to the prior rather than to its
    # own outcome.
    matrix = pd.DataFrame(
        {
            "username": ["u"] * 6,
            "account": ["a"] * 6,
            "queue": ["q"] * 6,
            "unit": [""] * 6,
            "jobname": ["j"] * 6,
            "timelimit_bucket": ["t"] * 6,
            "month": ["2020-01", "2020-01", "2020-02", "2020-02", "2020-03", "2020-03"],
        }
    )
    target = pd.Series([1, 1, 0, 0, 0, 0], dtype=np.int8)
    encoder = ClusterEncoder()
    encoder.fit_categories(matrix)
    encoded = encoder.fit_expanding(matrix, target)

    prior = float(target.mean())
    first_month = encoded.loc[matrix["month"] == "2020-01", "te_username"]
    assert np.allclose(first_month, prior), "first month must not see its own labels"
    # By March the history holds the January failures, so the encoding must have
    # moved above the February value.
    february = encoded.loc[matrix["month"] == "2020-02", "te_username"].iloc[0]
    march = encoded.loc[matrix["month"] == "2020-03", "te_username"].iloc[0]
    assert february > prior
    assert march < february, "March sees February's successes and should fall"


def test_unseen_category_falls_back_to_the_prior():
    matrix = pd.DataFrame(
        {
            "username": ["seen"] * 4,
            "account": ["a"] * 4,
            "queue": ["q"] * 4,
            "unit": [""] * 4,
            "jobname": ["j"] * 4,
            "timelimit_bucket": ["t"] * 4,
            "month": ["2020-01", "2020-01", "2020-02", "2020-02"],
        }
    )
    target = pd.Series([1, 0, 1, 1], dtype=np.int8)
    encoder = ClusterEncoder()
    encoder.fit_categories(matrix)
    encoder.fit_expanding(matrix, target)

    fresh = matrix.copy()
    fresh["username"] = "never-seen"
    encoded = encoder.transform_target_statistics(fresh)
    assert np.allclose(encoded["te_username"], encoder.prior)
    assert np.allclose(encoded["tc_username"], 0.0)


# --------------------------------------------------------------------------- #
# drift handling
# --------------------------------------------------------------------------- #

def test_identity_columns_are_grouped_by_field():
    columns = ["te_username", "tc_username", "te_username__queue", "cat_username",
               "te_jobname", "cat_queue", "te_queue__timelimit_bucket", "obs_rows"]
    username = columns_for_field("username", columns)
    assert set(username) == {"te_username", "tc_username", "te_username__queue", "cat_username"}
    assert "cat_queue" not in username
    assert "obs_rows" not in username


def test_blanking_gives_masked_rows_exactly_what_an_unseen_category_gets():
    encoded = pd.DataFrame(
        {
            "te_username": np.float32([0.9, 0.9]),
            "tc_username": np.float32([5.0, 5.0]),
            "cat_username": np.int32([7, 7]),
            "obs_rows": np.float32([3.0, 3.0]),
        }
    )
    masks = {"username": np.array([True, False])}
    out = blank_identities(encoded, masks, prior=0.25)
    assert out.loc[0, "te_username"] == pytest.approx(0.25)
    assert out.loc[0, "tc_username"] == 0.0
    assert out.loc[0, "cat_username"] == 0
    assert out.loc[1, "te_username"] == pytest.approx(0.9), "unmasked row untouched"
    assert out.loc[0, "obs_rows"] == pytest.approx(3.0), "non-identity column untouched"


def test_holdout_mask_withdraws_whole_categories():
    values = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    mask = category_holdout_mask(values, rate=0.3, seed=0)
    # Every row of a chosen category is masked; no category is half in.
    for category in values.unique():
        rows = mask[(values == category).to_numpy()]
        assert rows.all() or not rows.any()


def test_unseen_rate_matches_the_share_of_reference_rows():
    training = pd.DataFrame({"username": ["a", "b"], "jobname": ["x", "y"], "account": ["p", "q"]})
    reference = pd.DataFrame(
        {"username": ["a", "a", "zzz", "zzz"], "jobname": ["x", "x", "x", "x"],
         "account": ["p", "p", "p", "p"]}
    )
    rates = measure_unseen_rates(training, reference)
    assert rates["username"] == pytest.approx(0.5)
    assert rates["jobname"] == pytest.approx(0.0)


def test_range_guard_drops_a_feature_that_runs_off_the_training_range():
    # This is the regression that cost 0.018 on the leaderboard: a feature
    # tracking calendar time, whose scored values sit beyond anything trained on.
    training = pd.DataFrame({"bounded": np.random.default_rng(0).random(500),
                             "timelike": np.arange(500.0)})
    scored = pd.DataFrame({"bounded": np.random.default_rng(1).random(200),
                           "timelike": np.arange(400.0, 600.0)})
    kept, dropped = reference_range_guard(training, scored, ["bounded", "timelike"])
    assert kept == ["bounded"]
    assert dropped and dropped[0][0] == "timelike"


def test_range_guard_keeps_everything_when_the_ranges_agree():
    frame = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    kept, dropped = reference_range_guard(frame, frame, ["a", "b"])
    assert kept == ["a", "b"] and dropped == []


# --------------------------------------------------------------------------- #
# ensembling and scoring
# --------------------------------------------------------------------------- #

def test_blend_preserves_ordering_and_stays_in_range():
    a = np.array([0.1, 0.4, 0.35, 0.9])
    b = np.array([0.2, 0.3, 0.5, 0.8])
    combined = blend([a, b])
    expected_order = np.argsort(np.mean([rank(a), rank(b)], axis=0))
    assert list(np.argsort(combined)) == list(expected_order)
    assert combined.min() >= 0.0 and combined.max() <= 1.0


def test_predictor_selection_prefers_the_blend_within_the_margin():
    assert choose_predictor({"lightgbm": 0.61, "xgboost": 0.58, "blend": 0.605}) == "blend"


def test_predictor_selection_takes_a_clear_single_winner():
    # C's real numbers: LightGBM beats the blend by 0.037, far past holdout noise.
    assert choose_predictor({"lightgbm": 0.6616, "xgboost": 0.5679, "blend": 0.6248}) == "lightgbm"


def test_identifier_number_extraction():
    values = pd.Series(["USER12717_S", "JOBNAME61256", "GROUP204", None, "nodigits"])
    numbers = identifier_number(values)
    assert numbers.iloc[0] == 12717
    assert numbers.iloc[1] == 61256
    assert numbers.iloc[2] == 204
    assert pd.isna(numbers.iloc[3]) and pd.isna(numbers.iloc[4])


# --------------------------------------------------------------------------- #
# submission contract
# --------------------------------------------------------------------------- #

def test_submission_matches_the_published_row_ids(tmp_path):
    from src.predict import create_submission

    sample = pd.DataFrame({"row_id": ["S_JOB1_S", "C_JOB2", "Anvil_JOB3"],
                           "failure_probability": [0.5, 0.5, 0.5]})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)

    def fake_predict(accumulators, models, cluster, lifespan_roots=None):
        rows = {"S": ("S_JOB1_S", 0.8), "C": ("C_JOB2", 0.2), "Anvil": ("Anvil_JOB3", 0.6)}
        row_id, value = rows[cluster]
        return pd.DataFrame({"row_id": [row_id], "failure_probability": [value]})

    import src.predict as predict_module

    original = predict_module.predict_cluster
    predict_module.predict_cluster = fake_predict
    try:
        output = tmp_path / "out"
        zip_path = create_submission(tmp_path, tmp_path, sample_path, output)
    finally:
        predict_module.predict_cluster = original

    written = pd.read_csv(output / "submission.csv")
    assert list(written.columns) == ["row_id", "failure_probability"]
    assert written["row_id"].tolist() == sample["row_id"].tolist()
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["submission.csv"]


def test_submission_refuses_to_guess_a_missing_row(tmp_path):
    from src.predict import create_submission

    sample = pd.DataFrame({"row_id": ["S_JOB1_S", "C_JOB2"], "failure_probability": [0.5, 0.5]})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)

    def missing_c(accumulators, models, cluster, lifespan_roots=None):
        if cluster != "S":
            return pd.DataFrame({"row_id": [], "failure_probability": []})
        return pd.DataFrame({"row_id": ["S_JOB1_S"], "failure_probability": [0.8]})

    import src.predict as predict_module

    original = predict_module.predict_cluster
    predict_module.predict_cluster = missing_c
    try:
        with pytest.raises(ValueError, match="missing"):
            create_submission(tmp_path, tmp_path, sample_path, tmp_path / "out")
    finally:
        predict_module.predict_cluster = original
