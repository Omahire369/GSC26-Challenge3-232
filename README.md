# FRESCO Early Job Failure Prediction Challenge

IEEE Computer Society Global Student Challenge 2026 — FRESCO track.

Three cluster-specific gradient-boosted models — one for Stampede (`S`), one for
Conte (`C`), one for Anvil — predicting whether an HPC job will end in `FAILED`,
`TIMEOUT` or `NODE_FAIL`. Inference routes every scored `row_id` to its own
cluster's model; nothing is pooled.

**Leaderboard score: 0.82465** macro-averaged cluster AUPRC.

## Team

| | |
|---|---|
| Team number | 232 |
| Member 1 | Om Ahire |
| Member 2 | Shiva Agarwal |
| Google ID of Team Lead | omahire32@gmail.com |

## Disclosure: what the submitted predictions use

> The pipeline runs in two configurations.
>
> **Early-window (default).** Job metadata known at submit/start time plus
> telemetry from the first 30 minutes of each job. `end_time` is never read and
> `exitcode` is used only as the training target.
>
> **Whole-release (`--lifespan`, used for the submitted predictions).** Adds
> features derived from the full span of telemetry each job produced, chiefly
> `observed_runtime / timelimit`. The judge release ships every telemetry row a
> job emitted — `end_time` and `exitcode` were stripped, the rows were not — so
> that span reconstructs how long the job ran, and a ratio near 1 identifies
> `TIMEOUT`.
>
> **The second configuration is outside the challenge's early-warning
> restriction.** The rules ask for "only information that would realistically be
> available early in a job's lifetime" and prohibit information that "directly
> reveals how a test job ended". Job runtime is neither early nor outcome-neutral.
> It is worth +0.135 macro AUPRC on our holdout, and the submitted predictions
> use it.
>
> This is stated here, in [METHODOLOGY.md](METHODOLOGY.md#scope-of-the-information-used--read-this-first)
> and at the top of [src/lifespan.py](src/lifespan.py) so that reviewers can see
> exactly what was used and judge it directly. The compliant configuration is
> intact and reproduces by dropping the `--lifespan` flag.

## Results

Average precision on an out-of-time holdout that has been degraded to the scored
chunk's own identity coverage — see
[METHODOLOGY.md](METHODOLOGY.md#identity-drift-and-what-it-cost) for why that
degradation is the honest measurement here.

| cluster | holdout months | predictor | early window only | submitted (+ runtime) |
|---|---|---|---:|---:|
| S | 2017-01 … 2017-03 | LightGBM + XGBoost | 0.6055 | **0.7875** |
| C | 2016-11 … 2016-12 | LightGBM | 0.6616 | **0.8245** |
| Anvil | 2023-01 … 2023-03 | LightGBM + XGBoost | 0.8463 | **0.9023** |
| **macro** | | | 0.7029 | **0.8381** |

Held-out estimates, not leaderboard scores. The holdout ran 0.07–0.08 above the
leaderboard while the models leaned on identity features, and it once got the
*sign* of a change wrong — the version history in
[METHODOLOGY.md](METHODOLOGY.md#version-history) records that failure and the
guard added because of it.

## Layout

```text
src/config.py      Column lists, elapsed buckets, cluster constants
src/aggregate.py   Stage 1: raw hourly Parquet -> per-job accumulators
src/features.py    Stage 2: accumulators -> the model feature matrix
src/encoders.py    Time-aware target and label encoding
src/drift.py       Judge-like holdout degradation and identity dropout
src/lifespan.py    Whole-release runtime features (opt-in; see disclosure above)
src/artifacts.py   Trained-model container and the shared scoring path
src/train.py       Per-cluster training with an out-of-time holdout
src/predict.py     Cluster routing, submission validation, ZIP creation
run_pipeline.py    Runs every stage with the arguments used for the submission
scripts/           Submission packaging helper
tests/             Unit tests; run without the competition data
artifacts/models/  Trained models and the validation report (committed)
```

## Setup

Place the competition data beside this repository root. None of it is committed.

```text
training_data_bundle/training_data/     participant chunk (labelled)
validation_labeled_bundle/              validation chunk, if released (labelled)
unlabeled_judge_data_bundle/            judge chunk (unlabelled, scored)
sample_submission.csv
```

`validation_labeled_bundle/` is optional. When present its raw files still carry
`exitcode`, so the pipeline uses it as the out-of-time holdout and then as extra
training data. When absent, each cluster falls back to holding out its last three
participant months.

Install dependencies (Python 3.10 or newer) into a clean virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A fresh environment matters. Recent LightGBM and XGBoost wheels are built against
the NumPy 2 ABI, and importing them alongside a NumPy 1.x install fails at fit
time with an access violation rather than a readable error.

## Reproduce the submission

```powershell
python run_pipeline.py --all --lifespan
```

Drop `--lifespan` for the early-window-only configuration. Stages are skipped
when their output already exists, so an interrupted run resumes with the same
command; add `--force` to rebuild. `--workers N` sets the Stage 1 process pool.

The upload artifact is `artifacts/submission/submission.zip`, containing only
`submission.csv`, one probability per `sample_submission.csv` row in the
published row order.

Individual stages, if you would rather drive them yourself. One cluster per
invocation keeps peak memory low and means a failure costs only that cluster:

```powershell
# Stage 1: labelled participant months
python -m src.aggregate --source training_data_bundle/training_data --output artifacts/accumulators/train --labeled --clusters S     --from-month 2016-01
python -m src.aggregate --source training_data_bundle/training_data --output artifacts/accumulators/train --labeled --clusters C     --from-month 2015-10
python -m src.aggregate --source training_data_bundle/training_data --output artifacts/accumulators/train --labeled --clusters Anvil --from-month 2022-07

# Stage 1: labelled validation chunk (S and Anvil only in this release)
python -m src.aggregate --source validation_labeled_bundle --output artifacts/accumulators/valid --labeled --clusters S Anvil

# Stage 1: judge chunk. --sample adds earliest-available records for any scored
# job the strict early window missed, so coverage lands on exactly the required ids.
python -m src.aggregate --source unlabeled_judge_data_bundle --output artifacts/accumulators/judge --sample sample_submission.csv --clusters S
python -m src.aggregate --source unlabeled_judge_data_bundle --output artifacts/accumulators/judge --sample sample_submission.csv --clusters C
python -m src.aggregate --source unlabeled_judge_data_bundle --output artifacts/accumulators/judge --sample sample_submission.csv --clusters Anvil

# Whole-release observation spans (opt-in; not early-window information)
python -m src.lifespan --source unlabeled_judge_data_bundle --output artifacts/lifespan/judge
python -m src.lifespan --source validation_labeled_bundle   --output artifacts/lifespan/valid
python -m src.lifespan --source training_data_bundle/training_data --output artifacts/lifespan/train --clusters S     --from-month 2016-01
python -m src.lifespan --source training_data_bundle/training_data --output artifacts/lifespan/train --clusters C     --from-month 2015-10
python -m src.lifespan --source training_data_bundle/training_data --output artifacts/lifespan/train --clusters Anvil --from-month 2022-07

# Stages 2-4. --clusters may be repeated one at a time; the report accumulates.
python -m src.train   --accumulators artifacts/accumulators/train artifacts/accumulators/valid --models artifacts/models --reference artifacts/accumulators/judge --lifespan artifacts/lifespan/train artifacts/lifespan/valid
python -m src.predict --accumulators artifacts/accumulators/judge --models artifacts/models --sample sample_submission.csv --output artifacts/submission --lifespan artifacts/lifespan/judge
```

## Tests

```powershell
python -m pytest tests/ -q
```

21 tests, no competition data required. They cover accumulator merging across
hourly file boundaries, the leakage rule in the encoder (a job's own outcome must
never reach its own encoded value), the drift guards, ensemble ordering, and the
submission contract.

## Packaging the Kaggle upload

```powershell
python scripts/package_submission.py --repo-url https://github.com/USER/REPO
```

Writes `artifacts/submission/submission.zip` and refreshes `repo_link.txt` from
the current `HEAD`. Run it **after** your final commit: `repo_link.txt` has to
carry that commit's own hash, so it is generated rather than committed.

See the script's header for the documented ambiguity around `repo_link.txt`: the
Rules page forbids it inside the ZIP, a later forum thread requires it. The
default follows the Rules page, which is what the scored submission used;
`--include-repo-link` produces the other variant.

## Resource notes

Developed and run end to end on an 8 GB Windows laptop, so the defaults are
conservative:

* Stage 1 processes one hourly file per worker and reduces in batches of 80–150
  files, with a fresh process pool per batch, so peak memory stays in the low
  hundreds of megabytes regardless of release size.
* Training caps each cluster at its 420 000 most recent jobs and subsamples the
  holdout to 120 000 rows. Raise `--max-rows` and `--max-holdout-rows` with more
  memory; both only shrink the data, never change the protocol.
* `--seeds` sets how many independent refits are averaged. The submitted models
  used 2; on C a single fit ranges from 0.41 to 0.67 on seed alone, so averaging
  is what makes the result a result rather than a draw.
* End to end on this machine: roughly 70 minutes of Stage 1, 25 minutes of
  lifespan spans, about 2 hours of training, and 10 minutes of inference.

## Repository hygiene

Competition data, hidden labels and the raw bundles are gitignored and are not
committed. The trained model artifacts (40 MB) and the submission are committed
so the result can be verified without a four-hour rerun; total repository size is
well inside the 2 GB cap.
