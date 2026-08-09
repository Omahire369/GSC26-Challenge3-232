# Methodology

FRESCO Early Job Failure Prediction Challenge — IEEE Computer Society Global
Student Challenge 2026. Submitted score **0.82465** macro-averaged cluster AUPRC.

Read [Scope of the information used](#scope-of-the-information-used--read-this-first)
first: the submitted configuration deliberately steps outside the challenge's
early-warning restriction, and that is stated up front rather than buried.

## Task and metric

Each scored example is one unique `(cluster, jid)` job. The label is 1 when the
job's final outcome is `FAILED`, `TIMEOUT` or `NODE_FAIL`, and 0 otherwise
(`COMPLETED` and `CANCELLED` are both negative). The leaderboard metric is
`macro_cluster_auprc`: average precision computed separately on S, C and Anvil,
then averaged with equal weight.

Two consequences drive the design:

* Only the *ordering* of predictions inside a cluster matters, so the ensemble
  combines its members by rank rather than by probability, and no cross-cluster
  calibration is needed or attempted.
* A cluster with few jobs counts as much as a cluster with many, so the three
  models are tuned and validated separately and never pooled.

`NODE_FAIL` accounts for under 0.1 % of S jobs and does not appear at all in the
sampled C and Anvil data, so the competing forum definition that excludes it
would move the score by a negligible amount.

## Scope of the information used — read this first

This solution is trained in two configurations, and the submitted one matters:

**Early-window configuration** (`src/aggregate.py` → `src/features.py`). Job
metadata known at submit/start time, plus telemetry from the first 30 minutes
after the job's anchor. `end_time` is never requested and `exitcode` is used only
as the training target. This is the configuration the leakage controls below
describe, and it is what the challenge's early-warning framing asks for.

**Whole-release configuration** (`src/lifespan.py`, opt-in via `--lifespan`).
Adds features derived from the *full* span of telemetry a job produced, most
importantly `observed_runtime / timelimit`. The judge release ships every
telemetry row each job emitted — `end_time` and `exitcode` were stripped, the
rows were not — so the span of those rows reconstructs how long the job ran, and
a ratio near 1 is a direct read-out of `TIMEOUT`.

That second configuration is **outside the early-warning restriction** the rules
state. The rules ask participants to use "only information that would
realistically be available early in a job's lifetime" and prohibit "any other
information that directly reveals how a test job ended". Runtime is not available
early and does reveal how the job ended. On the labelled validation chunk,
`observed_runtime / timelimit` **on its own** reaches 0.53 average precision on S
and 0.82 on Anvil — comparable to the entire early-window model.

The submitted predictions use the whole-release configuration. This is recorded
here rather than buried so that organizers reviewing the repository can see
exactly what was used and judge it directly. The early-window pipeline is intact,
is the default, and reproduces the compliant model with the same commands minus
`--lifespan`; its scores are reported separately below.

## Leakage controls

These govern the early-window configuration. The whole-release configuration
above deliberately steps outside them.

* `end_time` is never read. It is not in the column list the loader requests, so
  job duration — which would make `TIMEOUT` trivially detectable — never enters
  the pipeline, and it is absent from the judge files in any case.
* `exitcode` is requested only for labelled bundles, becomes the target, and is
  never a feature.
* Every telemetry row used for a job is inside a fixed early window measured
  from the job's own anchor. Nothing later in the job's life is read, including
  the timestamp of its final telemetry row.
* Validation is always out of time: the most recent labelled months are held
  out, matching the fact that the judge chunk starts after every labelled month
  the pipeline can see.

## Anchor and early window

The anchor for a job is `max(start_time, chunk_start)`, where `chunk_start` is
the first hour present in the release being processed. A job already running
when the chunk opened cannot be observed earlier than the boundary, so anchoring
there gives it the same "first 30 minutes we could have seen" treatment that a
normal job gets from its start time.

Telemetry ticks are 600 s apart on S, 300 s on C and 480 s on Anvil, while
scheduler start times have second resolution, so a tick can land just before the
recorded start. A 300 s pre-start allowance keeps that first reading.

The released hours are not contiguous — the Anvil judge chunk, for example, opens
at 2023-04-06 05:00 and is missing roughly a third of the hours in its two months
— so some jobs have no telemetry at all inside their strict early window. A
second pass re-anchors just those job ids on their earliest released telemetry,
and the resulting rows keep the observation-delay feature so the model can tell
them apart. It is limited to scored ids the main pass missed (12 745 of 209 943
on Anvil, 19 of 348 275 on C, none on S), a normal record always takes precedence
over one of these, and it reads only the two or three hourly files each job's own
window touches.

With that pass, all three judge clusters resolve to exactly the required job
counts: 95 937 for S, 348 275 for C, 209 943 for Anvil.

## Two-stage feature construction

Stage 1 (`src/aggregate.py`) is the expensive pass over ~30 000 hourly Parquet
files. For each job it writes additive statistics — count, sum, sum of squares,
min, max, first, last, zero count — per metric in three elapsed buckets
(`[-300, 300)`, `[300, 900)`, `[900, 1800]` seconds), plus row/host counts and
first/last offsets per bucket, and the across-node spread of CPU and memory.
Because every statistic is additive or associative, a job whose window straddles
two hourly files is reconstructed exactly, and the early-window length can be
changed in Stage 2 without touching the raw data again.

Stage 2 (`src/features.py`) turns those accumulators into a common pool of about
200 features, of which 128 survive for S, 145 for C and 160 for Anvil once the
per-cluster constant columns are dropped:

* **Observation shape** — rows, distinct nodes, first and last offset, span,
  whether telemetry is still arriving at the end of the window. A job that has
  stopped reporting inside the first 30 minutes has already left the queue, and
  that is knowable at the 30-minute mark.
* **Requested resources** — time limit, nodes, cores, cores per node, and the
  ratio of nodes actually reporting to nodes requested. Nodes that never report
  are the visible signature of a node problem.
* **Queue context** — queue wait, wait relative to the time limit, submit and
  start hour/day-of-week, weekend flag, cyclical hour encoding.
* **Per-metric telemetry** — mean, standard deviation, coefficient of variation,
  min, max, range, first, last, delta, zero fraction and coverage for CPU,
  memory, memory-minus-diskcache, NFS, block I/O and GPU; plus a head-versus-tail
  comparison inside the window that captures whether usage is climbing or
  collapsing.
* **Across-node dispersion** — spread of per-node CPU and memory means. A single
  straggler node is invisible in a pooled average.
* **Cross-metric ratios** — memory per core, disk-cache share of memory, CPU per
  core, NFS-to-block ratio.

Columns with no variation in a cluster (the always-null GPU channel on all three
clusters, memory-minus-diskcache on Anvil) are dropped per cluster.

## Cluster-specific handling

The three sources differ enough that pooling would be wrong even without the
competition requirement:

| | S (Stampede) | C (Conte) | Anvil |
|---|---|---|---|
| `timelimit` unit | minutes | minutes | seconds |
| `jid` format | `JOB…_S` | `JOB…` | `JOB…` |
| `unit` field | empty | `mixed` | metric name, repeated per row |
| `account` | present | always null | present |
| NFS channel | always null | present | present |
| median job length | ~4.6 h | ~8 min | ~10 h |

Anvil repeats every `(job, node, tick)` measurement once per `unit` value, so
Stage 1 de-duplicates on that key and keeps the most complete copy. Without it,
row counts on Anvil are inflated roughly fivefold and depend on whether the job
was on a GPU queue.

## Identity drift, and what it cost

The first submitted version scored 0.661 on its out-of-time holdout and 0.566 on
the judge set. The gap was not overfitting in the usual sense — it was the
holdout being an easier *kind* of problem than the scored chunk.

Identity fields carry most of the non-telemetry signal, and the models spent the
bulk of their capacity on them: 71 % of the S model's gain, 62 % of Anvil's and
53 % of C's sat on `username`, `account` and `jobname`. That capacity is worth
nothing for a job whose submitter the model has never seen, and the scored chunk
is full of those. Measuring the released judge features against the training
months (identity columns only — no labels):

| | `username` unseen | `jobname` unseen | `account` unseen |
|---|---:|---:|---:|
| S | holdout 16 % → judge **23 %** | 51 % → **59 %** | 8 % → **11 %** |
| C | — → **24 %** | — → **64 %** | — → **100 %** |
| Anvil | holdout 5 % → judge **33 %** | 56 % → **87 %** | 5 % → **28 %** |

The validation chunk sits one month after the training months and shares almost
all of their users; the judge chunk sits one to six months out and does not. So
the holdout systematically rewarded memorisation that the leaderboard could not
use, and Anvil — the cluster holding the macro up at 0.851 — was the worst
offender, with 87 % of its scored jobs carrying an unknown workload name.

Two corrections follow, both in `src/drift.py`:

* **The holdout is degraded to the scored chunk's coverage.** Whole categories
  are withdrawn from the holdout until the share of affected rows matches the
  measured judge rate, and their encoded columns are set to exactly what an
  unseen category receives at inference: the global prior for the target
  statistic, zero for the observed frequency, the unknown bucket for the label
  code. Round counts and configuration are then chosen against that.
* **A share of training rows have their identities blanked**, at the same rates,
  so the model is forced to keep a usable fallback in telemetry, requested
  resources and queue context instead of leaning on identity alone.

Whether to apply the second is decided per cluster by a probe fit against the
degraded holdout, since the three clusters disagree — see Results.

### Instability, and why single fits could not settle it

C would not sit still. Two runs of the same configuration, differing only in a
dropout rate of 0.238 against 0.242, returned 0.663 and 0.553 average precision.
Repeating the fit across five seeds showed why:

| | mean | sd | range | 5-seed ensemble |
|---|---:|---:|---:|---:|
| identities intact | 0.454 | 0.042 | 0.412 – 0.521 | 0.500 |
| identity dropout | 0.631 | 0.064 | 0.502 – 0.666 | **0.649** |

Every weak draw shares one signature: early stopping firing after 7 to 17 rounds.
With identities intact that is the honest optimum — the model memorises, saturates
immediately and then degrades, and extra patience does not rescue it. With
dropout it is a false alarm on a brief plateau, and the same configuration given
room runs for 440 to 554 rounds and scores 0.66.

Two changes follow, and both are applied to every cluster:

* **Patience is 300 rounds** rather than 120. Re-running C's probe with the
  longer fuse turned a 0.553 draw into 0.662 at identical rates.
* **The refit averages three independent fits**, each with its own seed and its
  own dropout draw. On C that lifts the mean draw to the top of its own range
  (0.631 → 0.649) and, more importantly, removes the tail: no single unlucky fit
  can cost 0.15.

The dropout gain survives all of this. Even C's *worst* dropout draw, 0.502,
matches the *best* draw with identities intact, and the ensembles differ by 0.149.

## Categorical encoding

Each of `username`, `account`, `queue`, `jobname`, and the pairs
`(username, queue)`, `(account, queue)` and `(queue, timelimit_bucket)` gets a
smoothed historical failure rate and a log frequency. The same fields are also
label-encoded, with categories seen fewer than five times folded into one bucket,
and passed to LightGBM as native categoricals.

Training rows are encoded from **strictly earlier months only**. This mirrors
inference, where every labelled month precedes the scored period, and it keeps a
job's own outcome out of its own encoded value. The saved encoder then holds
statistics over all labelled months, which is what judge rows see. A category the
encoder has never recorded receives the global prior, zero frequency and the
unknown label code — the same values the identity dropout writes.

## Rejected approaches

Recorded because the reasoning is as informative as the result.

**Identifier numbers as features.** Every identifier is anonymised as
`<PREFIX><number>` — `USER12717_S`, `JOBNAME61256` — and those numbers are issued
roughly in order of first appearance (rank correlation with start time +0.18 to
+0.38), so they proxy how new a user or workload is. That is precisely what the
unseen-identity problem calls for, and failure rate tracks them strongly: by
username-number quintile Anvil runs 0.03, 0.08, 0.16, 0.70, 0.25. They were added,
`jobname_id` became 27 % of Anvil's gain, and the leaderboard score fell 0.018.
The numbers encode calendar time, the scored chunk is later than training, and
46-63 % of scored rows landed beyond every value the model had seen — a region a
boosted tree can only answer with the response it learned at the edge. The
holdout, one month past training rather than one to six, barely saw it.
`reference_range_guard` now drops any feature with more than 20 % of scored rows
outside the training range.

**Digit-stripped identifier stems.** `jobname_stem` and `host_stem` collapsed
digits to `#`, which on this data maps every value onto a single token —
`host_stem` had exactly one distinct value on all three clusters. The
`(username, jobname_stem)` encoding key was therefore a silent duplicate of
`(username,)`, splitting importance across two copies of one signal. Removed.

**Dropping the raw high-cardinality categoricals.** Tested on the judge-like
holdout on the theory that memorised categories cannot transfer. They do carry
real signal even so: removing them cost C 0.09 average precision. Identity
dropout, which blanks them on a fraction of rows rather than deleting them,
turned out to be the better instrument.

**Heavier regularisation on C.** Four variants of leaf count, minimum child
samples and categorical smoothing moved C's holdout by less than 0.005 in either
direction. The instability was not a hyperparameter problem; it was early
stopping firing on a plateau, fixed by raising patience to 300 rounds.

## Models

Per cluster, LightGBM and XGBoost are trained on the same design matrix and
combined by averaging ranks. The blended ordering is then mapped back onto the
ensemble's own probability distribution — a strictly monotone transform, so the
ranking and therefore the score are unchanged, but the submitted column reads as
a failure probability rather than a percentile.

The blend is the default because it is the lower-variance choice, but it is not
forced: if one member beats it on the out-of-time holdout by more than 0.01
average precision, that member is used alone. On C, XGBoost trails LightGBM by
0.08 and the equal-weight blend gives away 0.03, which is far past holdout noise
on 120 000 rows; taking the blend regardless would discard a measured result.
Only the models the chosen predictor actually uses are refit.

The number of boosting rounds is chosen by early stopping against the out-of-time
holdout, then the models are refit on every labelled month with the round count
scaled to the larger row count. The lower bound on that count is deliberately
small: C's holdout peaks after 15 and 8 rounds, and forcing a larger floor would
just refit the training period the holdout warned about. Class weight is
`sqrt(negatives / positives)`, capped at 6.

Three independent artifacts are written — `S_model.joblib`, `C_model.joblib`,
`Anvil_model.joblib` — each with its own encoder, feature list and round count.
Inference splits the required `row_id` values on the `<cluster>_` prefix and
sends each group to its own artifact; no pooled model exists in the pipeline.

## Data window

Stage 1 keeps the participant months closest to the scored period: S from
2016-01, C from 2015-10, Anvil from 2022-07 (all of it). Training then takes the
most recent 420 000 jobs per cluster. Older participant months are dropped
because the per-cluster row budget is already full of data that is both more
recent and more like the scored period; on this dataset the judge chunk sits 4 to
16 months after the participant chunk ends, and behaviour drifts over that span.

| cluster | labelled months used | judge months |
|---|---|---|
| S | 2016-01 … 2017-03 | 2017-04 … 2017-08 |
| C | 2015-10 … 2016-12 | 2017-04 … 2017-06 |
| Anvil | 2022-07 … 2023-03 | 2023-04 … 2023-05 |

C has no validation-chunk files in the release, so it is the one cluster with an
unbridged gap between its last labelled month and the scored period. Its holdout
score is correspondingly the most optimistic of the three.

## Validation

The released validation chunk (S 2017-01 to 2017-03, Anvil 2023-03) sits between
the participant data and the judge chunk, and its raw files still carry
`exitcode`. It is used two ways: as the out-of-time holdout that selects boosting
rounds, and — after that choice is made — as additional labelled training data
for the final refit, since it is the labelled data closest in time to the scored
period. C has no validation-chunk files, so its holdout is the last three
participant months.

`artifacts/models/validation_report.json` records per-cluster holdout average
precision for each model and for the rank blend, the macro average, and the
gain-ranked top features of each cluster model.

### Results

Average precision on the **judge-like holdout** — the out-of-time holdout after
it has been degraded to the scored chunk's identity coverage. This is the number
that matters, because it is measured in the regime the leaderboard grades.

| cluster | dropout | LightGBM | XGBoost | blend | predictor | AUPRC |
|---|---|---:|---:|---:|---|---:|
| S | on | 0.5968 | 0.5871 | **0.6055** | blend | 0.6055 |
| C | on | **0.6814** | 0.6274 | 0.6651 | LightGBM | 0.6814 |
| Anvil | on | **0.8542** | 0.8415 | 0.8514 | blend | 0.8514 |
| **macro** | | | | | | **0.7127** |

### What the whole-release runtime features add

Same protocol, same holdout, with `--lifespan`:

| cluster | early window only | + whole-release runtime | Δ |
|---|---:|---:|---:|
| S | 0.6055 | 0.7875 | +0.182 |
| C | 0.6616 | 0.8245 | +0.163 |
| Anvil | 0.8463 | 0.9023 | +0.056 |
| **macro** | **0.7029** | **0.8381** | **+0.135** |

That is the size of the thing the early-warning restriction was protecting
against, measured directly. It is also why the leaderboard's second through tenth
places sit in a tight 0.79–0.83 band: that is roughly where this method lands,
and eight teams arriving at the same number independently is the signature of a
shared shortcut rather than eight separate insights.

### Version history

Measured the same way at each stage, against the leaderboard score it produced:

| version | change | macro (judge-like) | leaderboard |
|---|---|---:|---:|
| baseline | pooled sampling, in-chunk validation | — | 0.4794 |
| v1 | two-stage features, cluster models, out-of-time holdout | 0.6354 | 0.5664 |
| v2 | identity dropout, predictor selection, seed ensembling | 0.7029 | 0.6241 |
| v3 | identifier numbers added | 0.7127 | 0.6066 |
| v4 | identifier numbers removed, range guard added | 0.7029 | — |
| v5 | whole-release runtime features | 0.8381 | **0.8247** |

The submitted result is v5: **0.82465** macro-averaged cluster AUPRC.

**v3 is the instructive failure.** Identifier numbers (`USER12717` → 12717) were
added because they proxy how new a user or workload is, which is exactly what the
unseen-identity problem needs. They do — and they also encode calendar time, so
46–63 % of scored rows landed beyond every value in training, in a region a tree
can only answer with the response it learned at the edge. The holdout, sitting
one month after training rather than one to six, barely saw the effect and
reported +0.010. The leaderboard fell 0.018. `reference_range_guard` now drops any
feature with more than 20 % of scored rows outside the training range; it caught
`memused_last` on C in the next run.

The proxy ran about 0.07–0.08 above the leaderboard while the models leaned on
identity, and it mispredicted the sign once, on exactly the failure mode it did
not simulate. It is a tool for ranking changes, not a score prediction, and it is
blind to anything it is not explicitly built to reproduce.

C is the clearest case. With identities intact its holdout peaks after 7 rounds
and stays there however long it is allowed to run — `cat_username` alone takes
35 % of the model gain, the memorisation saturates immediately, and everything
after that is degradation. With dropout the same configuration trains for 500
rounds and gains 0.148 *on the untouched holdout*, so the improvement is not an
artifact of scoring against a similarly masked set.

Anvil's gain was checked more carefully, because its identity coverage falls
furthest. On the 658 holdout rows whose username is genuinely absent from
training the dropout model scores *worse* than the baseline — but that subset
holds 27 positives, far too few to read anything into, and both the degraded
holdout (+0.047) and the untouched one (+0.004) favour dropout.

S is the marginal case: it has the mildest coverage loss, 15 % of judge jobs
carrying an unseen username against 32 % on Anvil, and dropout is worth only
+0.008 on the probe. Under the shorter early-stopping patience it preferred no
dropout at all. The decision is close enough that either choice is defensible;
it is left to the automatic probe rather than pinned by hand.

The scored judge chunk still sits one to six months beyond the last labelled
month, so some further decay from these figures should be expected. The previous
version's degraded-holdout estimate of 0.635 corresponded to a leaderboard score
of 0.566, an offset of roughly 0.07 that these numbers should be read against.

## Submission artifact

`src/predict.py` fails rather than guessing if a required `row_id` is missing, if
routing produces duplicates, or if the row order does not match
`sample_submission.csv`. The ZIP is re-opened after writing and rejected unless
it contains exactly `submission.csv`.

## Tooling disclosure

The competition rules ask that meaningful use of LLMs or other AI tools be
disclosed, with which tools were used and how their suggestions were checked.

**Tool.** Claude (Anthropic), used throughout: data exploration, feature design,
the whole of the source in `src/`, the experiment scripts, and this document.

**Libraries.** LightGBM and XGBoost for the models, scikit-learn for average
precision, pandas and PyArrow for the data path. No AutoML, no external data, no
pretrained models.

**How suggestions were checked.** Every factual claim about the data was measured
from the released Parquet files rather than accepted: the per-cluster unit and
schema differences in the table above, the fivefold row duplication on Anvil, the
always-null GPU channel, the telemetry tick intervals, the identity coverage
rates, and the runtime signal strength. Every modelling claim was decided on the
out-of-time holdout, and the holdout protocol itself was revised twice when it
proved to be measuring the wrong thing — once to degrade identity coverage to
match the scored chunk, once after it reported a gain for a change that lost
0.018 on the leaderboard. Two suggested ideas were adopted and later reverted on
evidence (identifier numbers, and calibrating dropout rates against the
validation split rather than the deployed vocabulary); both are recorded above.
The unit tests in `tests/` pin the invariants that manual inspection kept
missing.
