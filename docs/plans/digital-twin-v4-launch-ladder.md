# Digital Twin v4 Launch Ladder

Status: execution appendix

Owner: [`digital-twin-v4-execution-plan.md`](digital-twin-v4-execution-plan.md)

This file expands P3-P5 into attributable Launch Reviews. It is not a second
backlog authority. A row becomes executable only when its owner row in the main
plan is ready. `Observed lift` is always populated from a factual A/B artifact;
the simulator must never encode a desired uplift into the hidden world.

## 1. One Launch Review contract

Every LR changes one owned mechanism and freezes the other layers. Its artifact
records control/treatment versions, eligible unit, traffic, duration, sample and
feature manifests, served checkpoints, candidate widths, Top-K overlap, latency,
cost, SRM, MDE, confidence interval, primary/guardrail metrics, observed lift and
`LAUNCH | HOLD | REJECT`. Offline or shadow results never populate A/B lift.
The factual control is always the currently active accepted predecessor. If a
named intermediate row rejects, later challengers do not inherit it merely to
make the table linear; they compare against the still-active control.

The common path is:

```text
frozen candidate/time split
→ offline stage metrics
→ exact-feature shadow/replay
→ A/A and SRM
→ ramped factual A/B
→ mature short/long outcomes
→ launch decision
→ post-launch sample and distribution review
```

## 2. Core Feed algorithm and policy ladder

The order reflects a real system evolution: establish retrieval value, rank the
small pool directly, add coarse ranking only after fine cost becomes binding,
then calibrate primitive heads and introduce Value Tree and final-slate policy.

| ID | Control → treatment | Frozen boundary | Launch evidence |
|---|---|---|---|
| F-AA-00 | A → identical A | everything | SRM passes; confidence interval covers zero; replay is exact |
| F-R00 | no recommender → eligible random retrieval and random order | eligibility, corpus, slate width | establishes zero-personalization baseline; no lift claim |
| F-R01 | eligible random → eligible Popular | order only | stay/session and negative feedback; concentration guardrail |
| F-R02 | Popular → Popular + 30-day recent + locale/language eligibility | popularity formula | freshness/locale slices improve without coverage collapse |
| F-F00 | retrieval order → direct rule fine rank | small fixed candidate set | served Top-K changes; stay improves within latency budget |
| F-R03 | current routes → Following/author-affinity route | ranker and route quota | marginal unique relevant recall and creator concentration |
| F-R04 | current routes → Graph/I2I co-watch route | ranker and route quota | marginal unique recall and fixed-ranker delta, not route recall alone |
| F-R05 | current routes → cold-start exploration route | all mature-content routes | new-post coverage/quality and Feed guardrails; exact exploration probability |
| F-R06 | current routes → regional/topic Hot route | other routes and quota | trend response and fatigue/quality guardrails |
| F-R07 | current routes → Evergreen route | recent/hot routes | durable-content value without stale-content regression |
| F-F01 | rule fine → logistic fine | candidates/features | request ranking, calibration, stay and slice non-regression |
| F-F02 | logistic fine → XGBoost fine | candidates/features/tuning budget | request NDCG/NE and factual A/B justify serving cost |
| F-R08 | deterministic ANN baseline → Two-Tower route | corpus/Top-K/ranker | unique positive recall and downstream fixed-ranker lift |
| F-R09 | Two-Tower → Multi-interest | corpus/Top-K/ranker | interest coverage and tail value exceed latency/index cost |
| F-F03 | best accepted tabular fine → W&D | candidates/features/budget | sparse memorization adds supported Top-K and A/B value |
| F-F04 | W&D → DeepFM | same inputs and budget | automatic second-order crosses beat explicit wide crosses |
| F-F05 | DeepFM → DCNv2 | same inputs and budget | bounded explicit crosses justify latency and memory |
| F-F06 | best non-sequence → DIN | short sequence fixed | candidate-aware sequence ablation and A/B pass |
| F-F07 | DIN → Transformer | sequence/event contract fixed | ordering/longer context gain exceeds P99 and GPU cost |
| F-F08 | best sequence single-task → shared-bottom multi-task | labels/value frozen | primitive heads add Pareto value without calibration regression |
| F-F09 | shared-bottom multi-task → MMoE | tasks/labels/value frozen | Pareto gain with healthy gates/experts and no sparse-task regression |
| F-F10 | MMoE → PLE | tasks/labels/value frozen | reduced negative transfer exceeds extra complexity |
| F-C00 | direct fine → rule coarse + same fine | recall/fine/mixer | fine Top-K ≥97%, rare-value ≥99%, lower fine QPS/P99; business non-inferior |
| F-C01 | rule coarse → logistic coarse | candidate budget/fine | pass-through improves or cost falls with guardrails flat |
| F-C02 | logistic coarse → XGBoost coarse | same candidates/features | request-grouped Top-K preservation beats added cost |
| F-C03 | XGBoost coarse → W&D coarse | same candidates/budget | sparse memorization gain is supported and latency-safe |
| F-C04 | W&D coarse → DeepFM coarse | same candidates/budget | automatic interactions improve pass-through/cost Pareto |
| F-C05 | DeepFM coarse → DCNv2 coarse | same candidates/budget | bounded crosses improve pass-through/cost Pareto |
| F-C06 | DCNv2 → DCNv2 + fine-teacher distillation | teacher/version frozen | teacher Top-K and high-value candidates are preserved |
| F-C07 | same coarse model, wider recall input | fine QPS held constant | candidate expansion creates business lift; this is separate from C00 |
| F-V00 | raw heads → per-head/per-slice calibration | rank formula frozen | NE/ECE improves and ordering remains explainable |
| F-V01 | single stay objective → calibrated play/stay/finish VT | candidates/models frozen | unified LT improves; negative feedback and session guardrails pass |
| F-V02 | V01 → engagement and negative-feedback terms | coefficients only | incremental LT is attributable; no task probability is used uncalibrated |
| F-V03 | global VT → versioned segment coefficients | heads and segments frozen | new/low-activity/locale slices improve without Simpson reversal |
| F-M00 | VT order → impression/session dedup | all scores frozen | duplicate load falls with non-inferior LT |
| F-M01 | dedup → author/topic/format diversity | queue load frozen | diversity improves with bounded relevance displacement |
| F-M02 | one queue → final multi-queue load/quota mixer | queue candidates/scores | every displacement is logged; Feed and business guardrails pass |

Learned retrieval follows an accepted direct fine baseline so downstream value is
measurable. Coarse ranking is not introduced by calendar date: F-C00 is eligible
only after measured fine QPS, P99, memory or candidate-width pressure breaches a
pre-registered budget.

## 3. Samples and feature ladder

These are learning-input LRs. Each holds model architecture and serving policy
constant. Contract-only rows can pass as engineering launches without claiming
business lift; any quality LR still requires factual ranking change and A/B.

| ID | Control → treatment | Required measurement |
|---|---|---|
| L-D00 | post-stage rows → full request candidate universe | closure, stage masks and byte-exact replay; no business claim |
| L-D01 | implicit zero → applicability/maturity/censor masks | fake-negative rate, label counts and calibration |
| L-D02 | random negatives → source-aware exposed/hard/in-batch negatives | false-negative rate, proposal correction and candidate-distribution metrics |
| L-D03 | pointwise only → request pairwise loss | request NDCG/Top-K and calibration guardrails |
| L-D04 | pointwise/pairwise → request listwise loss | slate NDCG/value and training cost |
| L-D05 | naive fresh negatives → delayed-feedback correction | freshness, mature-label accuracy and probability calibration |
| L-X00 | ad hoc fields → one versioned feature/FID manifest | online/replay byte parity, collision/default/TTL report |
| L-X01 | static context → route/lifecycle/provenance features | route and lifecycle slices; no hidden-state access |
| L-X02 | batch counters → PIT real-time counters | freshness, hot/new-user slices and serving miss rate |
| L-X03 | aggregates → heterogeneous short sequences | sequence ablation and request-level ranking delta |
| L-X04 | IDs/metadata → versioned content embeddings | cold/new-content and semantic-tail slices |
| L-X05 | short sequence → long event sequence | incremental horizon curve, truncation and latency/memory frontier |

## 4. Learning-system cadence ladder

Cadence is independent of model, sample and feature changes. Every row freezes
their definitions and compares checkpoint age, ranking delta, business outcome,
fallback/reject rate and GPU/IO cost.

| ID | Control → treatment | What must be proven |
|---|---|---|
| L-T00 | static artifact → daily full retrain | deterministic baseline, mature labels and rollback |
| L-T01 | daily full retrain → daily incremental/warm start | parity or gain at lower cost; no accumulated drift |
| L-T02 | daily → hourly event-time microbatch | freshness changes supported rankings and hot/new-content outcomes |
| L-T03 | hourly batch Joiner → streaming Joiner/sample bus, hourly checkpoint | exactly-once lineage/recovery; quality claim only if served ranks change |
| L-T04 | hourly sparse snapshot → streaming sparse FID/embedding updates | PS freshness benefit, shard recovery and index/model compatibility |
| L-T05 | hourly dense checkpoint → frequent validated dense checkpoints | quality/freshness gain pays for publish/reject/fallback cost |
| L-T06 | one trainer → active/candidate continuous lanes | identical eligible stream, independent cursors, safe promotion and fallback |

Faster data movement with unchanged served rankings is a systems benchmark, not a
successful recommendation LR. Streaming labels remain masked until mature; the
pipeline never turns recency into fake negatives.

## 5. Multi-business extension

After F-M02, Search, Ads, Commerce, Local, Posting, Live and photo/card/article
repeat the same sequence with independent sample spaces: eligible/random baseline,
retrieval routes, direct ranker, coarse only under pressure, calibrated primitive
value, final-mixer interaction and factual A/B. Shared infrastructure does not
permit shared labels or automatic LT exchange.

## 6. Current ledger

As of 2026-08-25, F-R00 onward have not completed factual v4 A/B. P3-06 is an
offline equal-budget retrieval review: Lifecycle remains control and Graph, RRF,
Two-Tower and Multi-interest reject. The P3 LR probe validates registry plumbing
only and remains `HOLD`. Therefore the current number of learned-model launches
with measured business uplift is zero. Future reviews append results to immutable
artifacts and update this summary; they never rewrite expected outcomes here.
