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
| F-R00 | no recommender -> eligible random retrieval and randomized order | eligibility, corpus, slate width | establishes zero-personalization baseline; no lift claim |
| F-R01 | unbiased Random route -> country Behavioral Popular route | one route per arm, single-route passthrough, eligibility and slate width | stay/session, negative feedback, market match, pool concentration and topic-periodicity gate |
| F-R02 | Popular → Popular + 30-day recent + locale/language eligibility | popularity formula | freshness/locale slices improve without coverage collapse |
| F-F00 | randomized candidate order -> direct rule fine rank | small fixed candidate set | served Top-K changes; stay improves within latency budget |
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

## 5. Reliability and bug-fix ladder

Bug fixes can be launched and reviewed like business changes, but their claim is
**recovered correctness or avoided loss**, not new algorithmic lift. A bug that
invalidates assignment, labels, features or metrics invalidates the affected old
LR; fix it first and rerun that LR instead of opening a convenient new treatment.

| ID | Control → treatment | Required evidence |
|---|---|---|
| B-AB00 | broken/unknown assignment → deterministic atomic assignment | A/A, SRM, one factual commit and contamination audit |
| B-LOG00 | ambiguous selection probability → typed factual/randomized propensity and support | exact replay and unsupported-policy abstention |
| B-JOIN00 | premature/default-zero labels → maturity, censor and delayed-attribution masks | label correction counts, no future leakage and recalibration |
| B-FEAT00 | train/serve divergence → one exact feature/FID authority | byte parity, missing/default/TTL and score replay |
| B-CAS00 | silent cascade loss → complete candidate/stage decision log | stage closure and restored high-value pass-through |
| B-MIX00 | duplicate/double-claimed exposure → one final-slate owner | impression reconciliation and displacement trace |
| B-IDX00 | model/index/corpus mismatch → compatibility-gated snapshot | pre-score rejection, fallback and exact rollback |
| B-PS00 | stale/lost sparse shard → versioned shard recovery | freshness, checksum, replay and recovery-to-baseline |
| B-LAT00 | timeout/fallback amplification → bounded latency and explicit fallback | P50/P99, fallback rate, ranking distribution and business recovery |
| B-CKPT00 | process-local replay → content-addressed world checkpoint and fork | next-tick tensor parity across hidden world, platform, events, experiment and learning cursors |
| B-LR00 | HOLD incorrectly advances route → pending cumulative launch cursor | same experiment assignment and analysis start survive restore; max-window stop is pre-registered |

Urgent correctness fixes may use canary or switchback rather than withholding a
known fix from half the users. The review still records pre-fix loss, recovery,
confidence interval, rollback and which prior reports became invalid. Simulator
or evaluator defects that do not affect a factual product policy are engineering
reviews only and cannot claim user-value uplift.

## 5.1 Runtime and scale ladder

Runtime changes are independently reviewed system LRs. Their primary claim is
bounded cost, reliability or capacity; they may claim business non-inferiority
only when the served policy and factual assignments are held fixed. An OOMing
implementation is not a valid long-running control, so unsafe paths first pass
small-world exact parity and standard-world shadow/replay before replacement.

| ID | Control -> treatment | Fixed budget and acceptance |
|---|---|---|
| S-AUTH00 | implicit formula fallback -> explicit artifact-bound factual response authority | formula remains test-only; factual quality runs fail closed without the declared NeuralSCM artifact; support fallback is logged and bounded |
| S-WORLD00 | tick-zero registration surge -> equilibrium population initialization and burn-in | stable hour/weekday/locale/cohort traffic, restart parity and experiment duration powered from steady-state users rather than bootstrap traffic |
| S-MEM00 | dense 30-day exposure matrix -> rolling segmented Bloom plus 128-item exact session cache | 100K users/1M items; zero session repeats, measured Bloom FPR, bounded RAM/VRAM and exact small-world parity |
| S-ROUTE00 | eagerly execute every registered route -> execute only enabled routes | byte-identical enabled-route candidates and scores; lower latency/VRAM; disabled routes perform no ANN or business work |
| S-TRACE00 | fixed-width whole-tick candidate trace -> actual-width trace and projection-free request partitions | exact request/candidate closure and replay hashes; bounded bytes/request |
| S-IO00 | uncompressed request tensors -> streaming zstd partitions | byte-identical replay, content hash and crash-safe commit; lower storage without a whole-object memory copy |
| S-EVENT00 | process-resident GPU event history -> CPU/disk-backed event partitions plus bounded hot window | event order/idempotency/watermark and checkpoint restore parity; GPU history memory is constant in elapsed ticks |
| S-MICRO00 | monolithic request tick -> bounded request microbatches with one atomic tick commit | identical assignments, slates, structural-noise outcomes and next-world state across microbatch sizes; peak RAM/VRAM stays under budget |
| S-CKPT00 | full raw tensor snapshot each launch -> compressed full base plus incremental state/event generations | restart/branch/hash parity, bounded save/load RSS, measured compression and restore latency, garbage collection retains reachable lineage |
| S-LONG00 | four-tick launch proof -> 96-tick continuous world soak | no memory growth with elapsed ticks, zero swap/OOM, restart parity, event/sample closure and stable P99 |

Bloom dimensions, hash count, segment duration and rotation are tunable system
parameters only after S-MEM00. They are never changed inside an algorithm LR.
The canonical workload is one fixed profile (`feed-standard-v1`: 100K users,
1M items, 96 ticks/day); smoke/local profiles are tests, not competing evidence.

## 6. Multi-business extension

After F-M02, Search, Ads, Commerce, Local, Posting, Live and photo/card/article
repeat the same sequence with independent sample spaces: eligible/random baseline,
retrieval routes, direct ranker, coarse only under pressure, calibrated primitive
value, final-mixer interaction and factual A/B. Shared infrastructure does not
permit shared labels or automatic LT exchange.

## 7. Current ledger

As of 2026-08-26, S-MEM00, S-ROUTE00, S-TRACE00 and S-IO00 have executable
standard-profile evidence. The standard Random tick renders 65,063 requests in
1.04 seconds with 7.687 GB peak VRAM; request partitions fell from about
1.735 GB to 100.5 MB per tick, and the initialization checkpoint fell from
7.6 GB to 3.1 GB. These are system results, not user-value lift.

F-AA-00 exposed an invalid primary-only A/A gate: dwell covered zero while the
share family member failed. The immutable review remains failure evidence.
F-AA-01 added the pre-registered Bonferroni metric-family gate and passed four
ticks with SRM p=0.19948, zero cross-cell contamination and zero repeated Feed
impressions. F-R00 was submitted on the RTX host, but its completion is not
claimed until the immutable remote journal and branch head are readable. That
submission used the superseded personalized-rule baseline and therefore cannot
establish F-R00; the corrected runtime uses random retrieval plus
`exploration_rate=1.0`.

S-WORLD00 now has equilibrium burn-in, bounded event partitions and exact
next-tick checkpoint parity. S-AUTH00 remains the only blocker before factual
TikTok World launches: historical v23 bytes are absent, so v24 must re-earn the
authority gates. Its structural builder now owns one Markov world, trains only
on factual families and computes three request-local counterfactual deltas on
the held-out test family. Once v24 shadow, restart parity and F-AA pass, the
next immutable LR is corrected `Random -> Popular`; model and feature work then
enters as subsequent LRs rather than further simulator expansion.

P3-06 is an
offline equal-budget retrieval review: Lifecycle remains control and Graph, RRF,
Two-Tower and Multi-interest reject. The P3 LR probe validates registry plumbing
only and remains `HOLD`. Therefore the current number of learned-model launches
with measured business uplift is zero. Future reviews append results to immutable
artifacts and update this summary; they never rewrite expected outcomes here.
