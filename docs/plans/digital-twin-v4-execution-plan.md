# Recommendation Digital Twin v4 — Execution Plan

Status: active execution authority

Updated: 2026-08-25 (all remaining rows researched; pre-ladder contracts reopened)

Scope: synthetic engineering and interview reference; no production or internal-company claims

This plan turns the v4 architecture into an ordered, measurable rebuild. It is the
single authority for unfinished v4 work. The architecture document owns invariants;
this file owns delivery order, status, evidence and acceptance.

## 1. Evidence boundary

Every statement in reviews and reports must be classified as one of:

| Evidence class | Meaning |
|---|---|
| User-supplied target | Desired reconstruction of the user's known workflow; not a public claim |
| Public evidence | A cited paper, official engineering post or public dataset |
| Executable evidence | A content-bound local or RTX 4090 artifact produced by this repository |
| Design assumption | A testable synthetic choice with no claim of matching a company secret |

The following are user-supplied target requirements: TikTok-style Feed streaming
training, a normal Feed corpus limited to content created in the last 30 days,
separate cold-start/hot/evergreen retrieval authorities, orthogonal layer
experiments, full-chain ClickHouse drill-down and the Feed posting/consumption
loop. They must not be presented as facts discovered from public sources.

Public evidence establishes the transferable principles:

- RecFlow retains exposed and unexposed candidates filtered at six funnel stages.
- KuaiSim separates request-level, whole-session and cross-session response tasks.
- SARDINE explicitly models recommendation-induced behavior and feedback-biased
  future training data.
- Monolith couples real-time feedback, online training and sparse embedding
  lifecycle management.
- Instagram Explore publicly describes multi-source retrieval, multi-stage
  ranking, multi-task prediction and continual model refresh.
- Public Google work treats content cold start as a content-understanding problem
  and separately studies ranking for content-generation value.

## 2. Current-state audit

The audit below is the starting point. `Draft` means code exists in the working
tree but has not passed the repository acceptance path. `Implemented` means the
focused contract is executable; it does not mean the phase, benchmark or launch
review is accepted.

The accepted implementation baseline is
`f71ec908955d853cbf13d361975d7e2be6be47b6`; local and `origin/main` agree.
P0, P1, the declared Feed scope of P2 and P3-01..06 have content-bound reviews.
The clean committed source reproduces 202 historical tests, 70 v4 tests and the
repository gate. Architecture lint has zero errors and one declared warning
because this project uses its own `AssetGraph`, not Dagster decorators.

The accepted P3 RTX 4090 run covers 100K users, 1M items and two event-time
partitions at cascade width 96→48→16→8. It persists 1,267,664 fine rows and
7,808,576 mature labels, consumes both partitions independently in active and
candidate lanes, completes in 88.266 seconds and peaks at 8.538 GiB CUDA. This
is streaming/feature/registry evidence bound to the accepted source. The candidate
LR probe remains `HOLD` and provides no model or A/B lift claim.

The P3-06 RTX run uses 100K users, 1M items, four event-time partitions,
255,046 training queries and 10,000 evaluation queries. It finishes in 190.227
seconds at 14.840 GiB CUDA. Lifecycle remains control; Graph, RRF, Two-Tower and
Multi-interest reject. Only 89 different evaluation positives expose a severe
old-policy logging ceiling; randomized retrieval truth remains P3-09.

A semantic audit after P3-06 found four blockers that marker searches could not
detect. `CoarseRankExample.item_id` starts from old-policy `coarse_item_id`, not
the recall universe. `FineRankExample` retains only exposed items although the
scorer sees the wider coarse set. The trace stores features only after fine
selection. Finally, deterministic selected items receive
`exposure_probability=1`; this is factual selection, not counterfactual support.
P3-02/03/05 remain accepted for their earlier mechanics but reopen as
P3-02a/03a/05a before any learned ranking claim.

| Capability | Current evidence | Status | Blocking gap |
|---|---|---|---|
| Hidden user world boundary | Platform cannot directly read hidden preference state | Accepted | Non-Feed worlds remain separately gated |
| Atomic factual A/B world | One request receives one factual policy and commits once | Implemented | Longer-horizon interference tests remain incomplete |
| Delayed outcomes | Order/payment/refund/Pixel occurrence and ingestion time are distinct | Implemented | Production-like loss/duplicate/orphan distributions need calibration |
| Point-in-time projection | Delivered events, lifecycle and retained feature tensors replay across content-bound partitions | Partial for learned ranking | Full scoring-input tensors are not retained |
| Request cascade trace | Raw routes, lifecycle, lineage and cascade stages are retained | Implemented | Candidate propensities and scoring-input tensors remain |
| Feed retrieval mechanics | Lifecycle/Graph plus registry-backed Two-Tower/Multi-interest serving path | Accepted through P3-06 | Every challenger rejected; randomized retrieval truth remains P3-09 |
| Layered experiments | Ownership, independent assignment, composed policy and numeric served checkpoint logging | Implemented | Model-learning interference remains P4-05 |
| Feed post creation | Immutable `post_id`, source lineage, capacity/cooldown/exit failure and future Feed trace | Implemented | Rich media processing belongs to P5 Posting |
| Content lifecycle | Observable 30-day recent, cold-start, hot, evergreen, expired, moderation and deletion | Implemented | Threshold calibration belongs to P2 |
| Public catalog anchors | Product/POI lineage is typed through projection and events | Implemented for P1 | Post media/semantic processing belongs to P5 Posting |
| Behavior realism | v23 passes external, held-out-family, support, anti-exploitation and 100K semantic-shadow gates | Accepted for Feed | Retention, creator supply and every business response remain masked |
| Full-chain analytical store | Schema v4 persists retained feature/FID and task vectors; DuckDB/ClickHouse agree | Accepted for retained rows | Full scoring-input coverage remains P3-05a |
| Recall/coarse/fine sample authorities | Recall is source-corrected; coarse/fine contracts replay | Recall accepted; coarse/fine reopened | Coarse starts after old Top-K; fine is exposure-only |
| Continuous learning | Persistent dual lanes, cursors, registry, compatibility, fallback and serving adapter | Accepted for P3-04/05 | Cadence lift remains P4-04 |
| Model ladder | v4 retrieval ladder compares one factual dataset and budget; every challenger rejected | Accepted through P3-06, no learned retrieval launch | Coarse/fine ladders and randomized retrieval truth remain P3-07..09 |
| Search/Ads/Commerce/Live/Local/Posting | Surface actions and catalog types exist | Skeleton | Each lacks a closed business workflow and independent launch contract |

The accepted v23 NeuralSCM is the declared Feed response authority only. Its
KuaiRand path is user-disjoint, PIT-safe and support-gated; its structural path
uses real cascade requests, independent seeds and held-out world families. The
100K-user/1M-item/eight-tick shadow covers `98.9835%` of 412,006 requests and
passes replay gates with 2.46 GiB peak RSS. Retention, creator supply, Local,
Ads, Commerce and LT remain masked. No route or business-model LR follows from
this promotion.

## 2.1 Research disposition

The research question is not whether a larger handcrafted DGP can make a neural
ranker win. That would couple the evaluator to the desired model. The reviewed
public systems establish a more defensible division of responsibility:

| Evidence | What is adopted | What is explicitly not inferred |
|---|---|---|
| RecSim / RecSim NG | Entity-behavior-story decomposition, latent stochastic state, multi-agent ecosystem and accelerated vectorized execution | A configurable simulator is not automatically realistic |
| SARDINE / RecLab | Recommendation changes future observations; policy order can change under feedback | Offline AUC is not online truth |
| KuaiSim / RL4RS / Virtual-Taobao | Request, session and cross-session dynamics; learned response plus reality-gap and exploitation checks | One learned response model is not a causal oracle |
| AAAI-25 LLM user simulator / RecInter | Semantic scenario generation, memory and environment-changing actions are useful challengers | Token agents are not the high-throughput numerical A/B authority |
| Google recommender-ecosystem work | Consumer, creator, merchant and advertiser utilities interact over time | Creator value cannot be reduced to one immediate engagement label |
| Google content-generation ranking | Sparse posting intent, conditional losses and personalized value are legitimate ranking targets | Public evidence does not reveal a proprietary production formula |
| Monolith / Instagram Explore | Streaming feedback, multi-stage ranking and frequent refresh can create measurable freshness value | Faster training is not a launch unless served rankings and A/B outcomes change |

Decision: keep the vectorized event kernel, but replace its final behavior
authority with a calibrated partially observed neural structural causal model
ensemble. Retain the deterministic formula world only for invariant tests. Add
LLM agents only as a low-volume semantic and adversarial scenario lane.

### 2.2 Reuse audit and closed design decisions

The repository already contains more implementation than the v4 package exposes.
The remaining program is therefore an authority migration, not a greenfield
rewrite. Existing artifacts are reusable only after they consume v4 contracts
and pass v4 gates; an old synthetic launch report is not evidence for the new
world.

| Concern | Existing reusable authority | Closed decision |
|---|---|---|
| Neural user world | `fid_lab/feed_loop/world_model`: slate attention, censored stay mixture, multi-action heads, latent transition, ensemble and paired structural noise | Adapt these components behind the v4 hidden-world boundary. The current unified challenger is `HOLD`, so it cannot silently replace `world/behavior.py`. |
| External falsification | KuaiRand standard/random adapters, sequence benchmarks, shadow worlds and DR OPE under `world_model/external/kuairand` | Reuse the content-bound pipeline. KuaiRand constrains Feed behavior only; it does not validate Local, supply, retention, ads or LT exchange. |
| Continuous learning | `fid_lab/simulation/twin/training`: event window, Joiner, LR/W&D/DeepFM/DCNv2/MMoE trainer, registry and mixed-world loop | Port the trainer and registry behind v4 samples/checkpoints, then delete the legacy execution path. Do not create another trainer stack. |
| Model implementations | Historical retrieval, coarse, fine, posting and Local ladders | Reuse architecture code only. Every model is retrained on the same v4 factual request dataset, corpus, candidates and budget; historical lifts remain non-authoritative. |
| Value and LT | `fid_lab/value`, historical mixer and LT reports | Reuse formula/metric code after contract review. Value Tree ranks calibrated primitives; unified LT measures A/B outcomes and is never a supervised target. Historical Local-to-LT conversion is invalid. |
| Experimentation | v4 atomic assignment/composition plus legacy CUPED launch analysis | Extend one experiment authority with compatibility, power, SRM, interference and shared-training-data checks. Control and treatment never execute sequentially on one factual user world. |
| Multi-business | Shared v4 identity, catalog, event, experiment and logging contracts | Share infrastructure, not sample spaces or value definitions. Search, Ads, Commerce, Local, Posting and Live each own candidates, labels, model and Launch Review. |
| Performance | PyTorch tensor kernel, Arrow/Parquet partitions, ClickHouse diagnostics and RTX 4090 evidence | Profile before adding Rust/C++. GPU-vectorize hot numerical paths; use compiled code only for a measured CPU/serialization bottleneck with semantic-parity evidence. |

Research also closes four scope questions. RecFlow justifies retaining candidates
lost at every funnel stage. Monolith justifies measuring freshness against
reliability rather than assuming streaming wins. Google overlapping experiments
justify namespaces but not causal independence; shared training data can create
symbiosis bias, so data-diverted or corpus-co-diverted designs are required for
selected model-learning experiments. HSTU, TIGER and OneRec remain challengers
after the conventional cascade is identified; they are not P2 dependencies.

### 2.3 Backlog research closure

Eighteen rows remain, plus P3-02a/03a/05a contract corrections. “Research closed”
means the next falsifiable implementation and rejection boundary are known, not
that implementation or acceptance exists. Execution is governed by section 9.1.

A marker audit found no unchecked box or `NotImplementedError` under active code
and docs; the semantic audit nevertheless found the three reopened contracts.
`docs/interview/recommendation-data-contracts.md` now maps request-native
pairwise/listwise tuning explicitly to P3-08/P3-09 rather than creating another
backlog. Other `pending` values are event/report states. The three dated
plan files explicitly defer here and cannot create a second backlog.

| Backlog | Research conclusion | Adopted decision | Rejected shortcut |
|---|---|---|---|
| P2 user-world validity | RecSim NG supports modular latent multi-agent worlds, while KuaiSim and SARDINE show that request, session, cross-session, uncertainty and feedback loops must be tested separately. No public simulator guarantees live realism. | Keep the hidden probabilistic ecosystem and evaluate it as multiple falsifiable families. Calibrate only observed Feed behavior; treat geography, creator response, retention and LT as declared synthetic stress families until separate evidence exists. | Fitting one universal DGP to KuaiRand-1K or tuning it until MMoE wins. |
| P2 randomized evidence | KuaiRand provides random inserted exposures and known item-pool support. OPE is identified only for actions with logged propensity and overlap; randomized item exposure does not reveal counterfactual hidden-user transitions or Local outcomes. | Bind the v4 Feed challenger to a user-disjoint randomized calibration/evaluation split. Use DR/SNIPS policy estimates with ESS, maximum-weight and support gates. Keep structural intervention tests as a separate multi-world robustness gate. | Fabricating three interventions from observational rows or calling randomized Feed evidence universal causal validation. |
| P3 full-flow samples | RecFlow retains candidates lost at six funnel stages; sampled-softmax research shows that in-batch proposal bias and the always-present positive need explicit treatment. | Preserve the factual candidate universe and every stage decision; build Recall, Coarse and Fine examples from one request authority. Store source-specific proposal probability, false-negative risk and correction terms instead of inferring them in the learner. | Training every stage from exposed rows, applying one generic log-Q formula to mixed sources or comparing different candidate budgets. |
| P3 continuous learning | Monolith supports event-time online training, versioned sparse state and an explicit reliability/freshness trade-off. | Port the existing active/candidate trainer to v4 partitions. Every request logs the served checkpoint and both lanes consume the same eligible factual stream with watermark, resume and fallback. | Freezing models for a long A/B or claiming that faster checkpoints are a business launch without ranking delta. |
| P3 feature/FID authority | Monolith identifies sparse-state lifecycle and collision handling as system concerns; Meta's event-based features standardize heterogeneous sequences by stream, length and event schema. Neither source validates an internal FID layout. | One versioned manifest owns field semantics, point-in-time source, transform, bucket/hash, vocabulary, default, TTL and serving owner. Hash collisions are measured per namespace and the exact serving features are replayed offline. | Treating raw IDs as continuous values, exposing hidden-world state, or allowing train/serve code to define the same feature independently. |
| P3 model ladders | Public retrieval, sampled-softmax and multi-stage work support hard negatives, proposal correction and equal-budget comparisons; no paper says a deeper model must win. | Run retrieval, coarse and fine ladders only after sample parity. Freeze corpus, candidates, tuning budget and latency; require one explained pass, hold and reject. | Shaping the DGP for W&D/MMoE or treating global AUC as launch evidence. |
| P3 evaluation | RecFlow demonstrates stage-selection bias; cascade DR work requires the logging policy, examination model and support to match the ranked action. Global AUC cannot localize a cascade failure. | One request-aware evaluator replays the served slate and reports stage pass-through, request GAUC/NDCG, PR-AUC, calibration/NE, Top-K delta, latency and slices. Emit IPS/SNIPS/DR only when the exact logged action has support. | Treating position heuristics as propensities, comparing sampled random negatives or extrapolating unsupported policies. |
| P4 overlapping experiments | Overlapping experiments scale orthogonal changes, but shared learning data creates symbiosis bias and violates hidden-treatment assumptions. | Use namespaces plus a compatibility graph for normal parameter tests; use data-diverted, cluster or user-corpus co-diverted designs when the learned algorithm itself is the treatment. | Calling all layer experiments causally independent because their assignment hashes differ. |
| P4 Value/LT | Public work uses immediate behavioral surrogates only after validating their relationship to later revisits; short-term effects can differ from learned long-term effects. No source provides a universal LT formula. | Value Tree ranks calibrated primitives. Unified LT is measured after randomization with cohort curves, lagged outcomes, nonnegative synthetic exchange assumptions and sensitivity; it is never a request label. | Directly predicting request LT, selecting exchange weights on the test experiment or relabeling Local/GMV/ads value as LT. |
| P4 final-slate policy | Heterogeneous-feed work supports author, semantic and format diversity, but does not provide a transferable COPP formula. | One mixer owns impression/session dedup, format/business queues, quotas, diversity, eligibility, load and displacement. Every removed candidate records the exact constraint, before/after rank and displaced alternative. | Independent queues each claiming the same exposure, post-hoc logs without counterfactual displacement, or permanent business boost hidden inside model scores. |
| P5 Search | Public search systems combine lexical and semantic retrieval, query understanding/rewrite, vertical triggering and blending. Search intent and success differ from passive Feed response. | Build query/session state, lexical+semantic routes, reformulation and success labels, then connect only observed search intent and post-search behavior back to Feed. | Reusing Feed candidates/labels or evaluating Search only by CTR. |
| P5 Ads | Public Ads systems retain a retrieval→ranking funnel and predict both user and advertiser value under strict latency; auction, pacing and budgets introduce interference absent from organic Feed. | Model eligibility, bid, pacing, budget, retrieval, ranking, mixing, click and delayed conversion as one closed workflow with revenue and Feed-experience guardrails. | Treating Ads as an organic queue with a score boost or ignoring budget exhaustion and advertiser-side value. |
| P5 Commerce | Industrial e-commerce retrieval uses stage-specific ordered/clicked/unclicked/random signals; purchase and refund mature later than browsing. | Close shelf→detail→cart→order→payment→refund with inventory snapshots, conditional labels and deterministic order/payment lineage. | Marking every unclicked exposure as CVR-negative or counting unpaid/refunded orders as durable value. |
| P5 Local | Public POI work supports geo, category, graph and diversity signals, but no public source identifies this repository's open-loop attribution process. | Own POI/video/anchor eligibility, detail/map/YMAL, closed-loop order and separately observed Pixel events. Keep identity loss, attribution coverage and orphan rates explicit. | Treating nearest POI as intent, imputing missing Pixel feedback as zero, or converting Local actions directly into LT. |
| P5 Posting | Public content-generation ranking uses sparse participation targets, proxy intent, conditional losses and personalized value. It does not identify a proprietary creator experiment framework. | Separate draft→candidate→select→publish recommendation from Feed distribution; use proxy/qualified-publish tasks, future-supply outcomes and user/creator-randomized experiments with interference declared. | Calling product-entry lift algorithmic relevance or training publication on future distribution outcomes without point-in-time controls. |
| P5 Live/formats | Live has availability, concurrent actions and fresh/delayed outcomes; photo/card/article have different examination and dwell semantics. Multi-surface content encoders can share representations without sharing response labels. | Give Live, photo, card and article separate eligibility, examination, labels and calibration; share only versioned content representations and infrastructure. | Reusing video completion thresholds or one universal response head across formats. |
| Orchestration | The repository already has a logical asset graph and content-addressed partitions. A second scheduler does not improve semantics by itself. | Extend the existing graph with v4 sample, training, evaluation and publication assets. Add Dagster or another external scheduler only when cross-process retries, backfills or remote execution become a measured operational requirement. | Introducing a parallel DAG authority while the v4 learning path is still disconnected. |
| P6 scale/runtime | RecSim NG demonstrates accelerated probabilistic simulation; the current P1 run shows Arrow partitioning and GPU tensorization are viable. | Keep PyTorch/Arrow/Parquet as the authority, profile at 100K/1M/10M, tensorize measured hot paths, and introduce Rust/C++ only for a proven CPU or serialization bottleneck with parity. | Rewriting the simulator in Rust/C++ before a profile or using item/user count as the only realism measure. |
| P6 reliability/release | Large recommendation fleets require a model registry plus per-head calibration/discrimination health; feature freshness, snapshot quality and index compatibility can fail while requests return 200. | Seed index/model mismatch, stale features, bad snapshot, PS loss, delayed labels, timeout, overload and fallback. Accept recovery only when lineage, per-head quality and business metrics return to baseline; then delete legacy paths after content-bound parity. | Calling availability model health, preserving two runtime authorities indefinitely, or accepting rollback without replay evidence. |

### 2.4 Execution stack disposition

| Boundary | Decision now | Trigger for another mature wheel |
|---|---|---|
| Event/sample storage | Keep PyArrow/Parquet manifests and DuckDB pushdown; they already provide partitioned, lazy, content-bound replay. | Add Flink only when a replayable external source, cross-process state and transactional/idempotent sink must recover together. |
| Feature store | Compile one repository-owned PIT feature/FID manifest to online tensors and replay tensors. | Add Feast only when a real online store and multiple serving consumers exist; its PIT semantics are adopted now, not a second registry. |
| Training/ANN | Keep PyTorch, XGBoost and FAISS on the 4090. | Add TorchRec only when sparse tables no longer fit one GPU or measured sharding/communication is required; add Ray only for proven multi-process or multi-node scheduling need. |
| Registry/tracking | Keep immutable artifact manifests plus active/candidate/fallback aliases in the v4 registry. | Add MLflow only when a persistent multi-user service/UI is required; aliases cannot replace feature/FID/index/corpus compatibility gates. |
| Orchestration | Extend the existing logical `AssetGraph`; one asset declares inputs, outputs, resume and evidence. | Add Dagster only for remote execution, operational backfills or cross-process retries; never keep two DAG authorities. |

The causal acceptance contract is therefore changed from one ambiguous gate into
two independent claims:

1. **External policy evidence**: at least three full-support Feed policies are
   ordered on the user-disjoint KuaiRand randomized evaluation split with a
   declared propensity, ESS threshold, bounded importance weights and confidence
   intervals. The new neural world artifact must be content-bound as the outcome
   model; an OPE report bound to an older model does not count.
2. **Structural robustness**: at least three mechanism interventions are run on
   multiple frozen synthetic world families that were not used to train or tune
   the challenger. Sign, magnitude, batch/order invariance and exploitation
   probes must pass. This tests robustness; it is explicitly not external causal
   truth.

Failure of either claim keeps the neural authority on `HOLD`. Passing both can
promote only the declared Feed behavior scope; retention, creator supply, Local,
Ads, Commerce and LT remain separately gated.

## 3. Target system and invariants

### 3.1 Three independent systems

```text
Hidden ecosystem
  users, creators, content truth, merchants, advertisers, trends, calendar
        │ typed commands and observable events only
        ▼
Platform
  catalog/index → retrieval → coarse → fine → value/mix → exposure
        │ append-only factual logs
        ▼
Learning and experimentation
  projection → Joiner → samples → active/candidate trainers → checkpoints
        │ validated artifacts and experiment parameters
        └──────────────────────────────────────────────────────────────► Platform
```

The hidden ecosystem may observe the rendered product experience. It must not
read model scores, FIDs, experimental estimates or platform belief state. The
platform may observe events and derived point-in-time features. It must never
read latent preference, true content quality or future outcomes.

### 3.2 Feed posting/consumption loop

This is the first complete business loop to build:

```text
creator session
→ upload/edit draft
→ optional POI/product/topic selection
→ media understanding and safety processing
→ immutable post creation
→ cold-start eligibility and exploration
→ recent corpus (age <= 30 days)
→ possible hot or evergreen promotion
→ multi-route retrieval
→ coarse/fine/value/mix
→ impression/play/stay/slide/3s/finish/like/favorite/share/comment/follow/hide
→ creator impressions, engagement, retention and next-post decision
→ future supply
→ point-in-time Joiner and continuous training
```

Required invariants:

1. Choosing an anchor is not the publish event and is not the post identity.
2. A publish event creates one immutable `post_id` with `creator_id`,
   `publish_time`, media, semantics, anchors and safety/eligibility versions.
3. Every Feed exposure points to that same `post_id`; creator feedback aggregates
   only from factual downstream events.
4. Main recent-corpus membership is `0 <= request_time - publish_time <= 30d`.
5. Cold-start, hot and evergreen are separate lifecycle authorities, not aliases
   for arbitrary freshness/popularity formulas.
6. No content can enter a route without an auditable lifecycle and eligibility
   decision.

### 3.3 Core Feed route ownership

The target Feed baseline is deliberately smaller and semantically clean:

| Route | Corpus authority | Primary purpose |
|---|---|---|
| Recent ANN | eligible recent posts | semantic/personalized relevance |
| Recent Graph/I2I | eligible recent posts | behavioral continuation and co-watch |
| Following/Author | eligible posts from followed/affine creators | creator relationship |
| Cold-start | newly published, insufficient-feedback posts | exploration and supply feedback |
| Hot/Trending | time/region/topic-specific promoted posts | fast trend response |
| Evergreen | explicitly promoted older posts | durable high-value supply |

Geo/POI is owned by the Local business queue. Search, retargeting, ads, commerce
and live use their own trigger and eligibility contracts. They may enter the final
mixer but must not be mislabeled as core Feed retrieval routes.

## 4. Full-chain observability authority

This phase precedes additional model experiments. The simulator must answer one
request by data, not by rerunning Python and guessing.

### 4.1 Durable tables

| Logical table | Required grain and fields |
|---|---|
| `v4_request_log` | one request; user, surface, event time, context, experiment assignments, policy/artifact versions |
| `v4_route_candidate_log` | request × route × raw candidate; route score/rank, lifecycle, eligibility, sampling probability |
| `v4_candidate_decision_log` | request × item; every stage admission/rank/score/drop reason, scoring tensors, factual-selection kind and logging propensity only when randomized |
| `v4_event_log` | one factual event; occurrence/ingestion times, request/item/post/creator/order/payment IDs, dedup key |
| `v4_mature_label_log` | request × item × task; label, value, maturity time, mask, attribution and censor reason |
| `v4_training_example_log` | example identity and authority; feature/FID manifest, sequence watermark, label manifest, sampling correction |
| `v4_checkpoint_log` | model lane, data watermark, sample manifest, feature/FID/index versions, validation, publish/fallback state |

Every table is append-only at its source boundary. Corrections are new versioned
records, not silent mutation. Parquet is the portable source artifact; ClickHouse
is the analytical serving layer. A local DuckDB/Polars fixture validates SQL
semantics when a ClickHouse server is unavailable.

### 4.2 Required investigations

The repository must ship executable SQL or equivalent fixture tests for:

- one `request_id` end-to-end reconstruction;
- route volume, overlap and marginal unique relevant coverage;
- age, format, author head/tail, region and lifecycle distributions;
- recall→coarse→fine→mix→exposure pass rates and drop reasons;
- score, rank, calibration and Top-K overlap distributions;
- high-value candidate loss at every stage;
- feature/FID online-versus-replay differences and missing/default rates;
- label maturity, late arrival, duplicate, orphan and attribution coverage;
- model/index/checkpoint age, fallback, timeout and version mismatch.

Acceptance: a seeded fixture contains a known recall miss, coarse loss, fine
misorder, mixer displacement, immature label and version mismatch; each query
must identify the intended root cause independently of the producer code.

## 5. User and supply world work

### 5.1 User arrival, sessions and churn

- Add country, timezone, language, lifecycle, activity and device mixtures.
- Generate diurnal/weekly calendars and event-driven sessions, not one fixed
  request per synthetic row.
- Distinguish new, low-active, returning and high-active users.
- Model leave, next-request delay, next-session delay and permanent churn.
- Make response depend on sequence pattern, semantic affinity, fatigue, trend,
  novelty, creator relation and outside option through nonlinear latent
  interactions unavailable to platform features.

Acceptance: held-out family plots cover request, session and cross-session
statistics; hidden-state ablation must reduce learnability without collapsing
all observed AUC to random or making logistic regression near-perfect.

### 5.2 Content and creator lifecycle

- Replace independent reserved-item publication with deterministic post creation.
- Add media format, language, topic/entity, content embedding, duration, creator
  attributes and optional POI/product anchors.
- Add creator posting cadence, effort, topic choice, response to distribution,
  creator retention and content deletion/moderation.
- Add exogenous and endogenous trends with region/topic/time scope.
- Implement cold-start→recent→hot/evergreen/expired transitions.
- Preserve counterfactual independence: the world may react to exposure and
  actions but cannot optimize directly against model internals.

Acceptance: supply changes under factual policy exposure, but identical world
seeds remain invariant to batching, arm execution order and unused model scores.

## 6. Streaming samples and learning

### 6.1 Event and sample path

```text
factual event log
→ watermark-aware platform projection
→ request/candidate closure validation
→ label maturity and attribution
→ RecallExample / CoarseRankExample / FineRankExample
→ partitioned replayable sample bus
→ active and candidate trainer lanes
→ validation and checkpoint publication
→ shadow/replay/canary/A-B
```

All experimental arms contribute factual observations to the common stream.
Active and candidate models train from the same eligible stream and may use
different code/configuration; models are not frozen for the duration of an
experiment. Every request records the exact checkpoint it saw. Evaluation must
handle changing checkpoints and experiment exposure, rather than pretending the
whole experiment used one static artifact.

### 6.2 Three sample authorities

- `RecallExample`: query context, positive post, proposal source, sampled
  negatives, proposal probability, behavior strength and corpus snapshot.
- `CoarseRankExample`: the full recall universe, route scores, old-stage
  admission/rank, masked factual labels and typed factual/shadow teacher scores.
- `FineRankExample`: the full fine-scorer input set, exact PIT tensors and
  sequences, stage/exposure masks, mature labels and identified propensities.

Unexposed candidates never receive fake behavior negatives. Conditional funnel
tasks use masks or entire-space factorization. Delayed outcomes remain censored
until mature. Training examples retain `request_id + post_id` lineage and the
complete feature/FID manifest.

### 6.3 Cadence ladder

Cadence is an independently launched system change:

```text
daily batch → hourly nearline → event-stream sparse updates
→ more frequent validated dense snapshots
```

Each Launch Review holds model structure, features and traffic constant and
reports LT/stay/engagement, hot/new-content slices, checkpoint age, PS staleness,
fallbacks and resource cost. A faster pipeline that changes no served ranking or
business outcome is a system benchmark, not a successful launch.

## 7. Learned recommendation ladder

Only start after the request-level dataset and replay parity gates pass.

### 7.1 Retrieval

1. Popular/recent/following/cold-start/hot/evergreen baseline.
2. Graph/I2I on observable co-watch and engagement edges.
3. Two-Tower using corrected in-batch, exposed and mined hard negatives.
4. Multi-interest retrieval for distinct short/long/session interests.
5. Semantic-ID generative retrieval as a separate route.

Each version shares corpus snapshot, query set, Top-K, route quota and latency
budget. Report Recall@K, valid recall, marginal unique positives, head/tail,
freshness, lifecycle coverage and downstream fixed-ranker impact.

### 7.2 Coarse ranking

```text
rules/logistic regression → XGBoost → W&D → DeepFM → DCNv2
→ DCNv2 plus fine-ranker distillation
```

Training candidates must come from factual retrieval logs. Acceptance includes
fine Top-K pass-through, rare high-value label pass-through, calibration,
latency and route/lifecycle slices. A model cannot win by seeing a larger
candidate budget or hidden world fields.

### 7.3 Fine ranking

```text
logistic regression/XGBoost
→ W&D/DeepFM/DCNv2
→ DIN candidate-aware short sequence
→ Transformer long sequence
→ Transformer + MMoE
→ Transformer + PLE
→ HSTU-style long-sequence experiment
```

Features must cover user, item/post, creator, request/context, route, real-time
counters, short/long sequences, content embeddings and business primitives.
Models predict play/stay distribution, 3-second/finish, engagement, negative
feedback, creator/posting intent and eligible business actions with correct
conditional masks. Selection uses request-aware ranking metrics, calibration,
business slices, latency and A/B—not global AUC alone.

### 7.4 Value and policy

The model emits calibrated primitive probabilities and expected magnitudes.
Value Tree, COPP/constraints, quotas, deduplication, diversity and cross-business
mixing are separate versioned policy owners. Unified LT is the final A/B value
container; it is not a supervised label or a hand-authored model target.

Policy experiments occur after predictive parity so model and coefficient
effects remain identifiable. Coefficients may be nonnegative as required, but
must be fitted/argued from experimental exchange evidence and guarded by Feed
experience, ecosystem, safety and business constraints.

## 8. Experiment program

Experiments may be orthogonal by layer when ownership and interference permit:
retrieval, coarse, fine, calibration/value, mixer, feature freshness and training
cadence. Orthogonality is an efficiency tool, not a claim that layers are
causally independent.

Every request logs eligible population, assigned layers, composed policy,
assignment probability and served checkpoints. Each experiment pre-registers:

- unit, trigger, traffic, duration, control and rollback;
- primary metric, guardrails, MDE and power;
- interference risks and incompatible concurrent experiments;
- exact stage change and expected ranking delta;
- sample/checkpoint feedback handling;
- pass, hold and reject thresholds.

The iteration cadence is:

```text
train or configure
→ offline frozen-candidate evaluation
→ shadow exact-score replay
→ historical/counterfactual replay where identified
→ A/A and SRM
→ ramped factual A/B
→ guardrail and power decision
→ active-policy update or rollback
→ post-launch sample/distribution review
```

The simulator must deliberately produce pass, hold and reject cases. A system in
which every new route or complex model wins is coupled to its evaluator.

## 9. Multi-business completion contracts

These start after the core Feed loop, logging and learning path are accepted.

| Surface | Minimum closed workflow | Independent business labels | Shared-system interaction |
|---|---|---|---|
| Search | query→retrieve→rank→click→post-search Feed/action | reformulation, click, dwell, success | search intent and post-search recommendation |
| Ads | auction eligibility→bid/pacing→mix→impression→click→conversion | revenue, advertiser value, user cost | shared load and Feed guardrails |
| Commerce | shelf/detail→cart/order/payment/refund | GMV, margin, purchase quality | product anchors and transaction value |
| Local | POI video/anchor→detail/map/YMAL→order or open-loop Pixel | anchor click, container value, transaction | geo/context, POI supply and Feed mix |
| Posting | draft→candidate recommendation→selection→publish | publish penetration, qualified supply, creator retention | exact post creation and future distribution |
| Live | room retrieval→enter→stay/engage/gift | watch, interaction, gift, creator health | time-sensitive supply and cross-surface load |
| Photo/card/article | format-specific retrieval and consumption | dwell, swipe depth, save/share | format calibration and diversity |

Each surface owns its sample space, candidate authority, labels, model and
Launch Review. Shared identity, content, experiment and event contracts prevent
duplicate implementations.

## 9.1 Audited work register

This is the exhaustive v4 backlog. A new task must either refine one row or be
added here with an owner, dependency and executable acceptance bar. A checkbox
in another document is informative only; it cannot override this register.

| ID | Authority / next implementation | Current state | Dependency | Acceptance evidence |
|---|---|---|---|---|
| P1-01 | Split retrieval into route registry, Feed lifecycle routes and business routes | Done | P0 | architecture lint and ownership tests pass |
| P1-02 | Immutable post identity and posting source lineage | Done | P1-01 | publish→future candidate→creator trace and failure tests pass |
| P1-03 | Lifecycle transition authority and indexes | Done | P1-01 | boundary, removal, ANN rebuild and deterministic replay tests pass |
| P1-04 | Lifecycle/post observability | Done | P1-02/03 | request-time lifecycle and route admission pass DuckDB/ClickHouse |
| P1-05 | Creator feedback, retention, deletion and moderation | Done for P1 mechanics | P1-02 | future supply, exit, delete and moderation tests pass; P2 owns calibration |
| P1-06 | P1 scale and replay review | Done | P1-01..05 | 100K/2M/two-tick 4090 report passes with content hashes |
| P2-01 | Correlated population generator | Done as a versioned six-component low-rank synthetic family authority | P1 | accepted with explicit no-TikTok-demographic-fidelity boundary |
| P2-02 | Arrival, timezone calendar, session and churn process | Done for synthetic mechanics and externally observed Feed sequence channels | P2-01 | unsupported universal retention remains masked |
| P2-03 | Exogenous/endogenous trend and concept drift | Done with regional-topic shocks, recovery and policy-independence tests | P2-01 | accepted as synthetic stress evidence only |
| P2-04 | Neural slate response SCM | Done; v23 passes external, sequence and held-out-family structural gates | P2-01/02 | frozen weights and content-bound reports agree |
| P2-05 | Latent transition, return survival and creator response | Done for observed Feed trajectory; retention and creator response deferred as masked tasks | P2-04 | no false KuaiRand labels |
| P2-06 | Ensemble uncertainty, support and causal noise | Done; 100K shadow support 98.9835%, attack rejection 100%, replay passes | P2-04/05 | 97%/99% frozen gates pass without threshold movement |
| P2-07 | External evidence adapters | Done; user-disjoint KuaiRand DR-OPE and manifests pass | P2-04 | overlap, ESS, max-weight and unsupported-task boundaries retained |
| P2-08 | DGP validity and authority shadow | Done for Feed; 100K/1M/8-tick shadow and full repository gate pass | P2-01..07 | Launch Review accepts explicit Feed-only manual promotion |
| P3-01 | `samples/recall`: source-aware negatives and correction contract | Done for sample authority; P3-06 must consume the stored expected count | P1/P2 | four sources retain q/log-q, expected count, observed status and false-negative mask; exhaustive-softmax and peer-frequency tests pass; 100K scale uses O(requests × draws) memory |
| P3-02 | `samples/coarse`: teacher/order/conflict authority | Reopened as P3-02a: current rows start after old coarse Top-K | P1/P2 | rematerialize every recall candidate with old-stage admission/rank, labels masked when unobserved, and teacher scores explicitly marked factual or shadow |
| P3-03 | `samples/fine`: PIT cascade/sequence authority | Reopened as P3-03a: exposed-only rows and deterministic probability cannot support challenger OPE | P1/P2 | retain full scorer input set, exposure/admission masks and exact propensities only for randomized actions; deterministic actions declare zero counterfactual support |
| P3-04 | `learning`: persistent sample bus, active/candidate lanes and registry | Done for infrastructure; LR probe is not a model launch | P3-01..03 plus P3-05 | independent cursors, numeric served checkpoint, compatibility rejection, fallback/restart and 100K/1M scale pass on accepted commit |
| P3-05 | `platform/features`: one feature/FID manifest for train and serve | Manifest accepted; reopened as P3-05a because trace retains only post-fine-selection tensors | P3-03 | persist byte-identical tensors for every coarse/fine scoring input, with stage, manifest and checkpoint lineage |
| P3-06 | `platform/retrieval` + `learning/retrieval`: migrate retrieval ladder | Done; lifecycle control retained, Graph/RRF/Two-Tower/Multi-interest rejected | P3-01/04/05 | persisted RecallExample, retrieval feature contract, registry/serving replay and 100K/1M equal-budget report pass; old-exposure target concentration is explicit; Semantic-ID still waits for an accepted dense baseline |
| P3-07 | `platform/ranking/coarse`: migrate equal-budget ladder and distillation | Historical implementations only | P3-02a/04/05a/09a | Rule/LR/XGBoost/W&D/DeepFM/DCNv2 share the recall universe and budget; RankDistil-style Top-K preservation, calibration, latency, memory and slices decide |
| P3-08 | `platform/ranking/fine`: migrate multi-task and sequence ladder | Historical implementations only; request-native listwise path absent | P3-03a/04/05a/09a | pointwise then request-grouped pairwise/listwise deep-cross/DIN/Transformer/MMoE/PLE on identical scorer inputs; per-head calibration, gradient/gate health and latency decide |
| P3-09 | `validation/evaluation`: support audit first, integrated evaluator and P3 review last | Metrics fragmented; deterministic trace has no challenger support | P3-06 plus P3-02a/03a/05a/07/08 | 09a freezes grouping, pass-through, exploration and support; 09b replays served ranks, GAUC/NDCG/PR-AUC/NE/ECE, slices, cost, identified OPE and factual paired A/B |
| P4-01 | `platform/ranking/value`: calibrate primitive heads, then compose Value Tree | Legacy value code only | P3 | per-task/slice probability or magnitude calibration is frozen before coefficient tuning; coefficients are versioned and nonnegative; sensitivity and monotonicity tests expose which primitive changed each rank |
| P4-02 | `experiments/metrics`: measure unified LT after randomization | Historical synthetic metric only; no production exchange authority claimed | P4-01 | pre-register stay/return/DAU/commercial outcomes, horizon, MDE and power; cohort curves distinguish immediate and learned effects; exchange assumptions are fitted on separate experiments and sensitivity-bounded, never used as labels |
| P4-03 | `platform/mixing`: one COPP/final-slate authority | Legacy mixers exist; v4 has no exposure/session dedup owner | P4-01 | eligibility→dedup→queue load→quota/diversity→final slate is deterministic; every displacement logs source queue, before/after rank, constraint and alternative; no queue can independently claim an exposure |
| P4-04 | `learning/cadence`: daily→hourly→streaming launch ladder | Legacy loop proves mechanics only | P3-04/09 | hold model/features/traffic constant; compare checkpoint age, sample/feature freshness, ranking delta, hot/new-user slices, reject/fallback rate and GPU/IO cost; no ranking/business delta means systems benchmark, not launch |
| P4-05 | `experiments`: compatibility and interference-aware layered program | Atomic factual assignment exists; shared-data/interference analysis incomplete | P3/P4 | pre-registered namespace/compatibility graph, eligibility, SRM/A-A, power and rollback; normal parameter tests share factual flow, while algorithm-learning tests select data-diverted, clustered or user-corpus co-diverted design and report symbiosis risk |
| P5-01 | `scenarios/search`: close query and post-search loop | Request/action skeleton only | P2-P4 | query/rewrite→lexical+semantic routes→rank/blend→click/dwell/reformulation/success→post-search Feed; own samples/models/LR; success is not reduced to CTR |
| P5-02 | `scenarios/ads`: add advertiser market and auction workflow | Catalog/action skeleton; no pacing/budget state | P2-P4 | eligibility→bid/auction/pacing/budget→retrieve/rank/mix→impression/click→mature conversion; deterministic spend and attribution; advertiser/revenue/user-cost metrics plus Feed guardrails and market-interference test |
| P5-03 | `scenarios/commerce`: add inventory and transaction state machine | Product lineage only | P2-P4 | shelf→detail→cart→order→payment→refund uses PIT inventory/price, deterministic order/payment IDs, conditional masks and delayed maturity; unpaid/refunded value remains distinguishable |
| P5-04 | `scenarios/local`: add POI world and closed/open attribution | POI identity/route skeleton only | P2-P4 | POI video/anchor→detail/map/YMAL→closed order plus separately observed Pixel path; distance/category/graph/diversity are distinct signals; identity loss, duplicates, orphan rate and attribution coverage are measured, not imputed negative |
| P5-05 | `scenarios/posting`: add candidate recommendation and supply feedback | Immutable Feed post/lifecycle accepted; posting ranker absent | P1-P4 | draft/media→POI/product/topic candidates→select→publish→qualified future supply; selection, publish and distribution are separate labels; user- and creator-randomized designs declare interference and creator retention horizon |
| P5-06 | `scenarios/live` and format adapters: own examination/labels | Surface enums and actions only | P2-P4 | Live availability→enter→stay/interact/gift plus delayed outcomes; photo/card/article own examination, dwell and calibration; only content representation/infrastructure is shared; each format has an independent LR |
| P6-01 | `validation/profiles`: freeze diagnostic/standard/stress manifests | 100K P1 and P2 shadows accepted; historical 1M/10M are not v4 evidence | P1-P5 | 100K/1M/10M profiles record users, sessions, corpus, routes, candidates/request, events, labels, checkpoints and business complexity; multi-tick partition/resume and content hashes are mandatory |
| P6-02 | `validation/performance`: profile before optimizing | P2 microbatch/RSS defects closed; remaining hot paths unprofiled | P1-P5 | batch-size throughput/latency/RSS/CUDA frontier and profiler trace identify bottlenecks; tensor/compile/Rust/C++ changes require numerical, lineage and semantic parity; no runtime rewrite by intuition |
| P6-03 | `validation/failures`: model-quality and recovery campaigns | P0 seeded diagnostics only | P3-P5 | inject index/checkpoint mismatch, bad snapshot, PS shard loss, feature delay, late labels, timeout and overload; registry state, per-head calibration/NE, fallback and business metrics detect impact and return to baseline after rollback |
| P6-04 | release authority: delete legacy path and publish reproducibly | Missing | all accepted successors | parity manifest proves every retained consumer uses v4; delete superseded `simulation/twin` and duplicate authorities; zero orphan modules, clean public/secret scan, fresh-clone README run and one clean content-bound release commit |

### 9.1.1 Research-to-execution contract for every remaining row

This table closes tool selection and prevents each remaining row from reopening as an
architecture discussion. Existing dependencies are preferred. A new service is
allowed only after the stated trigger; domain semantics remain repository-owned.

| ID | Reuse first | Required artifact | Reject or defer when |
|---|---|---|---|
| P3-02a/03a/05a | Existing trace/Joiner/manifest; change one authority, not learner-side reconstruction | full-universe candidate Parquet, stage masks, exact scoring tensors and propensity/support audit | any row silently maps unobserved to negative, deterministic choice to nonzero challenger support, or learner recomputes served features |
| P3-09a | Repository evaluator shell plus scipy/sklearn; exploration is an explicit policy lane | frozen request grouping, pass-through baseline, randomized-action manifest and support matrix | no positive support for the intended comparison, or factual and shadow scores are conflated |
| P3-07 | XGBoost GPU request-grouped `rank:ndcg` + existing PyTorch W&D/DeepFM/DCNv2 + top-K distillation | `coarse-leaderboard.json` and teacher Top-K pass-through report | teacher/value pass-through misses the gate, or effect does not pay for P99/memory; never use post-coarse rows as the universe |
| P3-08 | Existing PyTorch DIN/Transformer/MMoE/PLE; masked BCE then request-grouped pairwise/listwise | `fine-leaderboard.json`, per-head model card and gate/expert diagnostics | primary/guardrail Pareto, calibration, sequence ablation or latency fails; HSTU waits for measured long-sequence headroom |
| P3-09b | Same evaluator and candidate authority used by 09a; OBP only as differential oracle after compatibility | `p3-evaluation.json`, replay diff and Launch Review | exact action propensity/support absent, grouping differs, or served Top-K does not change |
| P4-01 | sklearn logistic/isotonic calibration plus repository-owned nonnegative Value Tree | calibration maps, coefficient manifest and sensitivity surface | primitive calibration/NE regresses, monotonicity breaks, or coefficients hide a permanent queue boost |
| P4-02 | scipy bootstrap/power primitives + existing CUPED implementation | pre-registration, MDE/power file, cohort curves and unified-LT sensitivity report | SRM/A-A fails, horizon is immature, or exchange weights were fitted on the evaluated experiment |
| P4-03 | Existing deterministic mixer; no public COPP package is an authority for business constraints | final-slate trace and displacement Parquet with dedup/load/quota/diversity reason codes | any queue double-claims exposure, alternative is missing, or final-slate replay is nondeterministic |
| P4-04 | Clean-frozen sample bus, checkpoint registry and replay; no new stream processor | daily/hourly/streaming cadence matrix with freshness, rank delta, cost and fallback | fresher checkpoints do not change supported rankings/business metrics; Flink waits for external cross-process recovery need |
| P4-05 | Existing atomic assignment + scipy inference; Google symbiosis designs define the decision tree | namespace compatibility graph, power/SRM report and interference design record | shared learned data invalidates SUTVA and no data-/cluster-/corpus-diverted design has adequate power |
| P5-01 | FAISS semantic retrieval; add a small BM25 library only after license/dependency smoke | search session dataset, lexical/semantic/blend leaderboard and success/reformulation report | query state or success definition is absent, or post-search behavior is leaked into request features |
| P5-02 | PyTorch/FAISS for models; repository-owned deterministic auction, pacing and budget ledger | advertiser-market replay, spend reconciliation and Ads/Feed guardrail A/B | spend does not reconcile, delayed attribution is immature, or auction treatment contaminates inventory without measurement |
| P5-03 | Existing event/Joiner contracts + repository-owned transaction state machine | inventory snapshots and shelf→refund lineage/reconciliation report | PIT inventory/price is missing, conditional labels collapse, or unpaid/refunded value is counted as durable |
| P5-04 | FAISS/graph routes; add H3 only when measured geo-index scale requires it | POI candidate/anchor/container dataset plus closed/open-loop attribution coverage report | nearest distance substitutes for intent, identity/orphan loss is hidden, or missing Pixel is imputed negative |
| P5-05 | Existing lifecycle/supply loop + PyTorch/FAISS candidate ranker | posting candidate dataset and select→publish→qualified-supply experiment review | selection, publication and later distribution labels are conflated, or creator/user interference is undeclared |
| P5-06 | Existing PyTorch content representations with format-specific heads | Live and photo/card/article examination contracts and separate leaderboards | video completion semantics are reused, availability is absent, or one pooled metric masks a format regression |
| P6-01 | Existing CLI, Arrow/Parquet manifests and RTX runner | immutable 100K/1M/10M profile manifests with row/event/candidate/label complexity | scale changes semantics, resume hash differs, or counts alone are presented as realism |
| P6-02 | PyTorch profiler then NVIDIA Nsight; `torch.compile` only on a measured tensor hot path | CPU/GPU/IO trace and throughput-latency-memory frontier with numerical parity | optimization has no measured bottleneck, changes lineage/numerics, or Rust/C++ merely duplicates Python authority |
| P6-03 | Existing failure fixture/registry; deterministic fault campaigns rather than a new chaos platform | detection, fallback, rollback and recovery-to-baseline matrix | HTTP success is the only health signal, rollback cannot reproduce prior scores, or head/business health stays shifted |
| P6-04 | Existing architecture linter, public scan, Git and fresh-clone gate | consumer parity manifest, deletion ledger and clean release evidence | any consumer still imports legacy authority, generated/private material leaks, or release evidence is not bound to one commit |

The research disposition for every row is settled. Remaining uncertainty is
empirical and must be resolved by its acceptance artifact, not another design
essay. “Settled” means the next falsifiable implementation is known; it does not
mean a model, coefficient or simulator has already passed. In particular:

- RecSim NG is a reference for decomposition and vectorization, not a dependency
  to transplant wholesale into the PyTorch runtime.
- Neural-SCM is an ensemble challenger trained from declared evidence; it does
  not receive hidden fields as serving features and is not tuned to make MMoE win.
- LLM simulation is bounded to semantic content, rare scenarios and adversarial
  review; it never supplies numeric A/B truth.
- OPE is allowed only where logged propensity and support exist. Otherwise the
  report must say unsupported rather than fabricate a counterfactual lift.
- A complex model may lose. The required output is an explained pass, hold or
  reject with equal data, candidate, tuning and serving budgets.

## 9.2 Dependency graph and next accepted slice

```text
P1 Feed supply-consumption closure
  → P2 calibrated hidden ecosystem
    → P3 request samples + continuous learned cascade
      → P4 value, mixer, experiments and cadence
        → P5 business surfaces
          → P6 scale, failures and legacy deletion
```

P1, the declared Feed scope of P2 and retrieval through P3-06 are accepted. The
next slice repairs ranking contracts before training; accepted manifest mechanics
remain, but their candidate coverage is not sufficient for a model claim:

```text
accepted sample bus + exact feature/FID bytes + checkpoint registry
→ accepted P3-06 factual retrieval candidates and rejection diagnostics
→ P3-09a: freeze request grouping, stage baselines and support requirements
→ P3-02a/03a/05a: full candidate universes, stage masks and scoring-input tensors
→ P3-07: recall-universe coarse ladder and Top-K distillation
→ P3-08: scorer-input multi-task and sequence fine ladder
→ P3-09b: diagnose pass, hold and reject with the same evaluator
→ P3 Launch Review
```

The P3-04/05 infrastructure probe may train LR only to prove end-to-end update
and serving parity. It is not a model launch. LR/XGBoost/deep model comparisons
belong exclusively to P3-06..09, after the same sample and feature bytes are frozen.

Accepted P1 command set:

```text
python -m fid_lab.check
python -m fid_lab.simulation.digital_twin.observability.cli \
  --output <content-addressed-p1-fixture> \
  --scenario feed_posting_cycle --ticks 2
```

The repository gate and CLI must run on the same clean source state. The CLI
must materialize a multi-tick fixture whose DuckDB and ClickHouse diagnostics
agree. Focused tests and architecture lint are diagnostics inside the repository
gate, not competing acceptance commands.

## 10. Ordered delivery phases

### P0 — Authority and observability

Status: completed on 2026-08-25.

- [Done] Materialize all seven v4 log tables to versioned Parquet.
- [Done] Add logical asset keys, content hashes, a deterministic CLI fixture,
  DuckDB request/stage/route/label queries and ClickHouse equivalents.
- [Done] Migrate raw route outputs atomically through `RetrievalResult`,
  `RequestCandidateTrace`, Joiner, materializer and fixture tests while keeping
  scores out of the user-world `RenderedSlateBatch` boundary.
- [Done] Add an explicitly non-training failure fixture with one recall miss,
  request-bound orphan and rejected/index-mismatched checkpoint; DuckDB and
  ClickHouse independently detect exactly one of each. The normal fixture also
  proves immature labels remain censored rather than zero-filled.
- [Done] Execute all diagnostic SQL against ClickHouse 25.8.32.4. This found and
  closed aggregate-alias shadowing, false inventory/bid orphans and exposure
  position base drift that the DuckDB-equivalent fixture did not expose.
- [Done] Remove v4 dependence on legacy table names and generic score fields;
  scoped source search and the v4/SQL suite confirm only typed v4 authorities.
- [Done] Verify the trace tensor/memory budget before keeping route-level raw outputs
  in the GPU hot path. The 100K standard run passes by streaming one analytical
  table at a time; future multi-tick runs must also partition by event time.
- [Done] Add event-time partitioning, cross-partition schema and content gates,
  atomic locked manifests, exact resume/conflict semantics, corruption checks
  and lazy Arrow replay. A partition key cannot claim a different event time.
- [Done] Preserve `python -m fid_lab.check` as the single environment-owned
  repository acceptance entrypoint; direct global `pytest` is not supported.
- [Done] Split the dense digital-twin test directory by sample and
  observability boundary. The remaining `asset-body-io` warning is explicit:
  this repository uses a declarative `AssetGraph`, not Dagster decorators, so
  there is no real wrapper to declare; adding a fake decorator would weaken the
  check rather than protect an asset body.

Acceptance: one command generates a deterministic fixture and all diagnostic
queries identify seeded failures. Full lineage exists from request to checkpoint.

### P1 — Feed posting and content lifecycle

Status: completed and committed on 2026-08-25.

- [Done] Introduce immutable post creation and replace random reserved-item
  publication. `source_candidate_id` no longer aliases `post_id`.
- [Done] Implement the observable 30-day recent corpus and separate
  cold-start/hot/evergreen/expired state authority.
- [Done] Replace the old fake six-route taxonomy with six core Feed authorities;
  Local, Posting, Commerce, Live, Search and retarget routes have distinct owners.
- [Done] Close factual creator feedback, retention mechanics, deletion,
  moderation and future supply; P2 owns empirical calibration.
- [Done] Add request-time lifecycle/post lineage to durable analytical rows,
  split route ownership and pass DuckDB, ClickHouse, architecture and 4090 gates.

Acceptance: one creator publish can be traced to later Feed candidates,
impressions, consumption, creator feedback and subsequent posting; lifecycle
transitions are deterministic under replay.

### P2 — User-world families and calibration

Status: accepted for the declared Feed response scope on 2026-08-25. Non-Feed,
retention and creator-supply heads remain masked and independently gated.

- Reuse the existing neural-SCM/ensemble and KuaiRand components through a
  versioned v4 adapter; do not fork their model logic into `digital_twin`.
- Generated reports and Launch Review bind the canonical manifest and weights.
- Preserve the shared canonical feature contract and make dataset coverage a
  first-class manifest. An external-only artifact cannot claim native v4 fields.
- Preserve real-cascade family 1/2 train, family 3 validation and family 9 test.
- Complete scope-aware retention/supply evidence or leave those heads masked;
  do not use KuaiRand to manufacture labels it does not observe.
- Leakage, support-distance and anti-exploitation remain mandatory regression
  gates for every later world artifact.

Acceptance: one content-bound artifact passes external distribution, free-running
sequence, supported policy ordering, held-out-family structural magnitude,
batch/order, leakage, uncertainty, boundary, support and exploitation gates. Its
v4 shadow preserves exact discrete lineage, <=1 ms duration drift and bounded
float deltas, with >=97% supported factual Feed requests. Unsupported heads remain
masked. Reports, weights, data manifests, repository gate and clean commit agree
before any authority switch.

### P3 — Streaming learning and learned cascade

Status: retrieval is accepted; ranking manifests/mechanics are accepted but
candidate coverage and support are reopened. No learned model is active.

- Run P3-09a → P3-02a/03a/05a → P3-07/08 → P3-09b.
- Reuse the dual-lane trainer and manifest; do not reconstruct serving inputs.

Acceptance: at least one learned model passes, one holds and one rejects for a
diagnosed reason; ranking delta, sample lineage and resource costs are auditable.

### P4 — Value, policy and cadence

Status: research complete; implementation pending on accepted P3 predictions.

- Calibrate primitive predictions; add Value Tree, COPP, dedup/diversity, queues
  and cross-business load ownership.
- Launch daily→hourly→streaming cadence independently.
- Measure short-term metrics, creator ecosystem and unified LT without training
  directly on synthetic LT.

Acceptance: model, policy and cadence lift are separately attributable, with
nonnegative exchange, guardrails, rollback and cost curves.

### P5 — Multi-business systems

Status: workflow contracts defined; implementations remain skeletons.

- Complete Search, Ads, Commerce, Local, Posting, Live and photo/card/article in
  that order of dependency and available evidence.

Acceptance: every surface meets its workflow contract, has a typed sample space,
independent experiment owner and documented final-mixer interaction.

### P6 — Scale, reliability and deletion

Status: P0 scale baseline exists; end-state work pending.

- Run 100K diagnostic, 1M standard and 10M high-complexity RTX 4090 protocols.
- Tensorize environment and ranking hot paths; measure throughput, memory,
  numerical/deterministic parity and semantic invariance across batch sizes.
- Test index/model mismatch, PS shard loss, feature delay, late labels, fallback,
  overload and recovery.
- Delete superseded `simulation/twin` execution paths and duplicate legacy
  authorities after parity.

Acceptance: repository gate, multi-family calibration, full-flow SQL fixtures,
GPU evidence, replay hashes, rollback and public-information scan pass on one
clean commit.

## 11. Research basis

The source-to-decision crosswalk lives in
[`docs/research/digital-twin-v4-research-basis.md`](../research/digital-twin-v4-research-basis.md).
It is evidence support, not a second backlog authority. No public source validates
this simulator; only the executable acceptance gates above can accept a claim.
