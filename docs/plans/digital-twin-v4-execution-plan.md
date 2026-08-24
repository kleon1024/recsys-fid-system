# Recommendation Digital Twin v4 — Execution Plan

Status: active execution authority

Updated: 2026-08-25 (P1 implementation and research audit)

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

The latest published P0 authority is commit
`48769dd49000da64fcd5d28e24e651f0c724adb5`. The accepted P1 implementation is
commit `e70334d`; its reports remain independently bound by content hashes.
The unified remote gate runs 202 historical tests plus 49 focused v4 tests; the
architecture lint has zero errors. One P0 CLI invocation
materialized a 32-request fixture with all seven Parquet tables, table hashes,
932 raw-route rows, 612 merged candidate decisions, 525 events, 1,632 task-label
rows, 625 training examples and one checkpoint. This proves table closure and
query execution at fixture scale. The scale and seeded-failure claims below are
separate P0 artifacts; they are not inferred from this small fixture.

The RTX 4090 P0 standard run now covers 100K users, 2M catalog items and
44,658,634 persisted rows. It completed in 74.34 seconds with 4.25 GiB peak
CUDA memory, 5.02 GiB process RSS and 124.67 MiB compressed Parquet. Building
and writing one Arrow table at a time prevented seven simultaneous CPU copies.
This is the accepted P0 scale strategy; 1M/10M business experiments must use
time partitions rather than one monolithic analytical snapshot.

| Capability | Current evidence | Status | Blocking gap |
|---|---|---|---|
| Hidden user world boundary | Platform cannot directly read hidden preference state | Implemented | Held-out family calibration remains incomplete |
| Atomic factual A/B world | One request receives one factual policy and commits once | Implemented | Longer-horizon interference tests remain incomplete |
| Delayed outcomes | Order/payment/refund/Pixel occurrence and ingestion time are distinct | Implemented | Production-like loss/duplicate/orphan distributions need calibration |
| Point-in-time projection | Delivered events, lifecycle transitions and removals replay across content-bound partitions | Implemented | P2 calibration remains |
| Request cascade trace | Raw routes, request-time lifecycle, post lineage and every cascade stage are retained | Implemented | P3 proposal propensity and learned artifacts remain |
| Feed retrieval mechanics | Six lifecycle-owned Feed routes plus six separately owned business routes | Implemented | Learned retrieval starts only after P2 |
| Layered experiments | Ownership, independent assignment and composed factual policy | Implemented | Learned artifacts and continuous trainers are not connected |
| Feed post creation | Immutable `post_id`, source lineage, capacity/cooldown/exit failure and future Feed trace | Implemented | Rich media processing belongs to P5 Posting |
| Content lifecycle | Observable 30-day recent, cold-start, hot, evergreen, expired, moderation and deletion | Implemented | Threshold calibration belongs to P2 |
| Public catalog anchors | Product/POI lineage is typed through projection and events | Implemented for P1 | Post media/semantic processing belongs to P5 Posting |
| Behavior realism | Hidden state, examination, multi-action response, drift and delayed outcomes exist | Partial | Current response is still a hand-authored one-step SCM, not a calibrated neural-SCM ensemble |
| Full-chain analytical store | Seven partitioned Parquet tables plus P1 lifecycle/post fields; DuckDB and ClickHouse agree | Implemented | P3 trainer consumption remains |
| Recall/coarse/fine sample authorities | Typed request-level examples and partitioned replay authority exist | Partial | Recall negatives, propensity correction and trainer consumption remain |
| Continuous learning | Historical demos exist outside v4 | Not connected | No active/candidate streaming lanes, checkpoints or snapshot gates |
| Model ladder | Historical synthetic ladders exist | Invalid for v4 launch | Must train on the same factual request dataset and serving budget |
| Search/Ads/Commerce/Live/Local | Surface actions and catalog types exist | Skeleton | Each lacks a closed business workflow and independent launch contract |

Consequently, current route Launch Reviews test mechanism plumbing only. They
cannot yet decide whether Graph, Two-Tower or Multi-interest should launch, and
no P1 result may be called an online-equivalent lift before the P1-P3 gates pass.

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
| `v4_candidate_decision_log` | request × item; merged rank, coarse/fine/rerank decisions, all scores, drop stage/reason, propensity |
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
- `CoarseRankExample`: real recalled candidates, route provenance/scores,
  teacher scores/order, hard labels, sampling probability and stage decision.
- `FineRankExample`: real exposure-space candidates, point-in-time dense/sparse
  features, short/long sequences, multi-label values, maturity masks, propensity
  and served scores.

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

| ID | Owner / deliverable | Current state | Dependency | Acceptance evidence |
|---|---|---|---|---|
| P1-01 | Split retrieval into route registry, Feed lifecycle routes and business routes | Done | P0 | architecture lint and ownership tests pass |
| P1-02 | Immutable post identity and posting source lineage | Done | P1-01 | publish→future candidate→creator trace and failure tests pass |
| P1-03 | Lifecycle transition authority and indexes | Done | P1-01 | boundary, removal, ANN rebuild and deterministic replay tests pass |
| P1-04 | Lifecycle/post observability | Done | P1-02/03 | request-time lifecycle and route admission pass DuckDB/ClickHouse |
| P1-05 | Creator feedback, retention, deletion and moderation | Done for P1 mechanics | P1-02 | future supply, exit, delete and moderation tests pass; P2 owns calibration |
| P1-06 | P1 scale and replay review | Done | P1-01..05 | 100K/2M/two-tick 4090 report passes with content hashes |
| P2-01 | Correlated population generator | Missing in v4; current factors are mostly counter-random marginals | P1 | fit/sample users, creators and content with copula/flow or latent mixture; held-out joint statistics |
| P2-02 | Arrival, timezone calendar, session and churn process | Partial | P2-01 | request/session/cross-session distributions and hazard calibration by cohort |
| P2-03 | Exogenous/endogenous trend and concept drift | Partial formula | P2-01 | region/topic/time shocks, recovery and policy-independent counterfactual seed tests |
| P2-04 | Neural slate response SCM | Missing; current authority is handwritten | P2-01/02 | set/attention slate encoder; censored watch-time and logically masked multi-action decoder |
| P2-05 | Latent transition, return survival and creator response | Missing learned authority | P2-04 | free-running multi-step calibration, retention and supply trajectories without teacher forcing |
| P2-06 | Ensemble uncertainty and causal noise | Counter RNG exists; ensemble missing | P2-04/05 | paired potential outcomes, world-member disagreement and support-distance report |
| P2-07 | External evidence adapters | KuaiRand component exists outside v4 | P2-04 | versioned adapters for randomized, observational and historical A/B summaries; license manifest |
| P2-08 | DGP validity suite | Research protocol exists, not connected | P2-01..07 | distribution, sequence, intervention, policy-order and anti-exploitation gates all executable |
| P3-01 | Recall Joiner and corrected negatives | Typed shell only | P1/P2 | in-batch/exposed/mined negatives, false-negative mask, proposal probability and correction tests |
| P3-02 | Coarse Joiner and teacher lineage | Partial typed example | P1/P2 | factual recall candidates, teacher logits/order, conflict samples and Top-K pass-through |
| P3-03 | Fine Joiner and cascade labels | Partial typed example | P1/P2 | point-in-time features, sequences, conditional masks, delayed maturity and propensity/DR fields |
| P3-04 | Streaming sample bus and checkpoint registry | Partition store exists; trainers missing | P3-01..03 | watermark/resume, active/candidate lanes, model-data-feature-index compatibility and fallback |
| P3-05 | Feature/FID authority | Legacy implementation exists outside v4 | P3-03/04 | user/item/creator/context/route/counter/sequence/content manifests; collision and parity reports |
| P3-06 | Retrieval model ladder | Only route-mechanism LR exists | P3-01/04 | fixed corpus/Top-K/latency Popular→Graph→Two-Tower→Multi-interest→generative reviews |
| P3-07 | Coarse model ladder | Missing on v4 samples | P3-02/04/05 | Rule/LR/XGBoost/W&D/DeepFM/DCNv2/distillation with equal candidate and tuning budgets |
| P3-08 | Fine model ladder | Missing on v4 samples | P3-03/04/05 | LR/XGBoost→deep crosses→DIN/Transformer→MMoE/PLE/HSTU with multi-task masks |
| P3-09 | Request-aware evaluation and diagnostics | Partial | P3-06..08 | request GAUC, NDCG, calibration, Top-K delta, slices, latency and paired A/B attribution |
| P4-01 | Primitive calibration and Value Tree | Missing in v4 | P3 | calibrated probabilities/magnitudes; versioned nonnegative exchange coefficients and sensitivity |
| P4-02 | Unified LT measurement container | Design only | P4-01 | LT is an A/B outcome container, never a training label; MDE/power and business argue sheet |
| P4-03 | COPP, dedup, diversity, quotas and multi-queue mixer | Missing | P4-01 | one final-slate owner; exposure dedup; load, displacement and constraint attribution |
| P4-04 | Cadence ladder | Missing | P3-04 | daily/hourly/streaming comparisons with fixed model/features/traffic and cost/freshness curves |
| P4-05 | Orthogonal experiment program | Assignment core exists | P3/P4 | compatibility graph, factual traffic allocation, SRM/A-A, feedback-aware analysis and rollback |
| P5-01 | Search closed loop | Skeleton | P2-P4 | query/reformulation/retrieval/ranking/click/success/post-search Feed and independent LR |
| P5-02 | Ads closed loop | Skeleton | P2-P4 | auction, pacing, budget interference, click/conversion delay, revenue and Feed guardrails |
| P5-03 | Commerce closed loop | Skeleton | P2-P4 | shelf→detail→cart→order→payment/refund, stock and transaction value |
| P5-04 | Local closed/open loops | Skeleton | P2-P4 | POI video/anchor→detail/map/YMAL→order plus Pixel identity/attribution/loss |
| P5-05 | Posting recommendation | Feed posting identity only | P1-P4 | draft/media→POI/product/topic candidates→select→publish; user- and creator-randomized LR |
| P5-06 | Live and format-specific loops | Skeleton | P2-P4 | room availability/enter/stay/gift and photo/card/article-specific examination/actions |
| P6-01 | Scale profiles | P0 100K/2M evidence only | P1-P5 | 100K diagnostic, 1M standard, 10M stress on 4090 with fixed scenario manifests |
| P6-02 | Tensor/GPU performance | Partial vectorization | P1-P5 | no per-request Python hot loop; numerical and semantic parity across batch sizes |
| P6-03 | Failure injection and recovery | P0 seeded diagnostics only | P3-P5 | index/checkpoint mismatch, PS shard, feature delay, late labels, timeout, overload and recovery LR |
| P6-04 | Legacy deletion and public release | Missing | all accepted successors | delete superseded `simulation/twin`, zero orphan authority, clean public scan and reproducible README |

The research disposition for every row is settled. Remaining uncertainty is
empirical and must be resolved by its acceptance artifact, not another design
essay. In particular:

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

P1-01 through P1-06 are accepted. The next slice is P2-01 through P2-08; P3
learned-model work must not begin until the hidden-world validity suite can
falsify a candidate simulator. P2 execution order is:

```text
correlated population generator
→ arrival/session/churn and trend process
→ neural slate response and censored multi-action decoder
→ latent transition, survival and creator response
→ ensemble causal noise and external evidence adapters
→ distribution/sequence/intervention/policy/anti-exploitation gates
→ P2 Launch Review
```

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

Status: research complete; implementation is the active phase.

- Implement arrival/calendar/lifecycle mixtures, sessions, churn, trends,
  nonlinear response and held-out mechanisms.
- Add public-data calibration adapters where licenses permit; synthetic-only
  tasks remain explicitly labelled.

Acceptance: calibrated marginal, conditional and sequence statistics pass by
country/timezone/activity/content/lifecycle slices; leakage probes fail closed.

### P3 — Streaming learning and learned cascade

Status: contracts partial; trainers and v4 model ladders pending.

- Build partitioned sample bus, three Joiners, active/candidate trainers,
  checkpoint registry and snapshot validation.
- Run retrieval, coarse and fine ladders in dependency order.
- Add FID collision/version and offline-online replay checks.

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

- [RecSim: configurable sequential user simulation](https://research.google/pubs/recsim-a-configurable-simulation-platform-for-recommender-systems/)
- [RecSim NG: probabilistic multi-agent ecosystem and accelerated execution](https://google-research.github.io/recsim_ng/)
- [RecFlow: An Industrial Full Flow Recommendation Dataset](https://arxiv.org/abs/2410.20868)
- [KuaiSim: A Comprehensive Simulator for Recommender Systems](https://arxiv.org/abs/2309.12645)
- [SARDINE: Dynamic and Interactive Recommendation Environments](https://arxiv.org/abs/2311.16586)
- [RecInter: interaction-centric dynamic recommender simulation](https://aclanthology.org/2025.emnlp-main.956/)
- [AAAI-25 LLM-Powered User Simulator](https://ojs.aaai.org/index.php/AAAI/article/view/33456)
- [Monolith: Real Time Recommendation System](https://arxiv.org/abs/2209.07663)
- [Scaling Instagram Explore Recommendations](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
- [How Instagram Suggests New Content](https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/)
- [Content-based Related Video Recommendations](https://research.google/pubs/content-based-related-video-recommendations/)
- [Co-optimize Content Generation and Consumption](https://research.google/pubs/co-optimize-content-generation-and-consumption-in-a-large-scale-video-recommendation-system/)
- [Modeling Recommender Ecosystems](https://research.google/pubs/modeling-recommender-ecosystems-research-challenges-at-the-intersection-of-mechanism-design-reinforcement-learning-and-generative-models/)
- [Provider-aware recommendation ecosystem simulation](https://research.google/pubs/towards-content-provider-aware-recommendation-systems-a-simulation-study-on-interplays-among-user-and-provider-utilities/)
- [Slate off-policy evaluation](https://arxiv.org/abs/1605.04812)

These sources support architecture and evaluation choices. RecInter and LLM
simulator results justify an optional semantic-agent lane, not replacement of
the reproducible vectorized numeric kernel. No source validates this simulator;
only the executable acceptance gates above can accept a repository claim.
