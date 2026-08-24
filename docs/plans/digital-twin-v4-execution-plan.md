# Recommendation Digital Twin v4 — Execution Plan

Status: active execution authority

Updated: 2026-08-25

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
tree but has not passed the repository acceptance path.

Latest local verification on 2026-08-25: the raw-route contract migration is
complete and the focused v4/SQL suite has 42 passing tests. One CLI invocation
materialized a 32-request fixture with all seven Parquet tables, table hashes,
932 raw-route rows, 612 merged candidate decisions, 525 events, 1,632 task-label
rows, 625 training examples and one checkpoint. This proves table closure and
query execution at fixture scale; it does not yet prove GPU-scale logging cost,
all seeded failure classes.

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
| Point-in-time projection | Only delivered events update observable state | Implemented | Multi-partition replay parity remains missing |
| Request cascade trace | Raw route plus recall/coarse/fine/exposed masks and scores are retained | Implemented | Content lifecycle authority remains missing |
| Feed retrieval mechanics | HNSW, graph, geo, fresh, long-tail, popular, search, retarget | Implemented | Taxonomy does not match the target Feed lifecycle |
| Layered experiments | Ownership, independent assignment and composed factual policy | Implemented | Learned artifacts and continuous trainers are not connected |
| Feed post creation | Creator supply can publish reserved items | Partial | A posting action does not create the exact immutable post later consumed |
| Content lifecycle | Freshness and catalog timestamps exist | Partial | No 30-day recent authority, promotion, expiry or evergreen transition |
| Public catalog anchors | Deterministic product/POI links are drafted | Draft | Must pass tests and become request/event lineage |
| Behavior realism | Time, trend, freshness and drift additions are drafted | Draft | Must be calibrated by family and protected from feature leakage |
| Full-chain analytical store | Seven Arrow/Parquet tables, hashes, DuckDB case/stage queries and ClickHouse SQL | Partial | Seeded recall/version failures, CH server and scale cost remain |
| Recall/coarse/fine sample authorities | Typed examples and durable example index exist | Partial | Partitioned sample bus and trainer consumption remain |
| Continuous learning | Historical demos exist outside v4 | Not connected | No active/candidate streaming lanes, checkpoints or snapshot gates |
| Model ladder | Historical synthetic ladders exist | Invalid for v4 launch | Must train on the same factual request dataset and serving budget |
| Search/Ads/Commerce/Live/Local | Surface actions and catalog types exist | Skeleton | Each lacks a closed business workflow and independent launch contract |

Consequently, the current route Launch Reviews test mechanism plumbing only.
They cannot yet decide whether Graph, Two-Tower or Multi-interest should launch.

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

## 5. User and supply world TODOs

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

Status: pending; highest product priority.

- Introduce immutable post creation and replace random reserved-item publication.
- Implement the 30-day recent corpus and separate cold-start/hot/evergreen state
  machines and indexes.
- Replace current route taxonomy with the six core Feed authorities.
- Close creator feedback, retention and future supply.

Acceptance: one creator publish can be traced to later Feed candidates,
impressions, consumption, creator feedback and subsequent posting; lifecycle
transitions are deterministic under replay.

### P2 — User-world families and calibration

Status: pending.

- Implement arrival/calendar/lifecycle mixtures, sessions, churn, trends,
  nonlinear response and held-out mechanisms.
- Add public-data calibration adapters where licenses permit; synthetic-only
  tasks remain explicitly labelled.

Acceptance: calibrated marginal, conditional and sequence statistics pass by
country/timezone/activity/content/lifecycle slices; leakage probes fail closed.

### P3 — Streaming learning and learned cascade

Status: pending.

- Build partitioned sample bus, three Joiners, active/candidate trainers,
  checkpoint registry and snapshot validation.
- Run retrieval, coarse and fine ladders in dependency order.
- Add FID collision/version and offline-online replay checks.

Acceptance: at least one learned model passes, one holds and one rejects for a
diagnosed reason; ranking delta, sample lineage and resource costs are auditable.

### P4 — Value, policy and cadence

Status: pending.

- Calibrate primitive predictions; add Value Tree, COPP, dedup/diversity, queues
  and cross-business load ownership.
- Launch daily→hourly→streaming cadence independently.
- Measure short-term metrics, creator ecosystem and unified LT without training
  directly on synthetic LT.

Acceptance: model, policy and cadence lift are separately attributable, with
nonnegative exchange, guardrails, rollback and cost curves.

### P5 — Multi-business systems

Status: pending.

- Complete Search, Ads, Commerce, Local, Posting, Live and photo/card/article in
  that order of dependency and available evidence.

Acceptance: every surface meets its workflow contract, has a typed sample space,
independent experiment owner and documented final-mixer interaction.

### P6 — Scale, reliability and deletion

Status: pending.

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

- [RecFlow: An Industrial Full Flow Recommendation Dataset](https://arxiv.org/abs/2410.20868)
- [KuaiSim: A Comprehensive Simulator for Recommender Systems](https://arxiv.org/abs/2309.12645)
- [SARDINE: Dynamic and Interactive Recommendation Environments](https://arxiv.org/abs/2311.16586)
- [Monolith: Real Time Recommendation System](https://arxiv.org/abs/2209.07663)
- [Scaling Instagram Explore Recommendations](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
- [How Instagram Suggests New Content](https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/)
- [Content-based Related Video Recommendations](https://research.google/pubs/content-based-related-video-recommendations/)
- [Co-optimize Content Generation and Consumption](https://research.google/pubs/co-optimize-content-generation-and-consumption-in-a-large-scale-video-recommendation-system/)
- [Recommender Ecosystems: A Mechanism Design Perspective](https://ojs.aaai.org/index.php/AAAI/article/view/30266)

These sources support architecture and evaluation choices. They do not validate
the simulator until the executable acceptance gates above pass.
