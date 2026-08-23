# Architecture Visual Atlas

These diagrams define system boundaries and acceptance ownership. They are not claims that the local simulator already provides distributed production scale.

## 1. System context

```mermaid
flowchart TB
    Client["Feed, search, or recommendation client"] --> Gateway["Recommendation API"]
    Gateway --> Serving["Online recommendation serving"]
    Serving --> FeatureStore["Online feature store"]
    Serving --> VectorIndex["Viking-compatible vector index"]
    Serving --> ParameterServer["Online Parameter Server"]
    Serving --> Catalog["Item and policy catalog"]
    Serving --> EventLog["Impression and decision log"]
    Actions["User actions"] --> EventLog
    EventLog --> Joiner["Point-in-time Joiner"]
    Joiner --> Training["Offline and online training"]
    Training --> Registry["Artifact and model registry"]
    Registry --> ParameterServer
    Registry --> VectorIndex
    Registry --> Serving
    Observability["Metrics, traces, audits, experiments"] --- Serving
    Observability --- Training
```

## 2. Online candidate funnel

```mermaid
flowchart LR
    Pool["10M+ eligible items"] --> Vector["Vector recall"]
    Pool --> Graph["Graph or collaborative recall"]
    Pool --> Fresh["Fresh and cold-start recall"]
    Pool --> Popular["Popular fallback"]
    Vector --> Merge["Calibrated or rank-based merge"]
    Graph --> Merge
    Fresh --> Merge
    Popular --> Merge
    Merge --> Eligibility["Safety, region, exposure, liveness"]
    Eligibility --> PreRank["Coarse rank and coverage preservation"]
    PreRank --> Rank["Multi-task fine rank"]
    Rank --> Objective["Value tree and constrained objective"]
    Objective --> Rerank["Diversity and fatigue"]
    Rerank --> Mix["Cross-inventory mixing"]
    Mix --> Slate["Final top-K slate"]
```

## 3. Event-time training lineage

```mermaid
flowchart TB
    Impression["Impression at event time t"] --> Identity["request_id and item_id identity"]
    Action["Click, like, watch, conversion"] --> Identity
    Identity --> Window{"Label window closed?"}
    Window -- "No" --> Wait["Retain state; do not emit a negative"]
    Window -- "Yes" --> Dedupe["Deduplicate action event IDs"]
    Dedupe --> PIT["Point-in-time feature reconstruction"]
    PIT --> Example["Versioned multi-task example"]
    Example --> Split["Temporal train and validation split"]
    Split --> Train["Offline or micro-batch training"]
    Train --> Snapshot["Immutable model snapshot"]
```

## 4. Offline-online consistency isolation

```mermaid
flowchart LR
    Raw["Same logged raw event"] --> OfflineFeature["Offline transform"]
    Raw --> OnlineFeature["Online transform"]
    OfflineFeature --> FidCompare{"FIDs equal?"}
    OnlineFeature --> FidCompare
    FidCompare -- "No" --> FeatureIncident["Schema, hash, default, or cross incident"]
    FidCompare -- "Yes" --> OfflineScore["Offline inference"]
    FidCompare -- "Yes" --> OnlineScore["Online shadow inference"]
    OfflineScore --> ScoreCompare{"Scores within tolerance?"}
    OnlineScore --> ScoreCompare
    ScoreCompare -- "No" --> RuntimeIncident["Model, tensor, calibration, or runtime incident"]
    ScoreCompare -- "Yes" --> SlateCompare{"Candidates and slate equal?"}
    SlateCompare -- "No" --> PolicyIncident["Index, filter, policy, or fallback incident"]
    SlateCompare -- "Yes" --> Pass["Chain-consistency pass"]
```

## 5. Multi-objective decision path

```mermaid
flowchart LR
    Representation["Shared and task-specific representation"] --> Click["P click"]
    Representation --> Watch["Expected qualified watch"]
    Representation --> Like["P like or share"]
    Representation --> Negative["P hide or report"]
    Representation --> Retention["Long-term value proxy"]
    Click --> Calibrate["Per-head calibration"]
    Watch --> Calibrate
    Like --> Calibrate
    Negative --> Calibrate
    Retention --> Calibrate
    Calibrate --> Value["Context-aware value function"]
    Value --> Constraints["Safety, negative feedback, load, ecosystem constraints"]
    Constraints --> Slate["Constrained slate objective"]
```

## 6. Generative recommendation boundary

```mermaid
flowchart TB
    ItemData["Item text, media, and collaborative signals"] --> Encoder["Item encoder"]
    Encoder --> Quantizer["Residual quantizer or learned tokenizer"]
    Quantizer --> SemanticId["Versioned Semantic ID"]
    History["User behavior sequence"] --> Generator["Autoregressive recommender"]
    SemanticId --> Generator
    Generator --> Beam["Valid-prefix constrained beam search"]
    Beam --> Lookup["Live item lookup and deduplication"]
    Lookup --> Filter["Eligibility and policy filter"]
    Filter --> Rank["Discriminative rank or slate optimizer"]
    Codebook["Codebook and item mapping"] -. "atomic release" .-> Generator
    Codebook -. "atomic release" .-> Lookup
```

## 7. Outsourcing delivery gates

```mermaid
flowchart LR
    G0["G0: discovery and frozen requirements"] --> G1["G1: golden data and contracts"]
    G1 --> G2["G2: offline training and reproducibility"]
    G2 --> G3["G3: online shadow and replay parity"]
    G3 --> G4["G4: load, resilience, and security"]
    G4 --> G5["G5: controlled canary"]
    G5 --> G6["G6: handover and final acceptance"]
    G0 -. "owner approval" .-> G1
    G1 -. "independent evidence" .-> G2
    G2 -. "independent evidence" .-> G3
    G3 -. "independent evidence" .-> G4
    G4 -. "owner approval" .-> G5
    G5 -. "measured outcome" .-> G6
```

## 8. One launch protocol for every change

```mermaid
flowchart LR
    Change["Model, feature, strategy, realtime, product, or bug fix"] --> Train["Temporal train and validation"]
    Train --> Replay["Shadow and artifact replay"]
    Replay --> Cascade["Frozen recall, coarse, fine, value, and mixer"]
    Cascade --> AB["Powered A/B or interference-safe switchback"]
    AB --> Gate{"Primary, guardrail, parity, and cost pass?"}
    Gate -- "No" --> Reject["Reject, root cause, rollback"]
    Gate -- "Yes" --> Publish["Atomic manifest publication"]
    Reject --> Review["Launch Review"]
    Publish --> Review
    Review --> Next["Next hypothesis"]
```

## 9. Shared Feed loop and business-specific labels

```mermaid
flowchart TB
    Feed["Main Feed: play, stay, slide, quality view, negative"] --> POI["POI-anchored video distribution"]
    POI --> Container["POI detail, map, and YMAL"]
    Container --> Product["Product, order, payment, or Pixel"]
    Feed --> Posting["Posting page: shoot, select POI, publish"]
    Posting --> Supply["Qualified Local content supply"]
    Supply --> Feed
    Container --> Review["Review relevance and quality"]
    Feed --> FeedValue["Feed value"]
    Container --> Consumption["Local consumption Value Tree"]
    Product --> Transaction["Local transaction Value Tree"]
    Supply --> SupplyValue["Local supply Value Tree"]
    FeedValue --> LT["Accepted platform LT exchange"]
    Consumption -. "measured exchange only" .-> LT
    Transaction -. "measured exchange only" .-> LT
    SupplyValue -. "measured exchange only" .-> LT
```

## 10. Joiner and score diagnosis

```mermaid
flowchart LR
    Log["request + candidate + impression log"] --> Closure{"Identity closure and dedupe"}
    Events["play, stay, click, publish, order, Pixel"] --> Closure
    Closure --> Maturity{"Task label mature and observable?"}
    Maturity -- "No" --> Mask["label_mask = 0"]
    Maturity -- "Yes" --> PIT["Point-in-time feature join"]
    PIT --> Sample["Recall, coarse, or fine example"]
    Sample --> Version["FID, model, index, calibration, policy manifest"]
    Version --> Replay["Served versus replay score"]
    Replay --> Funnel["Route mix and stage pass-through"]
    Funnel --> Slice["Permission, city, category, head/tail, new item"]
    Slice --> Cause["Model, feature, sample, policy, or chain root cause"]
```

## 11. Semantic authority to tensor scale

```mermaid
flowchart LR
    Train["Temporal stateful training"] --> Publish["LR + XGBoost JSON + composite manifest"]
    Publish --> Hash["Artifact and 24-field schema hash gate"]
    Hash --> Semantic["Semantic Gymnasium contract oracle"]
    Hash --> Tensor["Torch state + XGBoost CUDA tensor world"]
    Semantic --> Dist["Control distribution parity"]
    Tensor --> Dist
    Semantic --> Effect["Treatment-effect parity"]
    Tensor --> Effect
    Dist --> Scale{"Parity pass?"}
    Effect --> Scale
    Scale -- "No" --> Repair["Repair DGP, state, or feature semantics"]
    Scale -- "Yes" --> Million["Million-user CRN A/B"]
    Million --> Gate["Feed, quality, negative, LT, and cost gate"]
```

## 12. Composite V4 world authority

```mermaid
flowchart LR
    Dataset["Request-level candidate dataset"] --> FeedKernel["External Feed kernel"]
    Dataset --> LocalKernel["Synthetic Local kernel"]
    Dataset --> SupplyKernel["Synthetic supply kernel"]
    FeedKernel --> Randomized["Randomized DR/OPE"]
    FeedKernel --> Shadows["Two independent stateful shadows"]
    Randomized --> WorldManifest["Simulator world manifest"]
    Shadows --> WorldManifest
    LocalKernel --> WorldManifest
    SupplyKernel --> WorldManifest
    PolicyManifest["Serving policy manifest"] --> Simulator["Tensor replay and A/B"]
    WorldManifest --> Simulator
    Simulator --> Stage["Recall, coarse, fine, mix attribution"]
    Stage --> Review["Launch Review and rollback"]
```

World-model authority and serving-policy authority are deliberately separate.
The external Feed kernel can validate Feed policy ordering without claiming POI,
supply, retention, commercialization, unified LT, or production deployment.
