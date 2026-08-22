# Production Recommendation System Reference

[![Reference acceptance](https://github.com/kleon1024/recsys-fid-system/actions/workflows/ci.yml/badge.svg)](https://github.com/kleon1024/recsys-fid-system/actions/workflows/ci.yml)

An executable reference architecture and public outsourcing RFP for an industrial Feed, search, and recommendation platform.

## Main Feed first

The current acceptance boundary is the short-video main Feed. Local Service
posting, POI distribution, detail, product, and review models remain downstream
experiments and cannot make the main-Feed gate pass.

```mermaid
flowchart LR
    U["User state: interest, satisfaction, fatigue"] --> Q["Feed request"]
    Q --> R["Six-route recall"]
    R --> C["Coarse Top 20"]
    C --> F["LR / W&D / DeepFM / DCNv2 / MMoE"]
    F --> V["Calibration and Value Tree"]
    V --> E["Exposure"]
    E --> A["Play, 3s, slide, stay, LT, HLT, interactions, negative"]
    A --> S["State transition, leave, return"]
    A --> J["Point-in-time Joiner and delayed labels"]
    J --> T["Retrain, shadow replay, user-level A/B"]
    T --> F
```

There are three deliberately separate execution tiers:

- a debuggable stateful trajectory engine for request, session, and retention semantics;
- an RTX 4090 tensor engine for million-user sequential simulation;
- a vectorized experiment engine for orthogonal layers, CUPED, and 0.1%-1% effects.

The latest synthetic [main-Feed Launch Review](docs/launch-reviews/2026-08-23-main-feed-foundation.md)
records both wins and rejected launches. It does not claim company-internal
metrics. The [simulation evidence review](docs/research/main-feed-simulation-evidence.md)
defines the public evidence boundary and why Rust is conditional on profiling.

The runnable [POI posting recommendation reconstruction](docs/architecture/poi-posting.md) adds multimodal draft fusion, permission-aware geographic features, impression-derived labels, hard-negative sampling, entire-space sparse publication, and a multi-task ranker.

The [production model suite](docs/architecture/model-suite.md) extends that supply-side model into POI-anchored Feed distribution, map/detail, YMAL, product, and review recommendation with separate model families, streaming samples, long sequences, cascade audits, and full-path consistency.

The [model evolution laboratory](docs/architecture/model-evolution.md) compares
mature open-source LR, XGBoost, WDL, DeepFM, DCN-Mix, DIN, MMoE, and PLE
implementations on one synthetic distribution. It also adds trained Semantic-ID
generation, closed/open-loop attribution, and stateful request/session A/B simulation.
Use the [failure runbook](docs/operations/failure-runbook.md) and
[senior project deep dive](docs/interview/project-deep-dive.md) for production
diagnosis and interview practice.

Public procurement package:

- [REQUEST_FOR_PROPOSAL.md](REQUEST_FOR_PROPOSAL.md): scope, delivery gates, acceptance criteria, security, commercial response, and vendor evaluation.
- [Bidder response template](docs/procurement/bidder-response-template.md): mandatory response format.
- [Architecture visual atlas](docs/architecture/visual-atlas.md): visual system atlas for technical and delivery review.

## Procurement status

| Item | Status |
|---|---|
| RFP | Public and open until an award or closure notice is posted |
| Delivery | Remote-first, milestone-gated outsourcing engagement |
| Scale response | Mandatory pricing for 100, 1,000, and 10,000 RPS tiers |
| Technical questions | Public `rfp-question` GitHub issue |
| Capability statement | Public `rfp-capability` GitHub issue |
| Commercial response | Private channel after capability review |
| Source license | Evaluation only until an explicit `LICENSE` or contract grant is added |

## Target architecture

```mermaid
flowchart LR
    Request["Feed request"] --> Recall["Multi-route recall"]
    Catalog["Item catalog and Viking-compatible index"] --> Recall
    Recall --> Filter["Eligibility and exposure filter"]
    Filter --> Features["Online FID feature join"]
    Features --> Coarse["Coarse rank"]
    Coarse --> Fine["Multi-task fine rank"]
    Fine --> Value["Value tree and ranking rules"]
    Value --> Policy["Constrained policy optimization"]
    Policy --> Mix["Organic, live, and ad mixing"]
    Mix --> Response["Auditable slate"]
    Response --> Impression["Versioned impression log"]
```

## Learning and publication loop

```mermaid
sequenceDiagram
    participant S as Serving pipeline
    participant L as Event log
    participant J as Point-in-time Joiner
    participant T as Trainer
    participant P as Parameter Server
    participant A as Consistency audit

    S->>L: Impression, FIDs, propensity, artifact versions
    L->>J: Impression and delayed action streams
    J->>J: Wait for label window and allowed lateness
    J->>T: Mature multi-task training examples
    T->>P: Idempotent bounded-staleness update
    P->>A: Immutable parameter snapshot and manifest
    A->>S: Shadow-approved version
```

## One atomic publication manifest

```mermaid
flowchart TB
    Manifest["Release manifest"] --> Schema["Feature schema, FID, hash, crosses"]
    Manifest --> Joiner["Joiner and label definitions"]
    Manifest --> Model["Model weights, task order, calibration"]
    Manifest --> Index["User tower, item tower, ANN index"]
    Manifest --> Policy["Value tree, constraints, mixer"]
    Manifest --> Runtime["Runtime, fallback, observability"]
    Schema --> Gate["Replay and shadow gate"]
    Joiner --> Gate
    Model --> Gate
    Index --> Gate
    Policy --> Gate
    Runtime --> Gate
```

The repository includes a local end-to-end reference implementation of the intended sparse-feature and ranking boundaries:

```text
raw event
  -> feature registry (slot ownership + declared crosses)
  -> stable signature
  -> packed FID V1/V2
  -> one shared encoded batch
  -> XGBoost / Wide&Deep / DeepFM / user-item-context three-tower
  -> AUC + log loss
```

The online system continues from that model lab:

```text
catalog (1,200 items)
  -> Viking vector + popular + fresh recall (400 route hits)
  -> weighted recall merge + deduplication (300)
  -> eligibility and exposure filtering
  -> online FID feature join
  -> coarse rank (240)
  -> fine multi-objective prediction + value tree (240)
  -> boost/bury ranking rules
  -> COPP adapter / constrained policy selection (40)
  -> calibrated organic/live/ad mixed ranking (20)
```

See the [system design](docs/architecture/system-design.md) for contracts and evidence boundaries.

Production engineering and interview references:

- [Practical engineering](docs/interview/practical-engineering.md): Joiner, training examples, online PS, consistency, offline/online AUC, Feed growth, multi-objective learning, public industry references, Euclidean distance, Lagrangian constraints, and generative recommendation.
- [Common interview Q&A](docs/interview/common-qa.md): compact production and fundamentals questions with answer boundaries.

This lab reproduces the **public FID bit contract**, not a proprietary internal framework. The name "SEO" and its predefined feature-combination functions could not be identified from public evidence, so their semantics are not guessed here.

## What FID means

The public ByteDance Monolith implementation stores a slot and signature in one unsigned 64-bit value:

| Layout | Slot | Signature | Reserved |
|---|---:|---:|---:|
| V1 | 10 bits | 54 bits | 0 bits |
| V2 | 15 bits | 48 bits | 1 bit |

```text
FID V1 = (slot << 54) | (signature & (2^54 - 1))
FID V2 = (slot << 48) | (signature & (2^48 - 1))
```

The slot identifies the feature field. The signature identifies a value inside that field. A signature can come from a raw numeric ID or a stable hash; FID packing itself does not define the hash function.

V1 to V2 conversion preserves the slot but truncates the upper six signature bits. It is therefore not generally reversible. The implementation and tests are in `fid_lab/fid.py` and `tests/test_fid_lab.py`.

## Durable boundaries

- `fid_lab/fid.py` is the only owner of bit packing and stable signatures.
- `fid_lab/schema.py` is the only owner of slot assignment, feature groups, bucket sizes, and crosses.
- `fid_lab/data.py` creates and encodes every example once.
- `fid_lab/models.py` contains models only; models cannot reinterpret raw features.
- `fid_lab/experiment.py` owns the fixed split, training loop, and comparable metrics.

This preserves the main production invariant: offline training and online inference must use the same feature definitions, hash version, slot registry, cross semantics, and model artifact.

## Model evolution

| Stage | What it learns | Main limitation |
|---|---|---|
| XGBoost | Nonlinear rules over one-hot FID buckets | Sparse IDs generalize poorly; no learned embeddings |
| Wide&Deep | Memorized linear terms plus higher-order DNN patterns | Important crosses may still need explicit declaration |
| DeepFM | First-order terms, all second-order embedding interactions, and DNN interactions | Every pair receives the same FM interaction form; sequence intent is absent |
| Three-tower | Separate user, item, and context representations plus a ranking head | This lab's architecture is explicitly defined, not claimed as a universal named model |

`DeepFM` and `DeepFFM` are not synonyms. DeepFM shares one embedding per feature across pairwise interactions. FFM-style models use field-dependent embeddings, which cost substantially more parameters. Add DeepFFM only when field-aware interactions are a measured requirement, rather than treating it as the default successor.

The production suite now adds separate retrieval, coarse-rank, fine-rank,
sequence, multi-task, Joiner, and experiment boundaries. The original compact
models remain as teaching baselines; the evolution benchmark uses
DeepCTR-Torch rather than maintaining another local copy of its model zoo.

## Run

```bash
git clone https://github.com/kleon1024/recsys-fid-system.git
cd recsys-fid-system
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m fid_lab.experiment
python3 -m fid_lab.online.demo
python3 -m fid_lab.online.benchmark
python3 -m fid_lab.training.demo
python3 -m fid_lab.generative.demo
python3 -m fid_lab.evolution.evaluation.benchmark --profile ci
python3 -m fid_lab.evolution.cli.generative_demo
python3 -m fid_lab.evolution.cli.ab_demo
python3 -m fid_lab.simulation.cli --users 2000 --items 4000
python3 -m fid_lab.feed_loop.cli --users 3000 --items 4000 --ab-users 500 --epochs 10 --device cuda:0
python3 -m fid_lab.feed_loop.tensor_cli --users 1000000 --steps 24 --device cuda:0
python3 -m fid_lab.feed_loop.power_cli --users 1000000
python3 -m fid_lab.check
```

Run one stage while studying it:

```bash
python3 -m fid_lab.experiment --models xgboost
python3 -m fid_lab.experiment --models wide_deep deepfm
python3 -m fid_lab.experiment --models three_tower
```

The generated data contains user and item latent effects plus country-category, category-device, and age-hour interactions. Model metrics are deterministic for the same seed, but they are reference evidence, not a production benchmark claim.

## Sources and evidence boundary

- [ByteDance Monolith](https://github.com/bytedance/monolith): public FID V1/V2 layout, `FeatureSlot`, and `FeatureColumn` implementation. The repository was archived in 2025, so it is useful as public design evidence rather than a current dependency.
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792): memorization plus generalization architecture.
- [DeepFM](https://arxiv.org/abs/1703.04247): shared embedding input for FM and deep components.
- [FAT-DeepFFM](https://arxiv.org/abs/1905.06336): evidence that DeepFFM is a field-aware branch, not another name for DeepFM.
- [Viking AI Search documentation](https://docs.byteplus.com/en/docs/viking-aisearch/Recommended_Input): public product boundary for recommendation input, recall strategies, merging, filtering, deduplication, reranking, and diversity.
- [SARDINE](https://github.com/naver/sardine): packaged Gymnasium runtime for dynamic recommendation environments.
- [KuaiSim](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c7f8f98f9a8f5650922dd4545254f28-Abstract.html): request-level, session-level, and cross-session simulator evaluation protocol.
