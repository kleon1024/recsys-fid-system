# End-to-End Recommendation System Design

## Acceptance outcome

The local system executes one versioned request through nine online stages and returns an auditable slate. On the deterministic 100-request acceptance smoke:

- full 20-item slate rate: 100%
- unsafe or inactive items returned: 0
- slates containing duplicate IDs: 0
- mean distinct categories per slate: 5.84 of 6

Latency is local in-process evidence only. It does not predict a networked production SLA.

## Request path

```text
RequestContext
    |
    +-- user embedding --------------------+
    |                                     |
    +-- user/country/device/hour --------+ |
                                         | |
Catalog -> Viking vector recall ---------|-+--+
        -> popular fallback -------------|----+--> RRF merge -> eligibility
        -> fresh/cold-start recall -------|----+                    |
                                                                      v
                                                        FID V2 online feature join
                                                                      |
                                                                      v
                                      coarse rank -> multi-task fine predictions
                                                                      |
                                                                      v
                                           value tree -> ranking rules -> COPP
                                                                      |
                                                                      v
                                                  organic/live/ad mixed ranking
```

Every returned candidate retains recall routes, FIDs, intermediate scores, predictions, and final score. Every request returns stage counts, timings, and artifact versions.

The feedback loop is executable as a separate authority-safe path:

```text
impression + delayed actions
  -> event-time Joiner
  -> mature multi-task examples
  -> online trainer
  -> versioned Parameter Server
  -> manifest + replay consistency audit
```

See [PRACTICAL_ENGINEERING.md](PRACTICAL_ENGINEERING.md) and run `python3 -m fid_lab.training.demo`.

## Stage contracts

### 1. Viking-compatible recall

`LocalVikingIndex` owns vector search. It normalizes item vectors at catalog construction and returns cosine-similarity top-K results with an index version. It is an exact NumPy index, not VikingDB itself. Its `recall(request, limit)` contract is the replacement boundary for a remote Viking client.

The public Viking documentation exposes personalized recommendation, popular and cold-start recall, recall merging, filtering, exposure deduplication, reranking, and diversity. The local system models those public capabilities without claiming internal implementation parity.

### 2. Multi-route recall merge

Vector, popular, and fresh routes run independently. Weighted reciprocal-rank fusion merges incomparable route scores:

```text
merged_score(item) = sum_route weight(route) / (K + rank_route(item))
```

Candidate identity is `item_id`; merging preserves all contributing routes and removes duplicates.

### 3. Eligibility and online features

Hard filters remove unsafe, inactive, country-ineligible, and previously exposed items before expensive scoring. `OnlineFeatureService` then uses the same `DEFAULT_SCHEMA` and `FidCodec(V2)` as offline experiments. The invariant is one slot registry, one hash contract, and one cross serialization online and offline.

### 4. Coarse rank

The local coarse scorer is deliberately lightweight. It combines vector similarity, category affinity, popularity, and merged-recall evidence, reducing 292 eligible candidates to 240. In production this boundary can load XGBoost or a distilled neural scorer.

The coarse pool remains large enough to preserve category coverage required downstream. An earlier 120-candidate truncation produced only one category and a seven-item final slate; the acceptance test prevents regression to that failure.

### 5. Fine rank and value tree

The fine stage predicts five independently interpretable leaves:

```text
p_click, p_like, p_long_view, quality, freshness
```

The value tree owns fusion:

```text
engagement = 0.45*p_click + 0.25*p_like + 0.30*p_long_view
ecosystem  = 0.75*quality + 0.25*freshness
value      = 0.72*engagement + 0.28*ecosystem
```

Weights live once in `config.py`; the independent test encodes the written formula. Public evidence defines “value tree” as a recommendation fusion formula, but the exact tree above is a local teaching design, not an internal company formula.

### 6. Ranking rules

The rule engine owns transparent score multipliers for freshness, high quality, and content type. Hard eligibility does not live here. This separates compliance filters from reversible business ranking adjustments.

### 7. COPP boundary

No credible public definition for the requested “COPP” recommendation component was found. The code therefore does not expand or redefine that acronym. `ConstrainedPolicyOptimizer` is the explicit local implementation behind an adapter named `copp`.

It maximizes adjusted value subject to:

- minimum fresh-item exploration
- maximum items per creator
- maximum items per category
- deterministic tie-breaking

If the real internal COPP contract becomes available, replace this implementation while preserving the adapter inputs, output, trace, and acceptance tests.

The similarly named published COPR method concerns consistency-oriented pre-ranking. It is relevant to coarse/fine rank alignment but is not treated as COPP.

### 8. Mixed ranking

The final mixer calibrates scores separately for organic, live, and ad content, then enforces type quotas and consecutive-category diversity. Optional pinning only moves an item already admitted by eligibility and policy; it cannot bypass safety filters.

## Version consistency

Each response declares:

```text
pipeline config
catalog
vector index
online feature service
FID layout
coarse model
fine model
COPP implementation
```

A production deployment should atomically bind these versions in one manifest. In particular:

- a user tower and Viking item index must be built from compatible model versions;
- a model must declare the FID schema and hash version it accepts;
- rollback must restore the model, index, value tree, rules, and mixer configuration together;
- shadow traffic should compare candidate IDs, FIDs, stage attrition, and final order before cutover.

## Production replacements

The local lab proves contracts and failure handling, not distributed scale. Production substitutions are intentionally narrow:

| Local component | Production replacement |
|---|---|
| NumPy exact vector index | VikingDB/Viking AI Search remote adapter |
| In-memory catalog | versioned item store and feature service |
| Analytic coarse scorer | XGBoost, distilled ranker, or ranking-consistent model |
| Analytic multi-task leaves | trained DeepFM/DCN/DIN/Transformer plus calibration |
| Greedy constrained optimizer | verified internal COPP implementation or slate optimizer |
| In-process traces | distributed tracing, stage metrics, and experiment logging |

The external API boundary should be added only with an authenticated Viking environment. Credentials, retry semantics, index creation, and live response schemas cannot be verified locally.

## Interview examination

Be ready to explain these concrete failures and decisions:

1. Why reciprocal-rank fusion is safer than adding raw scores from unrelated recall routes.
2. Why safety and exposure filtering happen before expensive ranking.
3. Why FID packing, hashing, and bucketization are three separate contracts.
4. Why a better fine-rank score can still damage the final slate if candidate coverage collapses.
5. Why AUC does not test value-tree calibration, creator caps, or mixed-ranking quality.
6. How to measure coarse/fine consistency using top-K recall, rank correlation, and NDCG.
7. How to migrate a vector index without serving user and item towers from incompatible versions.
8. Which rules are hard constraints, which are score adjustments, and which may degrade during fallback.
