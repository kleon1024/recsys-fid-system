# Practical Recommendation Engineering

This document connects the runnable local system to the engineering questions asked in Feed, search, ads, and recommendation interviews. Public systems are cited directly. Local designs and pattern-based interview additions are labeled as such.

## 1. Complete learning and serving loop

```text
online request
  -> recall -> rank -> mix -> impression log
                               |
                               v
user action stream -------> point-in-time Joiner
                               |
                               v
                     mature training examples
                               |
                               v
                    offline or micro-batch trainer
                               |
                               v
                  versioned Parameter Server (PS)
                               |
                               v
                manifest validation -> shadow -> publish
                               |
                               +----------> online request
```

Run the implementation:

```bash
python3 -m fid_lab.training.demo
```

The code demonstrates delayed-label joining, inverse-propensity clipping, three-task online learning, idempotent PS updates, stale-gradient rejection, offline/online AUC comparison, and chain consistency auditing.

## 2. Training examples and the Joiner

An impression is the causal anchor. A robust impression record contains:

```text
request_id, user_id, item_id, event_time, position, propensity
candidate source, model version, index version
raw feature references or materialized FIDs
schema/hash/cross/default versions
```

Actions arrive later and may be duplicated or out of order:

```text
event_id, request_id, item_id, action, action_time, received_at, value
```

The Joiner must enforce five invariants:

1. Join by an impression-scoped identity, not merely user and item IDs.
2. Accept an action only when `impression_time <= action_time <= label_deadline`.
3. Deduplicate by action event ID.
4. Do not emit a negative before the label window plus allowed lateness closes.
5. Reprocessing the same partition must produce byte-equivalent examples.

Premature negatives are a common production failure. A conversion arriving ten minutes after an impression is not a negative at minute two. Shortening the window makes freshness look better while corrupting the label distribution.

The propensity field records how likely the logging policy was to expose the item. Inverse-propensity weighting can partially correct selection bias:

```text
weight = min(1 / propensity, clip)
```

It does not solve unobserved confounding, bad propensity estimates, or support violations. Large weights also explode variance, which is why clipping and effective-sample-size monitoring matter.

## 3. PS online and model publication

“PS online” normally means a parameter-server architecture where sparse/dense parameters can be pulled for training or serving and updated without rebuilding one monolithic artifact for every micro-batch.

The local `VersionedParameterServer` enforces:

- immutable snapshots for readers;
- monotonically increasing model versions;
- idempotent `update_id` handling;
- bounded gradient staleness;
- shape and finite-value validation;
- explicit task order.

Industrial PS design additionally requires sharding, replication, hot-key handling, embedding admission/expiry, checkpointing, failover, optimizer-state ownership, and snapshot consistency. ByteDance’s public [Monolith paper](https://arxiv.org/abs/2209.07663) is the primary reference: it describes collisionless embedding tables, expirable embeddings, frequency filtering, fault-tolerant online training, and the trade-off between freshness and reliability.

Publishing must bind one manifest:

```text
model version
feature schema + FID/hash/cross versions
Joiner/label definition
task order and calibration
user tower + item tower + ANN index
value tree and ranking rules
training window and code revision
```

Never publish “the model” while allowing its feature configuration or item index to float independently.

## 4. Chain consistency

链路一致性 is broader than “offline and online code are similar.” The acceptance matrix is:

| Boundary | Required comparison | Typical failure |
|---|---|---|
| Event | schema, units, timezone, null semantics | milliseconds interpreted as seconds |
| Joiner | label window, lateness, attribution | premature negatives or future leakage |
| Feature | FID, hash, crosses, defaults | one language uses a different hash |
| Model | task order, tensor shape, calibration | click and like heads swapped |
| Retrieval | user tower, item tower, index version | N user tower queries N-1 index |
| Ranking | candidate set and score replay | offline evaluates a richer feature set |
| Policy | value weights, caps, fallback | experimental value tree only on one host |
| Logging | served version and propensity | impossible to reconstruct exposure policy |

Use three progressively stronger checks:

1. **Feature replay:** the same logged raw event must generate identical FIDs online and offline.
2. **Prediction shadow:** the candidate-level score delta must remain within a numerical tolerance.
3. **Slate replay:** candidate IDs, filtering decisions, calibrated values, and final order must be explainable stage by stage.

When replay differs, locate the first divergent stage. Comparing only final scores hides upstream candidate and feature loss.

## 5. Offline AUC and online AUC

AUC is the probability that a random positive receives a higher score than a random negative:

```text
AUC = P(score_positive > score_negative) + 0.5 * P(tie)
```

For Feed ranking, report more than global AUC:

- GAUC weighted over users or requests;
- AUC by country, device, user tenure, item age, creator size, and candidate source;
- top-weighted ranking metrics such as NDCG and Recall@K;
- calibration and expected calibration error per task;
- coverage and attrition at every funnel stage;
- counterfactual or propensity-weighted evaluation where assumptions hold.

“Online AUC” should mean AUC computed from examples actually served by a known model version after labels mature. It is not the A/B business metric. Online AUC can fall because traffic, candidates, features, labels, or serving behavior changed even if model weights did not.

Diagnose the gap in this order:

1. Replay identical mature examples through offline and online inference.
2. Verify model, feature, index, and calibration manifests.
3. Align label definitions, attribution windows, and observation maturity.
4. Compare candidate-source and slice distributions.
5. Check position and exposure-policy changes.
6. Check time leakage in the offline split.
7. Inspect latency fallbacks and missing-feature rates.

Research has repeatedly found that offline improvements have diminishing or unreliable correspondence with online outcomes because the recommender changes the data it later observes. See [Do Offline Metrics Predict Online Performance?](https://arxiv.org/abs/2011.07931) and Netflix’s [Page Simulator](https://netflixtechblog.com/page-simulator-fa02069fb269).

## 6. If AUC is almost unchanged, how can Feed algorithms drive growth?

Start from the product objective, not the model metric:

```text
North star: retained satisfied users / meaningful active days
Short-term diagnostics: qualified watch time, completion, saves, shares
Guardrails: hides, reports, quick skips, fatigue, latency, creator concentration
```

Global AUC can remain unchanged while each of these algorithmic levers creates growth:

| Lever | Why global AUC may miss it | Measure |
|---|---|---|
| Better recall | AUC is conditional on evaluated candidates | positive-item Recall@K, source coverage |
| Better top positions | AUC weights every pair equally | NDCG@K, top-slot utility |
| Calibration | monotonic score transforms preserve AUC | ECE, predicted/observed ratio |
| Multi-objective value | click discrimination may stay flat | watch, share, negative feedback, retention |
| Real-time interest | aggregate ranking changes only for fresh traffic | event-to-model lag, fresh-slice GAUC |
| Exploration | logged AUC favors the old policy | new-interest discovery, regret, IPS metrics |
| Cold-start supply | new items are a small AUC slice | time-to-first-qualified-exposure |
| Slate diversity | candidate-level AUC ignores interactions | intra-list diversity, session depth |
| Sequence modeling | gains cluster within sessions | next-N utility, session continuation |
| Reliability | offline AUC never sees fallbacks | full-model serve rate, p99 latency |

A strong 90-second interview answer:

> If AUC is flat, I first confirm we are comparing the same mature labels, candidates, and slices. Then I stop treating global AUC as the objective. For Feed growth I would map the funnel from recall coverage through top-K ranking, value fusion, slate construction, and feedback freshness. I would prioritize one measurable bottleneck—for example new-interest recall or quick-skip reduction—build an algorithmic intervention, verify guardrails offline, and run a powered A/B test on retained satisfied usage. The model change is successful only if it improves user value without shifting harm to negative feedback, latency, or creator concentration.

This is algorithm-driven because the intervention changes candidate discovery, value estimation, policy learning, or slate optimization. Merely changing button placement, notification copy, or autoplay UI is a product intervention.

## 7. Common Feed-team improvement mechanisms

### Retrieval and candidate coverage

- additional collaborative, graph, content, sequence, and fresh-item routes;
- multi-interest user representations instead of one averaged vector;
- target-aware or neural preranking earlier in the funnel;
- hard-negative mining and in-batch negative correction;
- compatible user-tower/item-index publication;
- quota allocation based on marginal recall contribution rather than route tradition.

Meta’s public Instagram Explore design uses retrieval, first-stage ranking, second-stage ranking, and final reranking, and describes multi-objective Two-Tower retrieval. X’s open-source systems expose candidate pipelines, filtering, ranking, and home mixing. Current reference: [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm). Historical 2023 reference: [twitter/the-algorithm](https://github.com/twitter/the-algorithm). Do not combine claims from the two snapshots without dates.

### Ranking and user value

- DIN/DIEN/Transformer/HSTU-style behavior sequence modeling;
- DCN/DeepFM/TokenMixer-style feature interaction;
- multi-task heads for click, watch, like, share, follow, hide, report, conversion;
- calibrated value fusion and user- or context-dependent weights;
- duration, position, selection, and delayed-feedback bias correction;
- teacher/student distillation for coarse/fine consistency.

Google’s [multitask video ranking paper](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/) explicitly addresses competing objectives and selection bias. Meta’s [News Feed ranking description](https://engineering.fb.com/2021/01/26/core-infra/news-feed-ranking/) describes predicting multiple actions and aggregating them into a single long-term-value score.

### Slate, ecosystem, and exploration

- category, creator, format, and topic diversity;
- fatigue and repeated-exposure control;
- creator and new-item exploration with bounded cost;
- constrained ad/live/organic mixing;
- long-term objectives such as return probability or satisfaction surveys;
- contextual bandits or policy learning when exploration and support are adequate.

Do not optimize diversity as a decorative reranker. It changes what is exposed, which changes future labels, creator incentives, and the reachable content graph.

## 8. Multi-objective learning

Separate four decisions:

1. **What to predict:** click, long view, like, share, follow, conversion, hide, report.
2. **How to share representation:** shared bottom, MMoE, PLE, task-specific experts, or separate models.
3. **How to train:** weighted losses, uncertainty weighting, gradient surgery, distillation, or constrained optimization.
4. **How to decide:** calibrated value tree, Pareto policy, or hard constraints.

Common failures:

- negative transfer when high-volume click gradients dominate rare conversion;
- seesaw effects where one task improves by hurting another;
- sample-space mismatch such as CVR observed only after click;
- uncalibrated heads combined as though probabilities were comparable;
- weights tuned against short-term metrics that damage retention;
- one global set of weights for contexts with different user intent.

Monitor per-head AUC, calibration, prevalence, gradient norms, expert load, slice metrics, and final business utility. A better task-head AUC does not prove the fused policy improved.

## 9. Lagrangian constraints

Suppose the Feed maximizes expected qualified watch value while constraining hide rate and ad load:

```text
maximize    E[watch_value(policy)]
subject to  E[hide(policy)] <= hide_budget
            E[ad_load(policy)] <= ad_budget
```

The Lagrangian is:

```text
L(policy, lambda_hide, lambda_ad)
= E[watch_value]
  - lambda_hide * (E[hide] - hide_budget)
  - lambda_ad   * (E[ad_load] - ad_budget)
```

Dual variables update in the direction of constraint violation:

```text
lambda <- max(0, lambda + learning_rate * (observed_cost - budget))
```

Interpretation: lambda is the learned shadow price of consuming one more unit of the constrained resource. If hide rate exceeds budget, its penalty rises. In production, dual updates need smoothing, caps, delayed-label handling, feasibility checks, and rollback protection.

ByteDance also publicly uses **Lagrange** as the name of a central AI-engineering console in a current [AML engineering role](https://joinbytedance.com/search/7447706774636939528). That platform name is distinct from the mathematical Lagrange multiplier.

## 10. Euclidean distance

For embeddings `u` and `v`:

```text
L2(u, v) = sqrt(sum_i (u_i - v_i)^2)
cosine(u, v) = (u dot v) / (||u|| ||v||)
```

If both vectors are L2-normalized:

```text
||u - v||^2 = 2 - 2 * cosine(u, v)
```

Therefore nearest neighbors by Euclidean distance and highest cosine similarity have identical ordering for normalized vectors. Without normalization they encode different notions: Euclidean distance includes magnitude, while cosine uses direction.

No reliable public evidence was found for a ByteDance recommendation platform called “Euclid.” Treat that internal-name hypothesis as unverified unless an authoritative internal contract is available.

## 11. Generative recommendation

Generative recommendation has at least four distinct forms:

1. **Generative retrieval:** generate an item ID or Semantic ID token sequence.
2. **Generative sequential modeling:** predict future actions/items from a behavior sequence.
3. **LLM semantic representation:** encode item content and user history with pretrained language models.
4. **Conversational or reasoning recommendation:** generate constrained choices and explanations.

TIGER represents items as discrete Semantic IDs and autoregressively predicts the next item; see [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065). Meta’s HSTU reframes recommendation as sequential transduction and demonstrates deployed scaling behavior; see [Generative Recommenders](https://arxiv.org/abs/2402.17152) and its [code](https://github.com/meta-recsys/generative-recommenders). ByteDance’s [HLLM](https://github.com/bytedance/HLLM) uses hierarchical item and user LLMs and publishes training/evaluation code.

A practical Semantic-ID chain is:

```text
item content/collaboration embedding
  -> residual quantization or learned tokenizer
  -> valid Semantic ID tuple
user history as Semantic ID sequence
  -> autoregressive model
  -> constrained beam search
  -> valid item lookup
  -> policy filtering and final ranking
```

Engineering questions that matter:

- How is the codebook versioned as items and semantics drift?
- Can every generated sequence resolve to a live, eligible item?
- How are duplicate beams and shared Semantic-ID prefixes handled?
- How are new items inserted without retraining the entire vocabulary?
- Does beam-search latency beat ANN retrieval at the required candidate count?
- How do multiple objectives control generation rather than only rerank output?
- What is the rollback unit: tokenizer, model, item mapping, and policy together?
- How do you compare generative recall with a strong ANN/two-tower baseline at equal latency and compute?

Generative recommendation does not remove filtering, policy, evaluation, or feedback loops. It changes candidate generation and representation; the rest of the production system still exists.

Run `python3 -m fid_lab.generative.demo` for a small residual-quantized Semantic-ID index and constrained prefix decoding. Its decoder uses an oracle prefix-similarity scorer so the demo verifies code construction and valid decoding; its overlap with exact vector recall is not a learned-model quality claim.

## 12. ByteDance reference map

Use only directly supported claims:

| Reference | Directly supported takeaway |
|---|---|
| [Monolith](https://arxiv.org/abs/2209.07663) | collisionless sparse embeddings and fault-tolerant online training |
| [Monolith source](https://github.com/bytedance/monolith) | public FeatureSlot/FID/training-serving implementation; repository is archived |
| [HLLM](https://github.com/bytedance/HLLM) | hierarchical item/user LLM recommendation with released code and weights |
| [TokenMixer-Large](https://arxiv.org/abs/2602.06563) | large industrial ranking model and scaling experiments |
| [Lagrange role](https://joinbytedance.com/search/7447706774636939528) | central console for AI engineering workflows |

FID V1/V2, Viking, value tree, and Lagrange are separately evidenced public concepts. Their coexistence does not prove that this local lab matches one internal ByteDance production stack.

## 13. Experiment discipline

For every Feed experiment, write before launch:

```text
hypothesis and causal mechanism
target population and power/MDE
primary metric and maturity window
guardrails and stopping conditions
expected funnel movement
logging and version requirements
novelty/interference risks
rollback owner
```

If AUC is flat and the hypothesis is candidate coverage, do not demand AUC movement. Demand better held-out positive recall at fixed budget and then validate satisfied usage online. Metrics must match the intervention’s causal location.
