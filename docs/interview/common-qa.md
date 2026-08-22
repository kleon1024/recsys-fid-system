# Common Recommendation Interview Questions

These are practice questions derived from common production-recommendation patterns. They are not claimed as confirmed questions from one company unless a direct source is linked.

## Data, samples, and Joiner

### 1. How do you construct a negative example?

Start from an impression, wait until the task-specific label and lateness windows close, then emit zero only if no qualifying action arrived. Never sample an unexposed item as an ordinary negative without defining the counterfactual assumption.

### 2. Why can random train/test splitting leak information?

It allows future item popularity, user behavior, embeddings, or aggregate features into training examples that precede them. Use temporal splits and point-in-time feature joins.

### 3. How do you join clicks and conversions?

Use an impression-scoped key, event-time bounds, attribution rules, action-ID deduplication, watermarks, and separate maturity windows. Log unmatched, late, duplicated, and ambiguous actions.

### 4. How do position and exposure bias enter training?

Labels are observed under the old ranking policy. Higher positions receive more observation independent of relevance. Log propensity and position, use randomized data where possible, and evaluate IPS/doubly robust methods with variance and support checks.

### 5. What is sample-ratio mismatch?

The observed split between experiment groups differs significantly from allocation. Treat it as an instrumentation or assignment failure and stop interpreting treatment effects until diagnosed.

## Features and consistency

### 6. FID, hashing, and embedding lookup: what is the difference?

FID packs field identity and signature. Hashing creates the signature or bucket. Embedding lookup maps the resulting key to trainable parameters. Version each contract independently.

### 7. How do you prove offline/online feature consistency?

Replay the same raw events through both transforms, compare FIDs and tensors, shadow online predictions, and retain the model/feature/index manifest with every impression.

### 8. What happens when a feature is missing online?

Use a versioned default defined by the schema owner, monitor missingness by slice, and train with the same default semantics. Silent zero filling is not safe unless zero is explicitly the default.

### 9. How do you migrate FID V1 to V2?

Dual-read or dual-write during a bounded window, publish compatible embedding namespaces, quantify collisions caused by signature truncation, shadow scores, and atomically switch model plus feature configuration.

### 10. Why should feature crosses have one owner?

Duplicate crosses can use different serialization, slots, windows, or defaults while appearing semantically identical. One registry prevents collisions and training-serving drift.

## Models

### 11. Wide&Deep versus DeepFM?

Wide&Deep memorizes explicit wide crosses and learns dense generalization. DeepFM adds FM second-order interactions and shares embeddings with the deep component, reducing manual cross engineering.

### 12. DeepFM versus FFM/DeepFFM?

DeepFM uses one embedding per feature. FFM uses a field-dependent embedding for each interaction partner, increasing expressiveness and parameter/memory cost.

### 13. DIN, DIEN, and Transformer sequence models?

DIN uses target-aware attention over behavior. DIEN explicitly models evolving interests. Transformers model richer long-range dependencies but require careful time encoding, sequence truncation, target conditioning, latency control, and leakage prevention.

### 14. MMoE versus PLE?

MMoE uses shared experts with task-specific gates. PLE separates shared and task-specific experts across extraction layers to reduce negative transfer. Neither guarantees that business-objective fusion is correct.

### 15. Why can increasing model size fail?

Data, labels, feature coverage, candidate coverage, optimizer stability, serving latency, or task interference may be the bottleneck. Establish a scaling curve across quality, compute, and latency rather than assuming parameter count is causal.

### 16. How do you distill a fine ranker into coarse rank?

Train coarse rank on teacher logits, pairwise order, or listwise targets while preserving hard labels. Measure top-K recall of fine-rank winners, rank correlation, NDCG, latency, and slice coverage.

## Metrics

### 17. Derive AUC.

AUC equals the correctly ordered positive-negative pairs plus half of tied pairs, divided by all positive-negative pairs. Efficient implementations rank scores rather than enumerate every pair.

### 18. AUC versus GAUC?

Global AUC permits cross-user pairs that never compete in one Feed. GAUC computes within-user or within-request AUC and weights groups, more closely matching personalized ordering while dropping single-class groups.

### 19. Why can AUC improve while CTR falls?

Candidate distribution changed, calibration deteriorated, top positions worsened, position bias leaked into training, latency triggered fallbacks, value fusion changed, or the A/B objective differs from click discrimination.

### 20. Why can two models have identical AUC but different log loss?

A monotonic transform preserves ordering and therefore AUC, while changing probability calibration and confidence, which log loss penalizes.

### 21. How do you compare offline and online AUC?

First replay identical mature examples. Then align population, candidate sources, labels, observation windows, feature/model versions, and slicing. Only after parity checks interpret distribution shift.

### 22. What should be monitored for a multi-task model?

Per-head AUC/GAUC, log loss, calibration, prevalence, missing labels, gradient norms, expert load, slice stability, fused-score distribution, and downstream business metrics.

## Online learning and PS

### 23. Synchronous versus asynchronous PS updates?

Synchronous updates improve consistency but wait for stragglers. Asynchronous updates improve throughput and freshness but introduce stale gradients. Bound staleness, make updates idempotent, and monitor update lag.

### 24. How do you handle unbounded embedding keys?

Admission thresholds, frequency filtering, expiry, dimension tiers, hot-key caches, collisionless tables where justified, and checkpoint/restore policies. Monitor memory by slot and key age.

### 25. How do you publish online-trained models safely?

Snapshot consistent parameters, attach the complete manifest, validate offline and shadow traffic, canary by traffic slice, and retain an atomic rollback for parameters plus features and index.

### 26. What if one PS shard is unavailable?

Choose an explicit policy: fail closed, bounded stale read, replicated read, or fallback embedding. Track affected slots and traffic; never silently mix arbitrary parameter versions.

## Feed growth and multi-objective decisions

### 27. AUC is flat. What do you do next?

Locate the causal bottleneck: recall, top-K ordering, calibration, freshness, exploration, slate diversity, multi-objective value, negative feedback, or reliability. Use a metric aligned with that stage and validate retained satisfied usage in A/B.

### 28. How do you prevent clickbait optimization?

Predict and penalize quick skips, hides, reports, and dissatisfaction; include qualified watch, saves, or surveys; constrain negative outcomes; inspect calibration and slices; optimize longer-term value.

### 29. How do you choose multi-objective weights?

Calibrate every head, define the business/user-value target, use historical and randomized evidence, inspect Pareto trade-offs, apply constraints for guardrails, and tune online. Raw action frequency is not a valid weight.

### 30. How do you explore new items?

Reserve bounded traffic or slate capacity using contextual uncertainty, creator/item priors, and eligibility constraints. Measure time to qualified exposure, downstream quality, regret, and ecosystem concentration.

### 31. How do you improve long-term retention rather than session time?

Use mature retention/satisfaction labels, sequence or policy models, shorter-term surrogate validation, holdouts, and guardrails. Account for delayed reward, confounding, and policy-induced distribution change.

### 32. When is an improvement a product lever rather than an algorithm lever?

UI layout, notification copy, and autoplay controls are product levers. Candidate sourcing, prediction, representation, value optimization, exploration, and slate policy are algorithm levers. Many successful experiments combine both, so attribution must name the changed mechanism.

## Retrieval, mixing, and system design

### 33. Why use multiple recall routes?

Different routes cover collaborative, semantic, fresh, graph, and popularity intents. Merge them based on marginal coverage and downstream value, deduplicate, and retain route attribution.

### 34. Why not add raw route scores?

They are not calibrated to one scale. Use calibrated probabilities, learned fusion, or rank-based fusion such as weighted RRF.

### 35. What is coarse/fine consistency?

The coarse stage should preserve candidates the fine stage would value. Measure recall of fine top-K, pairwise agreement, rank correlation, NDCG, and business-value loss at the coarse cutoff.

### 36. Reranking versus mixing?

Reranking adjusts one candidate set for slate effects such as diversity. Mixing combines heterogeneous inventories such as organic, ads, and live content with calibration, quotas, and constraints.

### 37. How do you version an ANN index?

Bind item-tower checkpoint, embedding transform, normalization, distance metric, item corpus snapshot, and index parameters. Serve it only with a compatible user tower.

## Generative recommendation

### 38. What is a Semantic ID?

A tuple of discrete tokens representing an item, often produced by residual quantization over semantic or collaborative embeddings. It allows an autoregressive model to generate item identifiers token by token.

### 39. Generative retrieval versus two-tower ANN?

Two-tower retrieval performs nearest-neighbor search in a continuous space. Generative retrieval decodes identifiers conditioned on history. Compare valid-item recall, cold start, diversity, latency, compute, index updates, and operational complexity at equal candidate budgets.

### 40. How do you guarantee valid generated items?

Use constrained decoding over an item-prefix trie, deduplicate beams, validate item eligibility and liveness, and version the tokenizer/codebook with the mapping table.

### 41. How do generative recommenders handle new items?

Semantic token sharing can generalize better than atomic IDs, but the item still needs a code, mapping, and eligibility state. Codebook drift and streaming insertion remain production problems.

### 42. Does generative recommendation replace ranking?

Not automatically. Generated candidates still require policy filtering, business constraints, calibration, safety, and often a discriminative ranker or slate optimizer.

## Math

### 43. Euclidean versus cosine distance?

For normalized vectors, squared Euclidean distance is `2 - 2*cosine`, so rankings are equivalent. Without normalization, Euclidean distance includes magnitude.

### 44. What is a Lagrange multiplier in Feed optimization?

It is the learned shadow price for violating a constraint. If observed hide rate exceeds its budget, the hide multiplier increases and penalizes policies causing hides more strongly.

### 45. Why must a dual variable be non-negative for `g(x) <= 0`?

The non-negative multiplier penalizes positive constraint violation while satisfying complementary slackness at the constrained optimum under regularity assumptions.

### 46. What can go wrong with online dual updates?

Delayed and noisy constraints, infeasible budgets, oscillation, distribution shift, and coupled constraints. Use smoothing, learning-rate control, caps, feasibility monitoring, and rollback.

## Project deep dive

### 47. Describe one recommendation incident.

State the user-visible symptom, first divergent stage, authoritative evidence, root cause, blast radius, mitigation, invariant added, regression test, and measurable recovery. Avoid presenting a suspected model issue when the real failure was features, index versioning, or fallbacks.

### 48. What did you personally own?

Name the decision and artifact you controlled, the trade-off you made, the code or experiment you delivered, and the acceptance evidence. “We improved recommendations” is not ownership.
