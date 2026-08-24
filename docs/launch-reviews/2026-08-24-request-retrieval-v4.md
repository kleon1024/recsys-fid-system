# Request-level Retrieval V4 Launch Review / 请求级召回 V4 上线复盘

## Decision / 结论

The shared Feed ANN route promotes the aligned Two-Tower model. Multi-interest is rejected. This is simulator authority only and is not evidence of a production TikTok launch.

共享 Feed ANN 路由上线 query contract 对齐后的 Two-Tower。Multi-interest 被拒绝。该结论只代表模拟器 authority，不代表 TikTok 生产上线。

The fixed serving budget is one 200,000-item corpus, a 24-item ANN pool per request, ANN Top-8, and 48 candidates after eight-route RRF merge. Every treatment changes only the ANN scorer.

固定服务预算为 20 万 item corpus、每请求 24 个 ANN pool、ANN Top-8、八路 RRF 合并后 48 个候选。每个 treatment 只替换 ANN scorer。

## Iteration history / 迭代过程

| LR | Change / 改动 | Offline / 离线 | Paired online result / 线上结果 | Decision / 决策 |
|---|---|---|---|---|
| R-LR-01 | POI-only 56,164-item corpus | Two-Tower Recall@20 0.286%, Multi-interest 0.452% | Anchor +0.074pp, but stay -0.0135s and LT uncertain | Reject |
| R-LR-02 | Full 200,000-item corpus | Two-Tower Recall@20 0.301% | Anchor and LT regress | Reject |
| R-LR-03 | Same corpus, but sequence-histogram query | Offline improves; serving query uses lifetime counters | Training-serving skew found | Reject |
| R-LR-04 | Category-affinity query contract aligned offline and online | Two-Tower Recall@20 0.258%, high-value Recall 0.696% | LT mean +0.00399, but old aggregate estimator cannot detect it | Measurement bug |
| R-LR-05 | Same model with true same-user paired estimator | Same frozen artifact | LT/user +0.00399, CI [0.00328, 0.00470]; stay +0.01066s, CI [0.00863, 0.01268] | Pass |
| R-LR-06 | Two-Tower to Multi-interest | Multi-interest high-value Recall 0.879%, but overall Recall 0.164% | LT/user -0.00113; stay -0.00352s | Reject |

## Why the first models failed / 为什么前几版失败

The POI-only model was trained on one corpus and served on another. It increased Local supply in the shared Feed ANN route but displaced content that produced Feed stay. The full-corpus model removed item OOD but still used different query semantics in training and serving. Both are consistency defects, not hyperparameter problems.

POI-only 模型训练和服务使用了不同 corpus。它在共享 Feed ANN 路由中增加了 Local 内容，却挤掉了贡献 Feed stay 的内容。Full-corpus 版本修复 item OOD 后，训练和服务仍使用不同的 query 语义。这两次失败都是一致性缺陷，不是超参数问题。

The former counterfactual function compared aggregate variances from two common-random worlds and ignored the covariance of the same users. V4 now optionally retains treatment-cell user outcomes in memory and estimates the variance of each user's treatment-minus-control difference. Raw user outcomes are not serialized into Git.

旧 counterfactual 函数比较两个 common-random world 的聚合方差，没有利用同一用户两种 potential outcome 的协方差。V4 现在可在内存中保留 treatment cell 的用户级 outcome，并直接估计每个用户 treatment-control 差值的方差。用户级原始 outcome 不写入 Git。

## Model and samples / 模型与样本

Training positives are mature long-view, quality-view, like, or Local anchor behaviors from valid impressions. Local detail, favorite, and conversion increase the IPS-corrected sample weight. Each query uses 60% positive-pool negatives, 25% same-request exposed hard negatives, and 15% random catalog negatives with subtract-log-q correction.

训练正样本来自有效曝光后的成熟 long-view、quality-view、like 或 Local anchor。Local detail、favorite 和 conversion 会提高经过 IPS 修正的样本权重。每个 query 使用 60% positive-pool negatives、25% 同请求 exposed hard negatives、15% 全库随机 negatives，并执行 subtract-log-q 修正。

The aligned query contains 12 observed category affinities, 12 Local category affinities, and eight point-in-time user/context features. The item tower consumes content topics, quality, freshness, POI and commerce fields, inventory, city, fulfillment, popularity, duration, and author bucket.

对齐后的 query 包含 12 维 observed category affinity、12 维 Local category affinity，以及 8 个 point-in-time 用户和上下文特征。Item tower 使用内容 topic、质量、新鲜度、POI 和 commerce、库存、城市、履约、热度、时长与 author bucket。

## Evidence boundary / 证据边界

The result proves that the simulator can distinguish corpus mismatch, query skew, estimator variance, and a genuine model regression. It does not prove external Local transaction lift because the Local response kernel remains synthetic and external Local validation is missing.

该结果证明模拟器能够区分 corpus mismatch、query skew、估计器方差问题和真实模型回归。由于 Local response kernel 仍是合成世界，并且缺少外部 Local 验证，它不能证明真实交易收益。
