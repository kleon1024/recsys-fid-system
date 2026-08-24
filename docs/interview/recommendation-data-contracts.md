# 推荐算法样本与特征面试口径 / Recommendation Data Contracts

面试回答统一按五个问题展开：目标、样本来源、偏差、离线验证、线上失败。代码是否存在
只是最低门槛；必须说明它消费哪个样本空间，以及失败后在哪一层定位。

## 召回样本 / Retrieval examples

目标是在固定 corpus 和 Top-K 下提高正样本召回，不用扩大候选预算伪造收益。

```text
query context
+ mature positive item
+ 60% in-batch positives from other requests
+ 25% same-request recalled hard negatives
+ 15% random-corpus negatives
+ source-conditional draw probability q
```

正样本来自成熟的 long-view、有效互动、POI detail 或交易行为。当前用户已经发生正反馈的
item 不进入其负样本池；hard negative 必须来自同一次请求实际召回但未选择的候选，不能
从其他请求拼一个“看起来相似”的 item。每条 negative 保存其来源内的单次抽取概率。
sampled-softmax 校正使用该来源的 expected count，即 `source_draw_count * q`。

离线在同一 item corpus、query 集、Top-K 和延迟预算下报告 Recall@K、long-tail recall、
coverage 和 ANN latency。线上常见失败是 false negative、热门 item 被反复当负样本、训练
负样本与线上候选不一致、embedding/index 版本错配，以及召回提升被粗排丢掉。

Interview English:

> I train the towers with mature positive actions and a stratified negative
> proposal. In-batch negatives come from other positive requests, hard
> negatives come from the same logged candidate set, and every draw retains its
> proposal probability for sampled-softmax correction.

## 粗排样本 / Coarse-rank examples

粗排的目标是在固定预算和延迟下保住精排真正有价值的候选。训练空间必须是实际召回日志，
不能使用全库随机 item 代替线上候选。

```text
request + recalled candidate
+ route and recall score/rank
+ point-in-time cheap features
+ mature hard labels and masks
+ teacher score/rank
+ served coarse/fine/value scores
+ sampling probability and version manifest
```

全部保留成熟正样本、teacher Top-K、粗精排冲突样本；普通负样本最多按 1:20 抽样并保存
q。蒸馏 loss 可以同时包含 hard-label、teacher probability 和 request 内 pairwise/listwise
顺序，但不能跨 request 比较 rank。门禁看 teacher Top-K pass-through、支付/转化正样本
通过率、NDCG、延迟和切片，而不是只看粗排 AUC。

线上失败包括 coarse budget 太小、teacher 分布漂移、不同 route 的分数不可比、普通负样本
占满训练集，以及粗排分数上线后一致性错误。

Interview English:

> Coarse-rank training examples must come from the logged retrieval pool. I keep
> all positives, teacher Top-K items and coarse–fine conflicts, downsample only
> ordinary negatives, and gate launch on teacher Top-K preservation and latency.

## 精排与级联标签 / Fine rank and funnel labels

精排只消费真实曝光及其 prediction-time 可用特征。每个任务独立判断成熟度；未曝光、不可
观测或未成熟的结果使用 `label_mask=false`，不能写成负样本。

```text
impression -> click/detail -> order -> payment
```

CTR 定义在有效曝光空间。独立 CVR 定义在已点击空间。ESMM 则在完整曝光空间联合训练
`pCTR` 与 `pCTCVR = pCTR * pCVR`，并强制 conversion implies click。取消、退款、Pixel
迟到、跨设备身份和 attribution window 必须另有明确合同。

pointwise BCE 用于概率头；pairwise/listwise 只能在同一个 request 的候选内构造。多目标
不能平均 AUC，必须定义 primary objective、guardrail、校准和 Pareto trade-off。线上最终
仍由用户级 A/B 判断，离线 AUC 只是 gate。

Interview English:

> Fine-rank examples are real impressions with point-in-time features and
> independently mature labels. For a click-to-conversion funnel, standalone CVR
> is click-conditional, while ESMM trains click and click-through conversion in
> the full impression space.

## 统一 request authority / Unified request authority

单条曝光无法回答候选在哪一层丢失。统一 authority 保存三张逻辑表：

```text
request
candidate decision at recall/coarse/fine/value/mix
mature candidate labels
```

主键分别是 request 和 `request + candidate + POI`。它必须能唯一归因：召回未找到、粗排
丢失、精排排错、混排挤掉或价值兑换错误。模型、feature schema、FID/hash、ANN index、
calibration 和实验参数都绑定同一 manifest。

## 特征工程与一致性 / Features and consistency

特征分为静态 sparse、内容语义、窗口 counter、实时状态、短序列、长序列和交叉特征。
所有特征满足 `feature_time <= prediction_time`。离线与在线使用同一 schema、默认值、
bucket、归一化和 FID 序列化；上线日志保存实际输入，离线逐字段 replay。

当前仓库的生产语义已覆盖 FID/hash/schema、窗口与序列、实时 feature campaign、版本
manifest 和 replay；它是有限规模 reference schema，不冒充真实公司的数百或数千特征。

## 当前实现边界 / Current implementation boundary

| Capability | Executable status |
|---|---|
| Three sample authorities | Implemented |
| Closed request-level candidate graph | Implemented |
| PIT join, maturity masks, Pixel attribution | Implemented |
| Correct 60/25/15 source semantics and per-draw q | Implemented |
| DCNv2 teacher distillation | Implemented, but the legacy scale benchmark is not yet request-graph-native |
| ESMM probability and loss invariant | Implemented |
| V4 randomized-exposure calibration and DR/OPE | Implemented; launch remains gated by policy-order and safety confidence |
| Request-native pairwise/listwise fine tuning | Planned in v4 P3-08/P3-09; no accepted implementation yet |
| Hundreds of real proprietary features | Out of scope; the repository uses a bounded public/synthetic schema |

因此标准答案不是“全有”。准确说法是：样本 authority 和闭环诊断已具备，召回 q 与 hard
negative 语义已经修正；粗排 benchmark 向 request graph 迁移、随机曝光 pairwise/listwise
训练及最终 A/B launch review 仍是下一轮工作。
