# Unified LT and Local Service simulation

## 中英文结论 / Bilingual conclusion

| 中文 | English |
|---|---|
| 必须先修底层模拟器，再扩大模型迭代。错误的价值口径、行为世界或实验估计会让更大的模型更快地产生错误结论。 | The simulator foundation must be repaired before expanding model iteration. Wrong value semantics, behavior worlds, or estimators only let larger models produce wrong conclusions faster. |
| 第一优先级是统一 LT、Local 闭环与开环行为、候选与指标闭包、已知真值和 A/B 统计。 | The first priority is unified LT, closed/open-loop Local behavior, candidate/metric closure, known truth, and A/B inference. |
| 第二优先级是稳定内容生命周期、搜索后推、重定向、实时库存、投稿供给干扰和多队列机会成本。 | The second priority is stable content lifecycles, post-search recommendation, retargeting, realtime inventory, posting-supply interference, and multi-queue opportunity cost. |
| 第三优先级才是把 Two-Tower、DCNv2、DIN、MMoE、PLE 和超长序列逐一接入同一 Launch Review。 | Only the third priority is bringing Two-Tower, DCNv2, DIN, MMoE, PLE, and long-sequence models into the same Launch Review. |
| 增加用户只能降低方差，不能修复缺失的因果机会或错误的价值公式。 | More users reduce variance; they cannot repair missing causal opportunities or an incorrect value formula. |

## Contract boundary

`LT` is the versioned platform-metric exchange container. It does not mean long
view, and the business Value Trees below are not its additive terms. Reports use
`long_view` and `quality_long_view` for video-consumption labels and reserve
`lt_value` for the agreed platform metric.

The exchange contract lives in `fid_lab.value.contracts`. Its checked-in rates
are explicit synthetic DGP truth with uncertainty metadata. They demonstrate
mechanics and cannot be presented as ByteDance or TikTok production economics.
No reliable public source defining an internal ByteDance LT container was found.

For an experiment-level incremental outcome vector, the container computes:

```text
LT = a * incremental stay
   + b * incremental active days / DAU proxy
   + c * platform-accepted incremental commercialization metric
```

Feed, Local consumption, Local transaction, Local supply, Ads, and Live each
retain their own Value Tree formulas for ranking. POI VV, anchor click, detail,
posting penetration, and their tree scores are diagnostic or business metrics;
they cannot be inserted into LT. A Local launch enters LT only through its
experimentally measured effect on stay, active days, or an accepted
commercialization measure.

Production exchange rates must come from randomized dose tests, switchbacks,
long-horizon holdouts, and finance-approved contribution margins. Business
teams own outcome definitions and constraints; the central value authority owns
the unit, estimation protocol, version, uncertainty, and rollback. A business
cannot inject an unversioned score multiplier at the mixer.

The default simulator does not accept Local transaction margin into LT. It is
reported as `local_commercialization_value` beside the Local Value Tree. Only
an explicit platform-level exchange contract may move that metric into LT.

默认模拟器不接受 Local 交易利润直接进入 LT。它作为
`local_commercialization_value` 与 Local Value Tree 分开报告；只有平台级明确认可的
兑换合同才能将其纳入 LT。

| Object / 对象 | Purpose / 作用 | May enter LT? / 可进入 LT？ |
|---|---|---|
| Feed / Local / Ads / Live Value Tree | Ranking, allocation, and business diagnosis / 排序、流量分配、业务诊断 | No / 否 |
| POI VV, anchor CTR, posting penetration, Local GMV | Local business outcomes / Local 业务结果 | No by default / 默认否 |
| Incremental stay and active-day/DAU | Platform experiment outcomes / 平台实验结果 | Yes / 是 |
| Commercialization outcome with central exchange approval | Platform-approved outcome / 平台认可结果 | Yes, with a versioned rate / 是，但必须有版本化兑换率 |

## Local Service behavior world

Local Service remains inside the main Feed rather than receiving an independent
exposure universe. The simulator now includes:

- POI-anchored video exposure, anchor click, detail, favorite, order, and value;
- closed-loop restaurant inventory, internal order, payment, and contribution;
- open-loop outbound intent and observable Pixel conversion;
- city match, POI quality, inventory availability, and fulfillment type;
- decaying post-search intent and stateful POI retargeting;
- eight recall opportunities: ANN, graph, geo, fresh, long-tail, popular,
  post-search, and retarget;
- Feed satisfaction, fatigue, leave, return, Local business metrics, and LT
  platform metrics in one trajectory without equating them.

This creates a reason for realtime features to help. A realtime feature launch
must first demonstrate positive oracle value during an intent shift. If oracle
value is absent, more users or a larger model cannot rescue the launch.

The public evidence boundary is deliberate. ByteDance's Monolith paper supports
collisionless sparse parameters and online training; the public Flink account
supports realtime counters, window counters, and sequence features; neither
publishes this project's LT exchange rates. KuaiSAR supplies an appropriate
public search-and-recommendation behavior source, while KuaiSim supplies the
request, session, and cross-session evaluation framing.

ByteDance's public A/B practice article adds two constraints used here. A
content- or supply-side change should still be evaluated on platform user-side
metrics, with content metrics as supporting evidence; after a short-term win,
a reverse experiment should remain to verify long-horizon value. This supports
the experiment protocol, not any proprietary LT formula.

字节公开的 A/B 实践还支持两条约束：内容或供给侧改动仍需观察平台用户侧指标，
内容侧指标只作辅助；短期实验通过后应保留反转实验验证长期收益。这些公开信息只
支持实验协议，不支持推测任何内部 LT 公式。

## Local ranking launch ladder

The same users and exogenous draws are replayed through five policy worlds:

```text
personalized Feed
-> static Local relevance
-> post-search feature
-> search plus retarget state
-> separate Local embedding correction
-> larger Local-value allocation
```

Every adjacent change receives both a stable-user randomized estimate and the
known paired-world DGP effect. The gate uses platform `lt_value` as the primary
metric, with stay/exposure and negative feedback as Feed guardrails. Local Value
Tree, anchor, conversion, and posting metrics are mediators, not independent
reasons to launch.

A Local metric can rise while LT remains neutral. That result is a hold, not a
business win: the added Local value may be too sparse, may replace equal Feed
value, or may be underpowered. Increasing a business multiplier to manufacture
a pass is prohibited; the next action must target intent precision, candidate
quality, funnel conversion, or experiment power.

The Local embedding iteration corrects generic affinity only for intent-matched
POI candidates. Its representation is generated from observable behavior with
lower simulated measurement noise; it never reads the latent preference used
by the DGP. The following load experiment freezes that model and changes only
allocation, making algorithm quality and business pressure separately
identifiable.

After a staged ramp, the reverse-holdout protocol serves the new model to both
cells during burn-in, then reverts only control to the old model. Measurement
starts after the switch. Platform LT is decomposed into stay, active-day, and
accepted-commercialization confidence intervals so noisy retention cannot be
misreported as model value.

分阶段放量后，反转实验先让两组都使用新模型；burn-in 结束后只将 control 切回旧
模型，并从切换后开始计量。平台 LT 必须分别报告 stay、active-day 与 accepted
commercialization 的置信区间，避免把留存噪声误报为模型价值。

## Mixer and COPP scope

The current online mixer calibrates Organic, Live, and Ad scores, applies type
caps, and prevents more than two consecutive items from one category. The local
`copp` adapter is a deterministic greedy constrained selector with Fresh,
creator, and category constraints. Neither is claimed to reproduce proprietary
TikTok internals.

The next mixer iteration must make load observable at five points:

```text
requested -> eligible -> admitted -> served -> effective
```

Its candidate envelope carries source queue, business prediction heads, Value
Tree score, organic opportunity cost, position response, frequency state,
pacing state, latency deadline, and rejection reason. It does not carry a
fabricated per-item LT exchange. Hard safety and eligibility precede
optimization; creator/category/advertising gaps are hard constraints; novelty,
redundancy, and exploration remain soft penalties.

## Failure localization

Every feature, recall, model, or realtime launch is diagnosed in this order:

```text
DGP opportunity
-> oracle effect
-> observable feature information
-> model learning and calibration
-> recall/coarse pass-through
-> fine-rank order change
-> business Value Tree and policy fusion
-> mixer admission and final exposure
-> triggered effect and overall ITT
-> long-horizon holdout
```

This ordering distinguishes a simulator that lacks the intended causal state
from a weak feature, a model miss, a cascade bottleneck, a mixer overwrite, or
an underpowered experiment.

## Acceptance bars

- Long-view terminology is absent from LT container fields and reports.
- Business Value Trees and the LT metric container are separate types and APIs.
- Local consumption, transaction, and supply remain separate Value Tree branches.
- Offline and tensor simulators consume the same exchange-rate authority.
- Closed- and open-loop Local conversions are reported separately and close to
  the aggregate conversion metric.
- Funnel prefixes are not double counted inside the Local Value Tree.
- Each GPU launch records one million users, three seeds, randomized estimates,
  known DGP effects, throughput, peak memory, and evidence limits.
- Every composite LT metric reports its 80% MDE and the total users required to
  detect known DGP truth under frozen variance.
- Commercialization-sensitive launches report λ=0, 0.1, 0.25, and 1.0 on the
  same trajectories plus observed and known-DGP break-even rates.
- Cluster estimators require repeated-DGP coverage calibration; a single
  significant p-value cannot override a missed known truth.
- Reverse holdouts exclude burn-in outcomes and retain component-level LT
  inference until the planned power target is reached.
- A launch passes only when LT improves significantly without Feed guardrail
  regression; Local-only improvement remains a hold.

## Public references

- [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663)
- [ByteDance realtime feature-system evolution](https://developer.volcengine.com/articles/7317094357104853001)
- [KuaiSAR: A Unified Search and Recommendation Dataset](https://arxiv.org/abs/2306.07705)
- [KuaiSim: A Comprehensive Simulator for Recommender Systems](https://arxiv.org/abs/2309.12645)
- [ByteDance recommendation A/B experiment practice](https://developer.volcengine.com/articles/7330151581432610855)
