# Feed semantic and supply epoch / Feed 语义与供给纪元

Status: active architecture authority

## Invariant / 不变量

The hidden ecosystem owns user needs and creator decisions. The platform sees
only published catalog fields and observable events. Expanding the taxonomy or
catalog never grants a ranker access to hidden state and never rewrites an old
Launch Review.

隐藏生态负责用户需求和作者决策。平台只能读取已发布的内容字段与可观测事件。扩展
topic 或 item 不得向排序器泄露 hidden state，也不得重写旧 Launch Review。

## Current semantic epoch / 当前语义纪元

`feed-semantic-v2` removes arithmetic mappings from contiguous item IDs to
topic, country, creator, merchant and content kind. The standard profile owns
512 topics and 64-dimensional public content embeddings. Random retrieval uses
a counter-random catalog permutation and samples without replacement. Popular
uses mature impression and engagement counters within country-level rotating
pools. A single route bypasses RRF; RRF is tested only as a separate multi-route
mechanism.

`feed-semantic-v2` 删除连续 item ID 到 topic、国家、作者、商家和内容形态的算术
映射。标准 profile 使用 512 个 topic 和 64 维公开内容向量。Random 通过 counter
random permutation 做无放回抽样。Popular 使用国家分层的成熟曝光和互动 counter，
并在热门池内轮转。单路召回不经过 RRF，RRF 只作为独立多路合并机制开实验。

## Expansion protocol / 扩容协议

Topic and item capacity are independent axes:

- More items increase density, freshness, creator competition and tail supply.
- More topics increase semantic resolution and the opportunity for retrieval
  and ranking to distinguish intent.
- A new taxonomy is a versioned World Epoch. Existing user interests and
  creator specialties are projected into the new taxonomy, followed by burn-in;
  they are not reset.
- Item tensors grow by append-only catalog generations. Active, removed and
  expired states remain bounded checkpoints; historical events remain Parquet
  partitions rather than resident tensors.

Topic 与 item 是两个独立扩容轴。新增 item 提升密度、时效性、作者竞争和长尾供给；
新增 topic 提升语义分辨率与排序空间。taxonomy 变化必须创建版本化 World Epoch，
迁移用户兴趣和作者专长并 burn-in，不能清零。item 通过 append-only catalog
generation 扩容；历史事件留在 Parquet 分区，不常驻 World tensor。

## Creator feedback loop / 作者反馈闭环

The current supply state already updates creator motivation, publishing cost,
cooldown and retention from factual impressions, positive actions and negative
feedback. The next supply epoch must additionally own creator topic specialty,
topic exploration, quality investment and expected-return uncertainty. A creator
then chooses whether to post, which topic/format to produce and how much quality
effort to invest. Publication creates an observable item; subsequent Feed
distribution changes future creator decisions.

当前供给状态已经根据真实曝光、正向互动和负反馈更新作者动力、成本、冷却时间与留存。
下一供给纪元还要负责作者 topic 专长、topic 探索、质量投入和收益不确定性。作者先决定
是否投稿、生产什么 topic/形态以及投入多少质量成本；发布后形成可观测 item，Feed
分发结果再反向影响后续供给。

Acceptance requires creator-cluster or switchback experiments, qualified supply
and creator retention, topic/author concentration, consumer stay and negative
feedback. Immediate posting count alone cannot authorize a creator incentive.

验收必须同时报告作者聚类或 switchback 实验、有效供给、作者留存、topic/作者集中度、
消费 stay 和负反馈。单独提升投稿量不能证明作者激励成立。

## Launch evidence / 上线证据

Every LR automatically writes a content-bound Launch Bundle: event-time request
partitions, route candidates, cascade decisions, exposed slate, point-in-time
features, events, mature-label masks, sample authorities, A/B report and one-click
DuckDB diagnosis. Changing index versions across ticks creates separate immutable
partitions; validation never weakens manifest compatibility to concatenate them.

每次 LR 自动生成 content-bound Launch Bundle，包括 event-time request 分区、召回
候选、级联决策、曝光 slate、point-in-time 特征、事件、成熟标签 mask、三类样本、
A/B 报告和一键 DuckDB 诊断。跨 tick 的 index version 必须形成独立不可变分区，
不能通过放宽 manifest 校验强行拼接。
