# L-FEED-FOUNDATION-010: Experience-aware DGP and video dedup

This is synthetic engineering evidence, not TikTok production evidence.

本文是合成数字孪生的工程证据，不代表 TikTok 生产指标。

## Root cause / 根因

The evolving factual world used `FormulaResponseAuthority`. Its utility model
could not identify an exact repeated item, creator saturation, or a run of
disappointing slates. Satisfaction and fatigue were updated after return/churn
had already been sampled. Popular therefore received repeated positive labels,
while video dedup appeared to reduce stay by removing artificially valuable
repeats.

连续世界实际使用 `FormulaResponseAuthority`。原行为模型看不到同一 item 重复、作者
饱和和连续低质 slate，而且先采样回访/流失，后更新满意度和疲劳。结果是 Popular 的
重复视频仍不断获得正反馈，消重反而看似伤害停留。

## DGP epoch migration / DGP 版本迁移

The migration is not a recommendation A/B. It changes the hidden environment
authority from `formula-invariant-oracle-v1` to
`behavioral-scm-v2:hidden-experience-v1`, backfills hidden exposure memory, and
starts a new request-stream authority. Runtime restore permits only this exact
preregistered manifest change plus additive state fields.

迁移不是推荐策略实验。它只把隐藏环境切换到可识别重复体验的 Behavioral SCM，回填
隐藏曝光记忆，并开启新的 request stream。Checkpoint 只允许预注册的 response
authority 变化和新增状态字段，其他 runtime drift 继续 fail closed。

| Evidence | Result |
|---|---:|
| Migration parent | `c9c2d232c0808b91825231b47f22016606bdc8bbeff948d73c5d3efdaba76586` |
| V5 migration checkpoint | `b26c9d414104c94e1fd2e061723ab45d524b02d3ffae4d7a5b815c428ba53f09` |
| Users with backfilled memory | 17,245 |
| Backfilled exposure rows | 1,807,032 |
| V4 request partitions eligible for V5 training | No |

## Behavioral falsification / 行为证伪

The same Feed video, shown again under the same request-keyed structural noise,
must have lower utility and play, more slide, more session exit, lower hidden
satisfaction, and higher fatigue. The executable test enforces all six
directions. The 4090 factual baseline at ticks 196-203 produced the same
direction:

在相同 request-keyed 随机噪声下，同一 Feed 视频再次出现必须降低 utility 和播放，
提高划走和退出，并降低隐藏满意度、提高疲劳。可执行测试固定这六个方向。4090 上
tick 196-203 的真实连续世界也得到相同方向：

| Outcome | Fresh video | Repeated video |
|---|---:|---:|
| Impressions | 1,280 | 15,280 |
| 3-second play rate | 16.48% | 0.00% |
| Long-view rate | 7.81% | 0.00% |
| Dwell-event rate | 23.59% | 0.007% |
| Slide rate | 43.52% | 49.78% |
| Request session-exit rate | 7.84% | 63.00% |

This result also exposes a broken serving baseline: most Popular impressions
were repeats. The absolute exit rate is therefore not a calibrated production
claim; it is a mechanism failure that video dedup must remove before cold-start
exploration or model comparison.

这也暴露了 serving baseline 的真实故障：Popular 曝光大部分已经重复。绝对退出率
不是经过真实日志校准的生产结论，而是必须在冷启探索和模型比较之前修复的机制故障。

## F-LR-010 / 视频消重实验

The treatment changed one owner only: exact short-video exposure dedup over the
then-current 64-item observable ledger and a 240-tick retention window. Search,
creator-page revisit, and other explicit-intent surfaces were unaffected.

Treatment 当时只改一个 owner：主 Feed 短视频在 64 条可观测曝光账本和 240 tick
窗口内不再重复。搜索、作者页回看和其他主动意图场景不受影响。

| Final gate, ticks 204-227 | Control | Treatment | Delta |
|---|---:|---:|---:|
| Triggered users | 579 | 508 | -- |
| Repeated-video rate | 22.83% | 0.00% | -22.83 pp |
| Dwell seconds / triggered user | 1.718 | 3.319 | +93.22% |
| 3-second plays / triggered user | 0.252 | 0.478 | +89.70% |
| Long views / triggered user | 0.130 | 0.262 | +102.12% |
| Negative feedback / triggered user | 2.420 | 2.520 | CI crosses zero |

Decision: **promote inside the synthetic factual world**. The new active
checkpoint is
`6da89d15187fff039e6bda8b13031ed3ebd8f506ef6a53db97b73820cce19230`,
and the active policy is `feed-window-dedup-v1`.

决策：**仅在合成连续世界中晋级**。新基线已经启用 `feed-window-dedup-v1`。

## Post-launch consistency correction / 上线后一致性修复

The post-launch audit found that the 240-tick policy window was wider than the
64-entry storage authority. With eight Feed impressions per request, the
effective memory could collapse to roughly eight recent Feed requests. The
fix does not change the behavioral objective: it separates the 64-event model
sequence from a 1,024-entry Feed-only serving ledger. Checkpoint
`3fa575dc9f52b72657ed8bc2a97dabfc44709ecf5c2940a08b3acb7f81de8a00`
contains 1,969,936 backfilled Feed impressions for 17,304 users; the maximum
per-user count is 728. RTX 4090 peak allocation rose from 4.81 to 5.50 GiB.

上线后审计发现，策略虽然配置为 240 tick，但存储只有 64 条；每个请求曝光 8 条时，
有效记忆可能退化为约 8 个 Feed 请求。修复后，64 条行为序列继续服务模型，新增的
1,024 条 Feed 专用账本服务强消重。它只记录 Feed 曝光，不受 Search、Posting 等
场景污染。该修改通过 additive checkpoint migration 回填，并未重启世界或清空用户。

## What is not complete / 尚未完成

Item cold start is now implemented as a propensity-logged randomized layer,
but it is not promoted. Its current evidence and HOLD decision are recorded in
[R-LR-012](2026-08-25-feed-v5-cold-start.md).

Item 冷启已经实现为记录 propensity 的随机实验层，但尚未晋级。当前证据和 HOLD
决策见 [R-LR-012](2026-08-25-feed-v5-cold-start.md)。
