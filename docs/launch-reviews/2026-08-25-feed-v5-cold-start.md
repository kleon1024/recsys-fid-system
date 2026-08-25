# R-LR-012: Feed item cold-start exploration

This is synthetic digital-twin evidence, not TikTok production evidence.

本文是合成数字孪生证据，不代表 TikTok 生产指标。

## Changed owner / 变更边界

The experiment reserves at most one Feed slot for an eligible cold-start short
video. Twenty percent of Feed traffic enters treatment assignment and ten
percent of supported treatment requests are randomized, so expected global
exploration is two percent. Every randomized request logs marginal admission,
position and slate propensity. Corpus, rankers, Value Tree and other business
surfaces remain fixed.

实验最多给符合条件的冷启短视频一个 Feed 槽位。20% Feed 流量进入 treatment，
其中有候选支持的请求按 10% 随机化，因此全局探索目标约为 2%。每次随机曝光记录
候选进入、位置和 slate propensity；语料、排序模型、Value Tree 和其他业务场景不变。

## Recovered prerequisite / 前置链路修复

The first attempt had zero support because a Feed recall experiment had
overwritten the shared `enabled_routes` list and silently disabled Posting and
other business routes. That prevented creator posting, new supply and cold-start
eligibility. Route ownership is now split: Feed routes are experimentally
mutable, while business-surface routes are additive and independently owned.
After recovery, ticks 236-251 produced 25 Posting requests, 693 Posting recall
candidates, 200 exposures, 18 creates, four publishes and four active cold-start
items.

第一次实验没有候选，不是冷启策略无效，而是 Feed 召回实验覆盖了共享 route 配置，
误关了 Posting 等业务路由，投稿、供给和冷启资格随之断链。现在 Feed route 与业务
surface route 分属不同 owner。修复后的 tick 236-251 产生了 25 个投稿请求、693 个
投稿召回候选、200 次曝光、18 次开拍、4 次发布和 4 个活跃冷启 item。

## Final gate / 最终门禁

The same stable assignment ran for five cumulative windows over ticks 252-339.
The fourth and fifth reviews used the expanded Feed exposure ledger without
resetting the world or experiment cursor.

同一稳定分流累计运行五个窗口（tick 252-339）。第四、第五个窗口使用扩容后的 Feed
曝光账本，没有重启世界，也没有重置实验游标。

| Evidence | Result |
|---|---:|
| Control / treatment triggered users | 983 / 990 |
| Feed requests | 13,061 |
| Randomized cold-start exposures | 257 |
| Invalid cold-start exposures | 0 |
| Global randomized request rate | 1.97% |
| Within-treatment randomized rate | 10.18% |
| Treatment candidate support | 99.88% |
| Logged propensity range | 1.11%-10.00% |
| Stay delta per triggered user | -3.37% |
| Stay 95% CI, absolute seconds | [-1.261, 0.651] |
| Negative-feedback delta | -6.57% |
| RTX 4090 peak allocation | 5.53 GiB |

Decision: **STOP INCONCLUSIVE** after the preregistered fifth review window.
Assignment, support, route provenance and propensity gates pass, but stay never
proved the five-percent non-inferiority margin. No threshold was relaxed. The
layer was removed and checkpoint
`e8d7388bfaecb0ad050dc02c0f06896d749d3dfaafc745261e8ecfe9774eff29`
returned the factual world to `multi-surface-route-isolated-v1`.

决策：预注册第五窗口后 **STOP INCONCLUSIVE**。流量、支持度、route provenance 和
propensity 均正确，但 stay 始终没有证明 -5% 非劣界限。门槛未放宽，探索层已撤回，
事实世界恢复到原基线。
