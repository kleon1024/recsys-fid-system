# POI Posting / Supply V4 Launch Review

## 中文结论

这次不是再用独立请求套一个规则公式。V4 固定 POI catalog，生成 5 万个重复 creator、每人 8 次投稿请求，并用模型不可见的神经行为世界产生选择、发布、内容相关性、供给质量、负反馈和后续 Feed stay。40 万请求、3 个 seed 均在 RTX 4090 上完成训练与 creator-cluster A/B。

最终上线到模拟器 authority 的组合是 `Popular + Geo recall -> Linear fine rank`。相对规则精排，发布率每请求增加 `0.0001336`，统一兑换 LT 每请求增加 `0.00000677`，95% 区间为 `[0.00000224, 0.00001131]`；相关供给增加，内容负反馈风险下降。所有数字都是合成世界效应，不是 TikTok 生产指标。

W&D 和 MMoE 的离线 AUC 更高。以 seed 20260824 为例，发布 AUC 从 Linear 的 `0.8279` 升到 W&D 的 `0.8543`、MMoE 的 `0.8538`。但在线增量比较必须使用已经通过的 Linear 作为 control。W&D 相对 Linear 的 LT 效应为 `-0.00000213`，MMoE 为 `-0.00000168`，区间都跨零，因此二者拒绝上线。这就是“复杂模型离线更强，但没有提供额外业务价值”的完整 Launch Review，而不是宣称复杂模型永远无效。

Semantic 和 History recall 也未通过。它们相对固定 Popular + Geo control 的发布与 LT 均值为负，所以候选层不晋级。召回、精排和端到端的失败被分别保留，不能用端到端平均值掩盖 stage regression。

样本口径是 request-level candidate closure。每个请求保留召回候选、route bits、曝光 Top-K、served score、creator、请求时序和标签 mask。`select` 在曝光空间成熟；`publish` 是 selected-and-published 整体空间标签；`relevance` 只有在实际发布后才可观测，其余位置 `label_mask=0`，不能写成负样本。

实验单位是 creator，而不是 request。paired replay 使用同一 creator 在 control/treatment potential worlds 的差；模拟在线 A/B 则按 creator 稳定随机分桶，每个 creator 的八次请求始终进入同一实验组。标准误按 creator cluster 计算，避免把同一作者的重复投稿当成八个独立样本。

## English conclusion

Supply V4 uses a fixed POI catalog and 50,000 repeated creators with eight posting requests each. A hidden neural response world generates selection, publication, post-publication relevance, supply quality, negative feedback, and downstream Feed stay. The 400,000-request, three-seed run was trained and evaluated on an RTX 4090 with creator-cluster inference.

The accepted simulator bundle is `Popular + Geo recall -> Linear fine rank`. Against the rule ranker, publication increases by `0.0001336` per request and exchanged LT increases by `0.00000677` per request, with a 95% interval of `[0.00000224, 0.00001131]`. Relevant supply rises and content-risk falls. These are synthetic-world effects, not production TikTok metrics.

W&D and MMoE improve offline publication AUC but fail to add value over the accepted Linear control. Their incremental LT effects are `-0.00000213` and `-0.00000168`; both confidence intervals cross zero. Semantic and History recall also regress against the fixed candidate control. The ledger therefore retains each rejected stage instead of allowing an offline metric or an end-to-end average to overwrite the last accepted control.

The training authority is a request-level candidate dataset. Selection is mature over valid exposures, publication is an entire-space selected-and-published label, and relevance is observable only after publication. Unobserved relevance rows use `label_mask=0`. Simulated online assignment is stable by creator, and uncertainty is clustered by creator so repeated requests are not treated as independent users.

## Release boundary

- Active simulator bundle: `popular_geo_plus_linear`.
- Rollback bundle: `popular_geo_plus_rule`.
- Throughput: about 46K requests/s for the first seed, with 7.67 GB peak allocated GPU memory.
- Artifact replay: exact score parity after save/load.
- External creator logs and randomized supply interventions: missing.
- Production readiness: hold; simulator authority only.

Evidence: [`2026-08-24-poi-posting-neural-v4-400k.json`](../../reports/launches/2026-08-24-poi-posting-neural-v4-400k.json) and [`simulated-poi-posting-control.json`](../../artifacts/releases/simulated-poi-posting-control.json).
