# L-LOCAL-SCALE-003 — Local intent ranker 10M power review

The 10-million-user-per-seed scale review changes only experiment population.
Catalog, candidates, model, features, DGP, and gates are frozen.

本次每 seed 1000 万用户的规模实验只扩大实验人数；Catalog、候选、模型、特征、
DGP 和门禁全部冻结。

| Launch | Change / 变更 | Platform LT | Known LT truth | Stay | Local tree | Decision |
|---|---|---:|---:|---:|---:|---|
| L-LOCAL-SCALE-001 | Separate Local embedding correction / 独立 Local 表征纠偏 | +0.034%, p=.235 | +0.061% | +0.113%, p<1e-34 | +3.33% | Hold |
| L-LOCAL-SCALE-002 | Expand Local load / 扩大 Local load | -0.081%, p=.0050 | -0.054% | -0.140%, p<1e-52 | +11.42% | Reject |

L-LOCAL-SCALE-001 is an algorithmic improvement rather than a product boost.
For intent-matched POI candidates it replaces the noisy generic affinity with
a lower-noise Local representation learned from observable behavior. It raises
both stay and Local outcomes, but 30 million users across three seeds still do
not provide 80% power for the smaller composite LT effect. The estimated total
requirement is 52.6 million users under the frozen variance.

L-LOCAL-SCALE-001 是算法改进，不是产品调权。对于命中 Local 意图的 POI 候选，
它用可观测行为学习的低噪声 Local 表征纠正通用 affinity。它同时提升 stay 与 Local
指标，但三个 seed 共 3000 万用户仍不足以对更小的综合 LT 效应达到 80% power；
冻结方差下预计需要 5260 万用户。

L-LOCAL-SCALE-002 proves why Local Value Tree cannot replace LT: Local tree
rises 11.42% while platform LT and stay both significantly decline.

L-LOCAL-SCALE-002 证明 Local Value Tree 不能替代 LT：Local Tree 提升 11.42%，
但平台 LT 与 stay 均显著下降。

The run sustained 9.98M–10.29M simulated requests/s on RTX 4090 with 299 MiB
peak allocated GPU memory.

Numeric authority / 数值权威：
`reports/launches/2026-08-23-local-intent-ranker-scale-10m.json`.
