# L-LOCAL-LT-002 — Local Service platform-LT ladder

Decision: `HOLD` for all five million-user launches. Local Value Tree and Local commerce
improved in several cells, but neither is an LT input. The platform LT estimate
uses only stay, active-day/DAU proxy, and platform-accepted commercialization;
this Local experiment contributed zero accepted commercialization.

结论：五次百万用户迭代全部 `HOLD`。多组实验的 Local Value Tree 与 Local 商业指标明显
上升，但两者都不是 LT 输入。平台 LT 仅使用 stay、active-day/DAU 代理指标和平台
认可的商业化兑换项；本组 Local 实验的认可商业化输入为零。

## Experiment / 实验

- 1,000,000 users × 24 requests × three seeds on RTX 4090.
- Stable UID 50/50 assignment and common-random paired DGP truth.
- Primary: `lt_value_per_user`; Feed stay non-inferiority: -0.5%.
- Local Value Tree, anchor, conversion, and Local commerce are diagnostics.

- RTX 4090 上每个 seed 为 100 万用户 × 24 次请求，共三个 seed。
- 稳定 UID 50/50 分桶，并保留 common-random paired DGP 真值。
- 主指标为 `lt_value_per_user`；Feed stay 非劣线为 -0.5%。
- Local Value Tree、anchor、转化和 Local 商业价值只作业务诊断。

## Results / 结果

| Launch | Change / 变更 | Platform LT | Stay | Local Value Tree | Local commerce | Decision |
|---|---|---:|---:|---:|---:|---|
| L-LOCAL-GPU-001 | Static Local relevance / 静态 Local 相关性 | +0.090%, p=.328 | +0.075%, p=.0091 | +15.55% | +17.58% | Hold |
| L-LOCAL-GPU-002 | Post-search / 搜后推 | +0.063%, p=.491 | +0.024%, p=.405 | +0.61% | -0.07% | Hold |
| L-LOCAL-GPU-003 | Exact-item retarget / 同物品重定向 | +0.082%, p=.370 | +0.065%, p=.024 | +3.32% | +3.94% | Hold |
| L-LOCAL-GPU-004 | Local embedding correction / Local 表征纠偏 | +0.121%, p=.185 | +0.152%, p<.000001 | +3.49% | +4.36% | Hold, LT underpowered |
| L-LOCAL-GPU-005 | Larger Local allocation / 扩大 Local load | +0.007%, p=.942 | -0.100%, p=.00056 | +11.64% | +13.19% | Hold |

The fifth launch is the important counterexample: business value rises while
platform value is neutral and known stay truth declines. It cannot pass by
arguing that Local GMV, POI VV, anchor CTR, or posting penetration should be
inserted into LT.

第五次迭代是关键反例：业务价值显著上涨，但平台价值中性，且 stay 已知真值下降。
不能通过把 Local GMV、POI VV、anchor CTR 或投稿渗透率直接塞进 LT 来制造通过。

## Superseded evidence / 作废证据

The earlier report directly converted Local Value Tree leaves into LT and was
removed from the evidence manifest. It is not a historical launch win.

旧报告曾把 Local Value Tree 叶子直接兑换进 LT，已从证据清单移除，不能作为历史
上线收益。

Numeric authority / 数值权威：
`reports/launches/2026-08-23-local-platform-lt-ladder.json`.
