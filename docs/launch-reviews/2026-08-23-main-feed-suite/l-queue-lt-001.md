# L-QUEUE-LT-001 — Organic, Live, and Ads mixing

Status: `CALIBRATION REVIEW REQUIRED`. Live improves stay and platform LT in the
synthetic world. Ads increase accepted ad contribution, but the checked-in LT
exchange rate is synthetic; ad-driven LT passes are sensitivity results, not
launch authorization.

状态：`需要兑换率校准复核`。在合成行为世界中，Live 提升 stay 与平台 LT。Ads
增加平台认可的广告贡献，但仓库中的 LT 兑换率仍是合成参数；广告驱动的 LT 通过
只能作为敏感性结果，不能直接授权上线。

| Launch | Change / 变更 | LT | Stay | Interpretation / 解释 |
|---|---|---:|---:|---|
| L-QUEUE-GPU-001 | Add Live / 增加 Live | +0.850% | rate independent / 与兑换率无关 | Pass |
| L-QUEUE-GPU-002 | Add conservative Ads / 保守广告 | +0.033%, p=.720 | +1.021%, p<1e-27 | Hold; break-even λ≈0.0028 |
| L-QUEUE-GPU-003 | Balanced ad load / 平衡广告 load | -0.018%, p=.846 | +0.513%, p<1e-7 | Hold; break-even λ≈0.0142 |
| L-QUEUE-GPU-004 | Aggressive ad load / 激进广告 load | +0.064%, p=.491 | +0.713%, p<1e-14 | Hold; break-even λ<0 in known DGP |

The two LT columns for Ads are sensitivity scenarios at accepted
commercialization exchange rates λ=0 and λ=0.1. They reuse identical user
trajectories and differ only in post-experiment accounting. Statistical
significance at a synthetic λ cannot authorize launch.

Ads 的两列 LT 分别对应认可商业化兑换率 λ=0 与 λ=0.1。两者复用完全相同的用户
轨迹，只改变实验后的价值核算；在合成 λ 下显著不能授权上线。

Value Tree scores still drive candidate comparison and constrained mixing. LT
is evaluated after exposure from measured platform outcomes. The mixer cannot
emit a private per-item LT score.

Value Tree 仍负责候选比较和约束混排；LT 在曝光后根据平台指标评估。混排器不能
输出业务自定义的单 item LT 分数。

Numeric authority / 数值权威：
`reports/launches/2026-08-23-multi-queue-platform-lt-ladder.json`.
