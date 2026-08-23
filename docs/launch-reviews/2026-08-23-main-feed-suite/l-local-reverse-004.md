# L-LOCAL-REVERSE-004 — Post-ramp reverse holdout

Decision: `retain_launch_long_horizon`, with the reverse holdout retained until
the composite LT effect reaches the planned power target.

结论：`retain_launch_long_horizon`，但继续保留反转实验，直到综合 LT 达到预设
power 要求。

Both cells run `local_intent_quality_rank_v4` for 12 burn-in requests. During
the following 36-request measurement window, control reverts to
`local_search_retarget_v3` while treatment keeps v4. Burn-in outcomes are not
counted.

两组在前 12 个请求都运行 v4；后续 36 个 measurement 请求中，control 反转回 v3，
treatment 继续 v4。Burn-in 行为不计入指标。

| Metric | Observed lift | p-value | Known DGP effect | 80% power requirement |
|---|---:|---:|---:|---:|
| Platform LT/user | +0.282% | .00174 | +0.116% | 14.16M users |
| LT stay component/user | +0.125% | .0445 | +0.237% | 1.61M users |
| LT active-day component/user | +0.434% | .00386 | approximately zero | not identifiable at this scale |
| Stay/exposure | +0.130% | .0101 | +0.233% | 0.90M users |
| Local Value Tree | +2.91% | <1e-11 | +3.48% | 0.25M users |
| Accepted commercialization | 0 | 1.0 | 0 | not applicable |

The model has a real positive stay effect and no accepted-commercialization
input. However, observed active-day lift is noise: its known DGP effect is
approximately zero and only one of three seeds has positive truth. Composite LT
therefore passes the current statistical gate but its observed magnitude is not
attributed to retention. The staged launch remains under reverse holdout.

模型对 stay 有真实正向作用，且 accepted commercialization 输入为零；但观测到的
active-day 提升属于噪声，其 DGP 真值接近零，三个 seed 中只有一个为正。因此综合
LT 虽通过当前统计门禁，但不能把观测幅度归因于留存，仍需保留反转实验。

Numeric authority / 数值权威：
`reports/launches/2026-08-23-local-reverse-holdout.json`.
