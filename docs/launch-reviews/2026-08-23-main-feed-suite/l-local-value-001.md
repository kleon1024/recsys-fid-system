# L-LOCAL-VALUE-001 — Feed-guarded Local Value Tree

Status: `hold_quality_long_view_risk` on a frozen treatment catalog.

## Change

For candidates no more than 0.03 below the base Feed score, add a 0.15 Local
proxy using interest, POI value, city match and quality. Supply is frozen, so
viewer-level randomization is valid.

## Known DGP effect

| Metric | Relative effect |
|---|---:|
| Stay per exposure | -0.014% |
| Long-view rate | +0.028% |
| Quality long-view rate | -0.094% |
| Anchor clicks | 0.000% |
| Local Value Tree | 0.000% |
| Platform LT container | -0.004% |

## Observed 300-user A/B

| Metric | Relative lift | p-value |
|---|---:|---:|
| Stay per exposure | +0.41% | .896 |
| Long-view rate | +5.83% | .338 |
| Quality long-view rate | -17.06% | .081 |
| Anchor clicks | -3.62% | .895 |
| Local Value Tree | -4.05% | .906 |
| Platform LT container | -2.09% | .818 |
| Negative feedback | +87.10% | .285 |

## Decision

Hold. The known Local and LT effects are effectively zero, while the observed
quality-view and negative-feedback directions are unsafe and underpowered.
There is no business case for increasing sample size without first changing the
ranking hypothesis.

结论是 Hold。已知 DGP 中 Local 与平台 LT 增量接近零，观测样本又出现高质长播下降和
负反馈上升方向；应先修改策略，不应为了得到显著结果盲目扩大样本。
