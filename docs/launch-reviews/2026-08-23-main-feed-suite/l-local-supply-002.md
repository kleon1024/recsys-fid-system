# L-LOCAL-SUPPLY-SWITCHBACK-002 — Posting-supply switchback

Decision: `HOLD_ESTIMATOR_MISS`. The city-period switchback detects posting and
quality-supply gains, but it does not reliably estimate the much smaller
platform LT effect.

结论：`HOLD_ESTIMATOR_MISS`。城市 × 时段 switchback 能识别投稿与优质供给提升，
但不能可靠估计更小的平台 LT 效应。

The test uses 100 cities, 28 periods, alternating two-period blocks, one-period
washout, and 15 million effective user-periods. Posting penetration rises
21.74% and quality-adjusted supply rises 34.03%. Those are Local supply Value
Tree outcomes, not LT.

实验包含 100 个城市、28 个时段、每两个时段切换 treatment，并剔除切换后的首个
washout 时段，共 1500 万有效 user-period。投稿渗透率提升 21.74%，质量修正供给
提升 34.03%；它们属于 Local Supply Value Tree，不属于 LT。

Observed LT is nominally significant at p=.0287, but the known DGP truth lies
outside its confidence interval. The estimator exaggerates the truth by about
60 times, so statistical significance is rejected as estimator failure rather
than reported as a launch win.

观测 LT 的名义 p 值为 .0287，但已知 DGP 真值落在置信区间之外，估计值约是真值的
60 倍。因此该结果被判为估计器失效，不能当作上线收益。

The exact two-way fixed-effects estimator was then audited over 500 DGP seeds.
LT confidence-interval coverage is 95.4%, its significance rate is 4.2%, and
mean bias is -0.0000295 LT/user. The estimator is calibrated overall; the
current seed is a legitimate tail miss and remains Hold.

随后对同一个 two-way fixed effects + city-clustered CR1 估计器运行 500 个 DGP
seed。LT 置信区间覆盖率为 95.4%，显著率为 4.2%，平均 bias 为
-0.0000295 LT/user。估计器整体校准合格，但当前 seed 属于合法尾部漏覆盖，因此
仍然 Hold。

Numeric authority / 数值权威：
`reports/launches/2026-08-23-local-supply-switchback-platform-lt.json`.
