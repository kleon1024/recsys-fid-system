# F-LR-014 — Triggered retarget ablation

Decision: `hold_unified_lt_uncertain`  
Change: remove `retarget_match` from the accepted feature artifact  
Experiment: 1M users, eight-step burn-in, 3.143% pre-treatment retarget cohort

Retarget eligibility is derived only from control-period anchor behavior, so a
treatment cannot select its own analysis cohort. Overall unified LT is +0.0186%
with an interval crossing zero. Eligible-user LT is +2.218%, also uncertain;
the projected all-user ITT is +0.0528%. The active and rollback artifacts remain
unchanged.

重定向人群只由 burn-in 阶段的行为决定，避免 post-treatment conditioning。整体和
triggered LT 均跨零，因此 Hold，不能把离线 AUC 的微小变化当作删除依据。

Evidence: `reports/launches/2026-08-23-feature-lr-intent-trigger-1m-gpu.json`
