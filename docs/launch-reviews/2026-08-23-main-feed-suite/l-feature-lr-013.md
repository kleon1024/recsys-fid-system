# F-LR-013 — Triggered post-search ablation

Decision: `hold_unified_lt_uncertain`  
Change: remove `post_search_match` from the accepted feature artifact  
Experiment: 1M users, eight-step burn-in, 4.001% pre-treatment search cohort

The former all-user search state was invalid experiment evidence. Search is now
an exogenous sparse event with a request TTL, and the eligible cohort is frozen
before treatment. Overall unified LT is +0.0341% with an interval crossing zero.
Within eligible users LT is +0.427%, also uncertain; the diluted projected ITT
is +0.0171%. The active and rollback artifacts remain unchanged.

旧实验把所有用户都当作搜后用户。本次只在 treatment 前冻结的 4.001% 搜后人群中
估计 conditional effect，并同时报告全量 ITT。两者均不显著，因此不能删除该特征。

Evidence: `reports/launches/2026-08-23-feature-lr-intent-trigger-1m-gpu.json`
