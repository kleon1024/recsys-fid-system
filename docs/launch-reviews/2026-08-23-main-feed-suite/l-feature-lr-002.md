# F-LR-002 — Add realtime user-state features

Decision: `hold_unified_lt_uncertain`  
Change: Sequence LR 9 features → Realtime LR 12 features  
Added: satisfaction proxy, fatigue proxy, session progress

Offline AUC improved from 0.6812 to 0.7538. The million-user A/B estimated
unified LT +0.180%, but its absolute confidence interval [-0.00710, +0.05067]
still crossed zero. Stay improved 0.234%, quality-long-view improved 0.324%,
and oracle regret fell 1.26%.

方向正确但 LT 证据不足，因此不是 reject，也不能 ramp。保留 shadow，扩大 observation
horizon 或使用预先声明的 CUPED covariate，而不是重复抽 seed。
