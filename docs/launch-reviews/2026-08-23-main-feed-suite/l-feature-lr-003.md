# F-LR-003 — Add Local intent context

Decision: `pass_unified_lt_nonnegative` in simulation  
Production readiness: `hold_synthetic_rates`  
Change: Realtime LR 12 features → Local-context LR 19 features

Added POI indicator, post-search match, retarget match, POI quality, inventory,
distance, and Local-interest affinity. Offline AUC moved from 0.7538 to 0.7653.
The million-user A/B estimated unified LT +3.802%; its absolute confidence
interval was [+0.43147, +0.48958]. Stay improved 10.02%, quality-long-view
19.44%, negative feedback fell 11.25%, and oracle regret fell 52.94%.

这是模拟器中第一个可 ramp 的小步 Feature LR。增量很大，是因为 control 缺失整组
Local intent 信号，不能类比成熟线上系统的千分位收益。公开项目兑换率仍为 synthetic，
所以真实 production readiness 保持 hold。

Next action: make Local-context LR the simulated serving control, retain a
long-term holdout, then split search, retarget, and geo into separate follow-up
Launch Reviews.
