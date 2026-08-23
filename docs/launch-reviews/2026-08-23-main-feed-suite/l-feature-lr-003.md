# F-LR-003 — Add Local intent context

Decision: `pass_unified_lt_nonnegative` in simulation  
Production readiness: `hold_synthetic_rates`  
Change: Basic + Realtime LR 10 features → add Local context, 17 features

Added POI indicator, post-search match, retarget match, POI quality, inventory,
distance, and Local-interest affinity. Offline AUC moved from 0.7294 to 0.7409.
The million-user A/B estimated unified LT +3.215%; its absolute confidence
interval was [+0.36502, +0.42330]. Stay improved 8.194%, quality-long-view
16.43%, negative feedback fell 10.48%, and online oracle regret fell 67.27%.

这是模拟器中第一个可 ramp 的小步 Feature LR。增量很大，是因为 control 缺失整组
Local intent 信号，不能类比成熟线上系统的千分位收益。公开项目兑换率仍为 synthetic，
所以真实 production readiness 保持 hold。

The release state atomically promotes this artifact and preserves Basic +
Realtime as rollback. Next split search, retarget, geo, and inventory into
separate proposals rather than expanding this bundle.
