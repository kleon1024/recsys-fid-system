# F-LR-005 — Add duration

Decision: `reject_unified_lt_negative`

Control: Basic + Realtime + Local

Offline AUC improved from 0.7409 to 0.7416, but the million-user A/B estimated
unified LT -1.329%; its absolute confidence interval was [-0.19719, -0.13897].
Stay fell 5.426%, negative feedback rose 3.38%, and online oracle regret rose
13.30%, while quality-long-view rose 24.67%.

Duration is the main source of the rejected five-feature bundle's regression.
It rewards longer videos and the quality-view threshold while reducing total
stay and LT. 离线 AUC 与长播指标同时上升，仍不能覆盖统一 LT 的明确负区间。该特征不晋级，
active control 保持不变。
