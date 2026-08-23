# F-LR-010 — Remove retarget match

Decision: `hold_unified_lt_uncertain`

Control: Basic + Realtime + Local + Category

Removing `retarget_match` left offline AUC at 0.74460. Unified LT changed by
+0.004%, with absolute confidence interval [-0.02879, +0.02992]. Stay fell
0.089%, quality-long-view fell 0.278%, and online oracle regret was flat.

The feature is neither proven useful nor safely removable under the current
population experiment. 重定向同样属于低触发率信号；保留 active，并把后续实验单位收窄到
具备历史意图或可重定向候选的请求。
