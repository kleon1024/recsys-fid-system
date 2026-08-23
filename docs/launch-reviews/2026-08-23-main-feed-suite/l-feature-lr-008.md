# F-LR-008 — Remove POI-video indicator

Decision: `hold_unified_lt_uncertain`

Control: Basic + Realtime + Local + Category

Removing `poi_video_indicator` moved offline AUC from 0.74459 to 0.74467.
Unified LT changed by +0.069%, with absolute confidence interval
[-0.02064, +0.03809]. Stay and quality-long-view were effectively flat, while
negative feedback rose 2.50%.

The removal is not proven nonnegative, so the smaller model does not replace
the active artifact. 离线 AUC 的微小上升不能证明该字段冗余；LT 区间跨零时保持现网
control，而不是为了减少一个字段强行上线。
