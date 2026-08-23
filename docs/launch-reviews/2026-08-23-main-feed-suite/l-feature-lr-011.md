# F-LR-011 — Remove POI quality and inventory

Decision: `hold_unified_lt_uncertain`

Control: Basic + Realtime + Local + Category

Removing `poi_quality` and `inventory_available` improved offline AUC from
0.74459 to 0.74487. Unified LT changed by +0.100%, with absolute confidence
interval [-0.01665, +0.04207]. Stay improved 0.139% and online oracle regret
improved 1.53%, but the LT interval still crossed zero.

This is a promising simplification candidate, not a pass. 离线与候选质量方向一致，但最终
LT 证据不足；继续 shadow 或扩大预声明样本，而不是反复换 seed 获得显著性。
