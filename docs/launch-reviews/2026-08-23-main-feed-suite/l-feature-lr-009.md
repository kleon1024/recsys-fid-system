# F-LR-009 — Remove post-search match

Decision: `hold_unified_lt_uncertain`

Control: Basic + Realtime + Local + Category

Removing `post_search_match` moved offline AUC from 0.74459 to 0.74463.
Unified LT changed by +0.043%, with absolute confidence interval
[-0.02389, +0.03483]. Online oracle regret improved 1.48%, but stay was flat
and the final LT evidence remained uncertain.

搜后推特征可能只影响较小的触发人群；全体用户 LT 功效不足不能等同于无价值。本轮保持
active，下一步应在 request-level trigger slice 上单独测触发实验，而不是删除字段。
