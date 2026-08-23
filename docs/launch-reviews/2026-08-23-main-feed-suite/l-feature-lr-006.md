# F-LR-006 — Add identity hash

Decision: `hold_unified_lt_uncertain`

Control: Basic + Realtime + Local

Offline AUC moved from 0.7409 to 0.7408. Unified LT changed by +0.004%, with
absolute confidence interval [-0.02879, +0.02971]. Stay fell 0.109%,
quality-long-view fell 0.261%, and online oracle regret rose 1.55%.

User, item, and author buckets do not show measurable incremental value at this
sample scale. 这不是“ID 特征无效”的结论，而是当前 hash 空间、样本覆盖与正则化下没有
上线证据。保持 shadow，不晋级，也不把它带入 Category 的 control。
