# F-LR-004 — Add hash-ID, category, and duration features

Decision: `reject_unified_lt_negative`

Change: Active Basic + Realtime + Local LR 17 features → add 5 hash/content features

Added: user/item/author hash buckets, category, normalized duration

Offline AUC improved from 0.7409 to 0.7453, yet unified LT fell 1.024%. The
absolute LT confidence interval was [-0.15867, -0.10035]. Stay fell 4.607%,
negative feedback rose 5.08%, and online oracle regret worsened 27.98%, even
though quality-long-view rose 25.82%.

这是典型的离线提升、线上价值下降：hash/category/duration 学到了 quality proxy，却把
排序推向更长、更高表面质量但整体 LT 更差的内容。Hold/Reject 不改变 release state，
active 仍为 Basic + Realtime + Local；这组特征必须拆分后逐项证明增量。
