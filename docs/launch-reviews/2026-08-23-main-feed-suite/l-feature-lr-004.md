# F-LR-004 — Add hash-ID, category, and duration features

Decision: `reject_unified_lt_negative`  
Change: Local-context LR 19 features → Full LR 24 features  
Added: user/item/author hash buckets, category, normalized duration

Offline AUC improved from 0.7653 to 0.7693, yet unified LT fell 0.574%. The
absolute LT confidence interval was [-0.10135, -0.04300]. Stay fell 3.07%,
negative feedback rose 3.11%, and oracle regret worsened 8.99%, even though
quality-long-view rose 20.70%.

这是典型的离线提升、线上价值下降：hash/category/duration 学到了 quality proxy，却把
排序推向更长、更高表面质量但整体 LT 更差的内容。上线停止在 Local-context LR；这组
特征必须拆分、重新校准并逐项证明增量，不能作为 Full LR 整包上线。
