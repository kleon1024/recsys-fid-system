# F-LR-001 — Add sequence-match features

Decision: `hold_unified_lt_uncertain`  
Change: Basic LR 7 features → Sequence LR 9 features  
Added: `short_sequence_match`, `long_sequence_match`

Offline AUC improved from 0.6101 to 0.6812, but the million-user A/B estimated
unified LT -0.189%. The absolute LT confidence interval was [-0.05182, +0.00594],
which crosses zero. Stay fell 0.108%, negative feedback rose 9.76%, and oracle
regret worsened 32.0%.

离线 AUC 大涨不能上线。统一 LT 区间跨零，并且 stay、负反馈和候选 regret 同时变差，
因此保留 Basic control，不把 sequence 特征带入下一条已上线链路。

Next action: repair candidate-conditioned sequence semantics before rerunning;
do not increase model complexity to hide a bad feature contract.
