# F-LR-002 — Add realtime user-state features

Decision: `pass_unified_lt_nonnegative` in simulation

Production readiness: `hold_synthetic_rates`

Change: Basic LR 7 features → Basic + Realtime LR 10 features

Added: satisfaction proxy, fatigue proxy, session progress

Sequence was not accepted, so it is not part of this control. Offline AUC
improved from 0.6101 to 0.7294. The million-user A/B estimated unified LT
+1.204%; its absolute confidence interval was [+0.11695, +0.17487]. Stay
improved 3.108%, quality-long-view improved 5.322%, and online oracle regret
fell 12.04%.

旧报告错误地使用未通过的 Sequence 作为 control，因此不能形成上线证据。修正为最后
已接受的 Basic control 后，Realtime 通过模拟 LT 门禁并成为新的 active control。
