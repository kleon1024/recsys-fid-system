# F-LR-007 — Add category hash

Decision: `pass_unified_lt_nonnegative` in simulation

Production readiness: `hold_synthetic_rates`

Control: Basic + Realtime + Local

Offline AUC improved from 0.7409 to 0.7446. Unified LT improved 0.366%; its
absolute confidence interval was [+0.01699, +0.07560]. Stay improved 0.785%
and quality-long-view improved 1.654%. Negative feedback rose 4.13% and online
oracle regret rose 14.29%.

Category is the only part of the rejected bundle that clears the unified LT
gate, so it is atomically promoted and the prior control becomes rollback.
这是“最终价值通过、诊断指标存在代价”的上线，不是所有指标全面改善。真实生产仍需接受
长期兑换率和 hard-constraint review；当前只更新模拟 release state。
