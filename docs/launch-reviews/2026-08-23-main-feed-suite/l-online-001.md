# L-ONLINE-001 — Joiner-to-PS streaming correctness LR / Joiner 到 PS 流式正确性评审

Status / 状态: `reject_negative_feedback`.

This run validates the semantic online-learning path only. It is deliberately
small and CPU-bound; all scale and power conclusions use the RTX 4090 tensor
engine. / 本次只验证在线学习链路的语义正确性。该路径刻意保持小规模 CPU
执行；规模和统计功效结论统一使用 RTX 4090 tensor engine。

## Scope / 范围

- 2,955 mature joined examples; 2,371 train and 584 evaluation examples. / 2,955
  条成熟 Joiner 样本，训练 2,371 条，独立评估 584 条。
- Two epochs and ten idempotent parameter-server updates; replayed update was
  rejected as `duplicate_update`. / 两轮训练、10 次幂等 PS 更新，重复更新被
  `duplicate_update` 拒绝。
- Loss fell from 1.559 to 1.203. This proves optimization runs, not model
  quality. / Loss 从 1.559 降到 1.203，只证明优化过程可运行，不证明模型有效。
- Schema, FID, Joiner, model/index versions, feature replay and prediction
  shadow all passed; maximum replay delta was zero. / Schema、FID、Joiner、模型与
  索引版本、特征回放及 shadow prediction 全部通过，最大回放误差为 0。

## Offline evidence / 离线证据

| Head / 任务 | AUC | PR-AUC | ECE |
|---|---:|---:|---:|
| Long view / 长播 | 0.525 | 0.328 | 0.064 |
| High-quality long view / 高质量长播 | 0.501 | 0.115 | 0.153 |
| Negative feedback / 负反馈 | 0.570 | 0.0058 | 0.220 |

The sample is intentionally too small for model selection. The near-random
quality head and poor rare-event calibration forbid an online ramp. / 样本只用于
正确性验证，不用于模型选择；高质量长播接近随机、稀疏负反馈校准较差，因此不能放量。

## Fresh-user A/B diagnostic / 新用户 A/B 诊断

| Candidate / 候选 | Stay per exposure / 单曝停留 | Platform LT / 平台 LT | Quality long-view rate / 高质长播率 | Decision / 决策 |
|---|---:|---:|---:|---|
| PS replacement | -6.09% (p=.057) | +5.44% (p=.538) | +7.42% (p=.470) | Hold / 暂缓 |
| Balanced replacement | -7.13% (p=.021) | +19.43% (p=.043) | +1.92% (p=.839) | Reject primary regression / 拒绝主指标回退 |
| 0.25 blend | -2.80% (p=.400) | +10.92% (p=.238) | -10.88% (p=.262) | Reject negative feedback / 拒绝负反馈恶化 |

The blend increased observed negative feedback by 316% (p=.048); its known DGP
effect was also positive, so this is a real guardrail failure rather than an LT
trade. Accepted commercialization contributes zero to LT in this Local run. /
混合策略的观测负反馈上升 316%（p=.048），已知 DGP 方向也为正，因此这是实际护栏
失败，不能用 LT 抵消。本次 Local 实验中，可接受商业化价值对 LT 的贡献为 0。

## Decision / 决策

Keep this CPU path for Joiner, PS idempotency and exact replay tests. Do not use
it for scale claims or as serving authority. The next serving attempt must train
on the GPU path, clear frozen-candidate Top-K overlap and calibration gates, and
then enter the same shadow/replay/A/B protocol. / 保留该 CPU 路径验证 Joiner、PS
幂等和精确回放，不用它证明规模能力，也不设为线上 authority。下一版必须在 GPU
路径训练，先通过冻结候选 Top-K overlap 与校准门禁，再进入同一套 shadow、replay
和 A/B 协议。
