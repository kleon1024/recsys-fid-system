# L-MODEL-001 — Fine-rank model ladder under platform LT

Decision: keep LR. W&D, DeepFM, DCNv2, and MMoE are all rejected in this
synthetic launch. Model complexity is not a launch criterion.

结论：保留 LR。本次合成实验中 W&D、DeepFM、DCNv2 和 MMoE 均被拒绝；模型复杂度
不是上线条件。

## Authority / 权威边界

These are actual scikit-learn, DeepCTR-Torch, and PyTorch models. They train on
the same exploratory impression log, score the same candidate batches, change
the selected videos, and drive fresh-user request/session trajectories. This is
fine-rank evidence; it does not claim that the same Python inference path can
serve a million-user tensor experiment.

这些是真实训练的 scikit-learn、DeepCTR-Torch 与 PyTorch 模型。它们使用同一份
探索日志、排序同一批候选、改变最终曝光，并驱动 fresh-user 请求与 session 轨迹。
这属于精排证据，不声称 Python 推理链路能够直接服务百万用户 tensor 实验。

- 3,000 logging users, 8,000 items, and 45,794 impressions.
- Session 0 train: 20,122; session 1 validation: 12,450.
- Sessions 2–3 test: 13,222 impressions from 1,357 returning users.
- Fresh-user A/B: 5,000 disjoint users; clipped-IPS resampling ESS: 4,048.
- LT inputs: stay, active-day, and accepted commercialization only. Accepted
  commercialization is zero in this model launch.

## Offline and candidate quality / 离线与候选质量

| Model | AUC | GAUC | Oracle regret | Parameters |
|---|---:|---:|---:|---:|
| LR | 0.7175 | 0.6794 | 0.0625 | — |
| W&D | 0.6349 | 0.6450 | 0.1403 | 59,515 |
| DeepFM | 0.6370 | 0.6470 | 0.1409 | 59,515 |
| DCNv2/DCN-Mix | 0.6011 | 0.6096 | 0.1505 | 87,429 |
| MMoE | 0.6923 | 0.6823 | 0.0929 | 28,671 |

The synthetic DGP is dominated by learnable low-order signals at this sample
size. LR therefore generalizes better than the larger models. Lower neural
training loss is not evidence of better candidate ordering.

该 DGP 在当前样本量下主要由可学习的低阶信号构成，因此 LR 的泛化优于更大模型；
神经网络 training loss 更低不能证明候选排序更好。

## Fresh-user A/B / Fresh-user 实验

| Treatment vs LR | Known stay | Known platform LT | Known quality view | Known Local tree | Decision |
|---|---:|---:|---:|---:|---|
| W&D | -4.49% | +0.32% | -24.11% | -2.24% | Reject quality guardrail |
| DeepFM | -4.20% | +0.66% | -23.26% | -5.61% | Reject quality guardrail |
| DCNv2 | -4.23% | +2.12% | -22.31% | +1.90% | Reject quality guardrail |
| MMoE | -5.58% | -1.46% | +5.40% | -24.95% | Reject stay regression |

Positive composite LT does not rescue W&D, DeepFM, or DCNv2 because their
quality-view and stay regressions are hard guardrails. MMoE improves its quality
head but worsens stay and LT. A gate-order defect initially returned an
underpowered LT Hold before checking significant stay regression; hard
guardrails now execute first and the launch is rejected.

W&D、DeepFM、DCNv2 的综合 LT 即使为正，也不能覆盖 quality-view 与 stay 硬护栏
回退。MMoE 提升了 quality head，但 stay 与 LT 都下降。此前门禁顺序错误，先返回
LT 风险 Hold、遮蔽显著 stay 回退；现在硬护栏优先，结果为 Reject。

Numeric authority / 数值权威：
`reports/launches/2026-08-23-feed-model-ladder.json`.
