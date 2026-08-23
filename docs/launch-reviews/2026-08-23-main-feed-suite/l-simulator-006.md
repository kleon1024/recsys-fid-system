# L-SIMULATOR-006 — V4 request-level model-capacity audit

Change type: simulator falsification and model-input contract repair

Decision: Hold V4; sequence capacity is not proven

Hardware: RTX 4090 24GB, CUDA 12.6

Evidence: `reports/world-model/2026-08-23-v4-model-capacity-300k-gpu.json`

## What changed

This review does not tune the DGP until a preferred model wins. It asks whether
the V4 world exposes useful information that requires richer model capacity.
Every challenger uses the same frozen time splits, 300,000 training requests,
100,000 validation requests, 100,000 test requests, 10,000 request-level
candidate slates, sampled V4 long-view labels, and the V4 ensemble probability
as an audit-only oracle.

Three benchmark defects were repaired before comparison:

- head truncation had selected only the earliest request-step blocks; uniform
  sampling now preserves every step in each frozen time split;
- W&D omitted `user_bucket_norm`; sparse and dense fields now cover all 28
  canonical features exactly once;
- the global serving-feature hash included one challenger's private W&D input
  mapping. New artifacts now separate canonical feature schema from model-input
  schema while the historical V3 release keeps its immutable legacy hash.

本次不是为了让深度模型获胜而修改 DGP，而是验证 V4 是否真的产生了需要更强模型才能
利用的信息。修复时间截断、W&D 缺失用户特征和 schema 耦合后，所有模型才进入同口径比较。

## Equal-data result

| Model | Information view | AUC | Request oracle regret |
|---|---|---:|---:|
| Logistic regression | selected sparse/dense | 0.58391 | 0.00845 |
| XGBoost | selected sparse/dense | 0.58575 | 0.00333 |
| Wide & Deep | selected sparse/dense | 0.58006 | 0.01414 |
| DeepFM | selected sparse/dense | 0.57955 | 0.01466 |
| DCNv2 | selected sparse/dense | 0.58233 | 0.01097 |
| DIN request ranker | selected + sequence | 0.58119 | 0.01065 |
| Slate Transformer | selected + sequence + slate | 0.58080 | 0.01059 |

The result is not a W&D hyperparameter failure. XGBoost is already near the V4
information ceiling, and the sequence-aware models receive little additional
usable signal.

## Direct context falsification

On 50,000 held-out requests, permuting the full behavior sequence changes V4
long-view probability by only 0.00121 on average, or 1.64% of the baseline
prediction standard deviation. Replacing the full candidate slate with repeated
copies of the selected candidate changes it by 0.01984, or 26.96% of baseline
standard deviation.

Therefore V4 uses slate context but effectively ignores temporal sequence. The
source is structural: V4 was fitted to V3-generated labels, where short/long
match, satisfaction, fatigue, and other serving-visible scalar summaries already
contain most of the teacher's response signal. A GRU in the architecture cannot
recover temporal causal structure absent from the training world.

## Gates and decision

| Gate | Result |
|---|---|
| Sequence context is material | Fail |
| Slate context is material | Pass |
| Deep interaction model improves over logistic regression | Fail |
| Request model improves AUC over tabular models | Fail |
| Request model reduces candidate regret | Fail |

V4 remains a research artifact and cannot become simulator authority. The next
training authority must consume real or randomized sequential evidence. The
official KuaiSim repository was inspected at commit
`2ae32aa25a0aac103194a66e6864e3d2ac6d6580`; its bundled KuaiRand-Pure snapshot
contains chronological multi-behavior logs and its user-response contract
models item history, feedback history, request-level ranking, whole sessions,
and cross-session retention. KuaiRand-1K remains the preferred reproducible
source for strict long-sequence work because the official KuaiRand documentation
states that Pure is incomplete for sequential research.

Until that external sequence lane is artifact-bound and replayed, V3 remains the
executable synthetic authority and neither V3 nor V4 may claim TikTok real-user
fidelity.
