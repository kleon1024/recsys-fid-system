# L-SIMULATOR-007 — External sequential behavior kernel

Change type: replace V3-label distillation with external chronological behavior evidence

Decision: Pass external sequence capacity; hold V4 causal-authority promotion

Hardware: RTX 4090 24GB, CUDA 12.6

Evidence:

- `reports/datasets/2026-08-24-kuairand-external-sequence-manifest.json`
- `reports/world-model/2026-08-24-kuairand-external-capacity-4090.json`

## Source and boundary

The source is the official [KuaiSim repository](https://github.com/Applied-Machine-Learning-Lab/KuaiSim)
at commit `2ae32aa25a0aac103194a66e6864e3d2ac6d6580`. Its bundled KuaiRand-Pure
snapshot contains 1,436,609 standard-policy interactions, static user/item
metadata, seven primitive feedback labels, and watch duration. Raw files remain
outside Git; every source and tensor split is SHA-256 bound. Dataset material is
CC BY-SA 4.0 and is attributed in the manifest.

The official [KuaiRand documentation](https://github.com/chongminggao/KuaiRand)
states that Pure has incomplete sequential coverage and recommends 1K or 27K
for rigorous sequence, OPE, and reinforcement-learning work. Pure therefore
provides external capacity evidence, not final randomized causal authority.

## Point-in-time dataset

Interactions are sorted independently within each user. Every example stores
the preceding 64 item IDs and seven feedback vectors; the target interaction is
never included in its history. Only basic item metadata and user metadata are
used. Future item statistics are excluded.

| Split | Dates | Rows |
|---|---|---:|
| Train | 2022-04-09 to 2022-04-18 | 1,079,797 |
| Validation | 2022-04-19 to 2022-04-21 | 61,315 |
| Test | 2022-04-22 to 2022-05-08 | 295,497 |

The model jointly predicts click, long view, like, comment, forward, follow,
hate, and normalized watch duration. W&D and Transformer consume the same user,
item, author, tag, categorical metadata, dense metadata, labels, and time split;
only the Transformer receives point-in-time history.

## RTX 4090 result

| Model | Long-view AUC | Train time |
|---|---:|---:|
| Logistic regression | 0.55292 | 0.6 s |
| XGBoost | 0.64832 | 1.4 s |
| Wide & Deep | 0.70236 | 12.3 s |
| Sequence Transformer | 0.73868 | 37.5 s |

The Transformer also improves all seven binary behavior heads over W&D. Its
validation loss improves through epoch four and the best state is restored.
After permuting complete histories across test requests, long-view AUC falls
from 0.73868 to 0.59807; mean absolute probability movement is 0.18013 and P95
is 0.44610. This is the capacity separation absent from the V3-distilled V4.

三个容量门禁全部通过：W&D 超过 Logistic Regression，Transformer 超过最强
tabular 模型，history permutation 证明增量确实来自历史序列，而不是模型参数量。

The published kernel was reloaded from its SHA-bound state dict and exercised on
128 requests with 20 candidates each. It produced finite probabilities for all
seven behaviors, sampled the selected candidate, and advanced only the selected
item and feedback into the next point-in-time history. This closes the model-to-
slate interface; it does not yet constitute a policy A/B result.

## Decision boundary

The external sequence kernel is accepted for integration into the V4 request
and slate simulator. V4 is not yet promoted because the bundled Pure snapshot
does not include random-exposure logs. Final promotion still requires:

- artifact-bound KuaiRand-1K or equivalent randomized logs;
- randomized intervention recovery and frozen-policy ordering;
- a request/slate adapter that samples multi-behavior outcomes and updates
  history without exposing latent or future information;
- shadow replay and A/B gates against V3 with non-negative unified LT.

No TikTok production-fidelity or real-user-lift claim is made.
