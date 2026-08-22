# L-ONLINE-001 — Joiner-to-PS streaming LR

Status: `reject_primary_regression`. The online-training mechanics pass, but no
tested serving policy protects the main Feed primary metric.

## Change

Consume mature main-Feed FineRankExample records, train LT/HLT/negative-feedback
heads in microbatches, publish versioned parameters, replay the frozen snapshot,
then compare fresh-user trajectories against the established heuristic policy.

## Samples and training

- Mature joined examples: 7,322.
- User-disjoint train/evaluation: 5,841 / 1,481.
- Positive rates: LT 32.19%, HLT 10.64%, negative feedback 0.355%.
- Four epochs, 512-example microbatches, 48 applied PS updates.
- Duplicate update replay: rejected as `duplicate_update` at PS version 48.
- Loss: 1.431 first epoch → 0.882 last epoch.

The first 256-dimensional, 256-bin feature hash was diagnosed as both sparse and
collision-heavy. The corrected version uses 16 bins and a 4,096-dimensional PS.
Independent-user AUC improved from 0.553→0.632 for LT, 0.523→0.644 for HLT,
and 0.463→0.578 for negative feedback. This proves the feature/architecture fix,
but not online business value.

## Consistency and shadow

Schema, FID layout, Joiner, model version, index version, task order, feature
replay, and prediction shadow all pass. Serialized/snapshotted score delta is 0.

## Fresh-user A/B

| Candidate | Stay | LT | HLT | Negative | Long-term Value | Decision |
|---|---:|---:|---:|---:|---:|---|
| PS v1 replacement | -5.48% | +0.01% | +19.02% | -67.13% | +16.74% | Reject stay regression |
| HLT-balanced replacement | -10.90% | -9.84% | -2.27% | -47.53% | +19.61% | Hold HLT risk |
| 0.25 PS-score blend | -5.21% | -8.70% | +3.30% | +68.52% | +2.88% | Reject stay regression |

For the blended candidate, known DGP truth is stay -0.96%, LT -1.09%, HLT
+0.87%, negative -8.33%, and long-term Value +2.18%. The 500-user observed
estimate is noisy, but neither observed nor DGP evidence supports ramping.

## Root cause and decision

The PS model now learns non-random task structure, yet a small linear hashed
model cannot reproduce the established policy's continuous candidate ordering.
Optimizing sparse negative feedback and HLT changes duration/content mix and
loses stay/LT. Keep the online training path in shadow; do not publish it as the
serving authority. The next model change must use a distilled teacher or deeper
incremental model and clear offline Top-K overlap before another user A/B.
