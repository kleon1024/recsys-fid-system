# L-SIMULATOR-004 — Externally calibrated Feed behavior kernel

Decision: accept V3 as the current research epoch; hold the Local intent ranker  
Evidence boundary: public KuaiRand standard-policy marginals plus synthetic GPU A/B

## Root cause

The V2 counter RNG preserved batch invariance but used additive stream offsets.
Play, stay-noise, like, and negative-feedback draws therefore had correlations
above 0.998. V2 also defined long view at 10 seconds while KuaiRand uses video
completion for short videos and 18 seconds otherwise. Finally, nonlinear stay
truth was injected by one policy class, so changing the model could change the
environment response function.

These defects invalidate V2 as the authority for future model comparisons. Its
records remain historical engineering evidence only.

## Repair and external calibration

V3 uses an avalanche-mixed counter RNG. The maximum measured cross-stream
correlation fell to 0.0013 while results remain invariant to GPU batch
partitioning. The immediate-response kernel is owned by the simulator signal
version and is identical for control and treatment.

The calibration snapshot contains 1,436,609 public KuaiRand-Pure interactions,
27,077 users, and 7,551 videos. The raw files remain outside Git; the report
binds their exact SHA-256 hashes.

| Metric | KuaiRand standard log | V3 control | Relative error |
|---|---:|---:|---:|
| Positive play | 86.846% | 86.915% | +0.08% |
| Three-second play | 58.808% | 56.868% | -3.30% |
| Long view | 33.184% | 36.086% | +8.74% |
| Like | 1.848% | 2.137% | +15.62% |
| Hate / negative | 0.0494% | 0.0496% | +0.29% |
| Stay seconds per exposure | 22.893 | 23.107 | +0.94% |

## Launch result

At one million users and 24 steps, `local_intent_quality_rank_v4` versus the
personalized Feed control produced unified LT -0.162%. The absolute 95%
confidence interval is [-0.03949, 0.00371], so the decision is
`hold_unified_lt_uncertain`.

This does not prove production lift. The available official KuaiSim subset has
standard-policy logs and no random-exposure file, so it calibrates behavior
marginals but cannot provide unbiased off-policy evaluation.

## Next model launch

`L-MODEL-001` used the old CPU candidate world and only 45,794 impressions. The
next model ladder must train and evaluate LR, XGBoost, W&D, DeepFM, DCNv2, and
MMoE on one V3 request-level candidate dataset. No result may be compared across
V1, V2, and V3 evidence epochs.

Numeric authority:
`reports/launches/2026-08-23-feed-calibrated-v3-1m-gpu.json`.

