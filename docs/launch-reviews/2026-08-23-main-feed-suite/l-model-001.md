# L-MODEL-001 — LR to multi-task MMoE Value Tree

Status: `pass_primary_metric` in the synthetic initial-model gate. This is not
company production evidence and is not the expected lift of a mature Feed.

## Change

Replace the single-task LT logistic ranker with a four-expert MMoE. Separate
gates and towers predict LT, HLT, and negative feedback; ranking combines the
three calibrated task outputs. No product-trigger, candidate-budget, or DGP
parameter changed.

## Samples and training

- Train: 6,726 session-0 impressions.
- Validation: 3,959 session-1 impressions.
- Test: 4,101 session-2/3 impressions from 423 returning users.
- Online A/B: 500 disjoint fresh users.
- Selection-bias correction: clipped-IPS resampling from exploratory logs.
- Parameters: 26,295; RTX 4090 train time 0.167 seconds.
- Shadow serialization replay maximum score delta: 0.

The warm test produced AUC 0.6560, PR-AUC 0.3858, NDCG 0.8571, LogLoss 0.5802,
and ECE 0.0964. LR retained higher global AUC, so this launch is not justified by
offline AUC. It is evaluated on multi-objective trajectory value.

Gate entropy is 1.364 for LT, 1.275 for HLT, and 1.268 for negative feedback
against a four-expert maximum of 1.386. Expert utilization is nonzero for every
task, so the run shows neither gate collapse nor a dead expert.

## Randomized A/B

| Metric | Observed lift | Known DGP ITT | p-value |
|---|---:|---:|---:|
| stay/exposure | +6.479% | +2.889% | 0.00384 |
| LT rate | +8.546% | +5.166% | 0.04667 |
| HLT rate | +10.869% | +7.658% | 0.09277 |
| negative feedback | +0.000% | -10.000% | 1.00000 |
| long-term Value | +10.533% | +14.253% | 0.06725 |

The observed treatment-control estimate is noisy at 500 users; known DGP truth
is reported only to audit the estimator and is never available to the model.

## Decision

Pass only as the initial synthetic model upgrade: primary stay and LT clear,
HLT and long-term Value are directionally positive, negative feedback does not
regress, and shadow replay is exact. A real mature-system ramp would require a
larger A/B, per-head calibration improvement, and cold/warm slice guardrails.
