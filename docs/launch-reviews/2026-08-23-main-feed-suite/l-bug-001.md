# L-BUG-001 — Stop counting inactive users as plays

Status: `pass_metric_correction`. Synthetic measurement-chain fix.

## Root cause

The throughput engine sampled play for users whose trajectory had already ended
and counted those draws in the numerator while excluding them from exposures.
The observed play rate could therefore exceed one.

## Fix and replay

Mask play by active state before aggregation. The broken metric reproduced at
1.046560; the fixed metric is 0.952045.
Underlying stay/LT/HLT trajectories are exactly identical in shadow replay:
`True`.

## Randomized safety check

| Metric | Relative lift | p-value |
|---|---:|---:|
| stay_per_exposure | +0.0304% | 0.5532 |
| lt_rate | +0.0945% | 0.6575 |
| hlt_rate | +0.6487% | 0.2347 |
| negative_rate | -0.3164% | 0.7222 |

No business regression is significant. Training is not applicable because this
is a metric aggregation defect, not a model change.

## Decision

Pass the metric correction. Historical play-rate dashboards produced by the
broken definition must not be compared directly with the corrected series.
