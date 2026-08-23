# L-BUG-001 — Stop counting inactive users as plays

Status: `pass_metric_correction`. Synthetic measurement-chain fix.

## Root cause

The throughput engine sampled play for users whose trajectory had already ended
and counted those draws in the numerator while excluding them from exposures.
The observed play rate could therefore exceed one.

## Fix and replay

Mask play by active state before aggregation. The broken metric reproduced at
1.047324; the fixed metric is 0.952153.
Underlying stay/long-view/quality-view trajectories are exactly identical in shadow replay:
`True`.

## Randomized safety check

| Metric | Relative lift | p-value |
|---|---:|---:|
| stay_per_exposure | -0.0815% | 0.1582 |
| long_view_rate | -0.3905% | 0.08142 |
| quality_long_view_rate | -0.0159% | 0.9765 |
| negative_rate | +0.2863% | 0.7495 |
| lt_value_per_exposure | +0.0226% | 0.9101 |
| local_value_tree_score_per_exposure | -1.7764% | 0.2684 |
| anchor_click_rate | -1.2616% | 0.1941 |
| conversion_rate | -2.0830% | 0.681 |
| ad_load | n/a | 1 |
| effective_ad_load | n/a | 1 |
| ad_contribution_per_exposure | n/a | 1 |
| organic_opportunity_cost_per_exposure | n/a | 1 |
| feed_value_tree_score_per_exposure | -0.1398% | 0.7839 |
| ads_live_value_tree_score_per_exposure | n/a | 1 |
| accepted_platform_commercialization_per_exposure | n/a | 1 |
| local_commercialization_value_per_exposure | -1.8509% | 0.7454 |
| active_days_per_user | -0.0716% | 0.7992 |
| accepted_platform_commercialization_per_user | n/a | 1 |
| lt_value_per_user | -0.0870% | 0.6512 |
| coarse_feed_oracle_recall | +0.0000% | 1 |
| coarse_pass_fraction | +0.0000% | 1 |
| fine_oracle_regret_per_exposure | -0.0250% | 0.6204 |
| poi_candidate_fraction | -0.0420% | 0.02326 |
| lt_stay_per_user | -0.1178% | 0.1062 |
| lt_active_days_per_user | -0.0716% | 0.7992 |

No business regression is significant. Training is not applicable because this
is a metric aggregation defect, not a model change.

## Decision

Pass the metric correction. Historical play-rate dashboards produced by the
broken definition must not be compared directly with the corrected series.
