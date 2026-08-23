# Main Feed Independent Launch Reviews

All launches use the same immutable evidence JSON, common-random shadow, stable user-level A/B, and multi-metric gate.

| Launch | Category | Decision | Primary lift |
|---|---|---|---:|
| [L-FEATURE-001](l-feature-001.md) | feature | hold_underpowered_or_neutral | -0.0461% |
| [L-STRATEGY-001](l-strategy-001.md) | strategy | hold_underpowered_or_neutral | -0.0159% |
| [L-REALTIME-001](l-realtime-001.md) | realtime | hold_underpowered_or_neutral | -0.0749% |
| [L-PRODUCT-001](l-product-001.md) | product | pass_primary_metric | +0.2707% |
| [L-VALUE-001](l-value-001.md) | business_value | hold_underpowered_or_neutral | -0.1208% |
| [L-LONGTERM-001](l-longterm-001.md) | long_term_value | hold_underpowered_or_neutral | -0.0178% |
| [L-CHAIN-001](l-chain-001.md) | chain_diagnosis | pass_primary_metric | +0.1387% |
| [L-MODEL-001](l-model-001.md) | fine_rank_model | reject all advanced candidates | LR remains authority |
| [L-ONLINE-001](l-online-001.md) | streaming_training | reject_negative_feedback | CPU correctness only; GPU scale required |
| [L-LOCAL-SUPPLY-001](l-local-supply-001.md) | posting_supply | pass to switchback only | 2 → 5 published videos; interference remains |
| [L-LOCAL-VALUE-001](l-local-value-001.md) | local_value_tree | hold_quality_long_view_risk | No known Local or LT increment |
| [L-LOCAL-LT-002](l-local-lt-002.md) | local_service_lt | hold | Local gains do not enter LT |
| [L-LOCAL-SUPPLY-002](l-local-supply-002.md) | supply_switchback | hold_estimator_miss | 95.4% calibrated coverage |
| [L-QUEUE-LT-001](l-queue-lt-001.md) | multi_queue | calibration_review | Ads exchange rate is synthetic |
| [L-LOCAL-SCALE-003](l-local-scale-003.md) | local_model_scale | hold / reject | 52.6M-user MDE requirement |
| [L-COARSE-001](l-coarse-001.md) | coarse_rank | pass weak baseline; hold mature changes | 99.9% oracle pass-through |
| [L-LOCAL-REVERSE-004](l-local-reverse-004.md) | reverse_holdout | retain staged launch and holdout | LT +0.282%, retention component noisy |
| [L-TENSOR-003](l-tensor-003.md) | published_model | pass_unified_lt_nonnegative | LT +0.265%, 95% CI lower bound +0.00393 |
| [L-SIMULATOR-003](l-simulator-003.md) | GPU candidate graph | accept simulator; hold Local launch | 2.69M req/s; LT -0.0172%, CI crosses zero |
| [F-LR-001](l-feature-lr-001.md) | sequence features | hold_unified_lt_uncertain | LT -0.189%, CI crosses zero |
| [F-LR-002](l-feature-lr-002.md) | realtime features | pass_unified_lt_nonnegative | LT +1.204%, promoted |
| [F-LR-003](l-feature-lr-003.md) | Local context | pass_unified_lt_nonnegative | LT +3.215%, promoted |
| [F-LR-004](l-feature-lr-004.md) | hash/category/duration | reject_unified_lt_negative | LT -1.024%, active unchanged |
| [F-LR-005](l-feature-lr-005.md) | duration | reject_unified_lt_negative | LT -1.329%, active unchanged |
| [F-LR-006](l-feature-lr-006.md) | identity hash | hold_unified_lt_uncertain | LT +0.004%, CI crosses zero |
| [F-LR-007](l-feature-lr-007.md) | category hash | pass_unified_lt_nonnegative | LT +0.366%, promoted |
| [F-LR-008](l-feature-lr-008.md) | remove POI indicator | hold_unified_lt_uncertain | LT +0.069%, CI crosses zero |
| [F-LR-009](l-feature-lr-009.md) | remove post-search | hold_unified_lt_uncertain | LT +0.043%, CI crosses zero |
| [F-LR-010](l-feature-lr-010.md) | remove retarget | hold_unified_lt_uncertain | LT +0.004%, CI crosses zero |
| [F-LR-011](l-feature-lr-011.md) | remove quality/inventory | hold_unified_lt_uncertain | LT +0.100%, CI crosses zero |
| [F-LR-012](l-feature-lr-012.md) | remove geo/interest | reject_unified_lt_negative | LT -3.073%, active unchanged |
| [F-LR-013](l-feature-lr-013.md) | triggered post-search ablation | hold_unified_lt_uncertain | 4.001% trigger; overall LT +0.0341% |
| [F-LR-014](l-feature-lr-014.md) | triggered retarget ablation | hold_unified_lt_uncertain | 3.143% trigger; overall LT +0.0186% |
