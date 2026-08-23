# L-V3-MULTITASK-001 — Primitive multi-task Feed ranker

Decision: `PASS` for guarded residual reranking. Direct MMoE replacement remains
`HOLD`; `mmoe_multitask_guarded_a015_t008` is the active synthetic control and
`rule_personalized_v1` is its rollback.

## Scope

The request-level snapshot contains 1,099,046 mature exposures from 50,000
synthetic users. Every request retains 48 recalled items, 20 coarse survivors,
the exposed item and propensity, 28 point-in-time features, a 24-event behavior
sequence, lifecycle and region slices, and delayed labels. The snapshot binds
to the rule logging bundle that produced it; the promoted serving bundle keeps
the same index, feature and behavior versions and records that logger as rollback.

The ranker predicts primitive outcomes: 3-second play, normalized stay seconds,
completion ratio, long view, quality long view, like, negative feedback,
next-session return, POI anchor click, and conversion. `LT total` is excluded
from training and exists only as a post-exposure A/B outcome.

## Iteration record

| Iteration | Offline result | 1M-user trajectory result | Decision |
|---|---:|---:|---|
| Long-view LR | AUC 0.5934 | LT -17.29%; long view +5.51% | Reject objective mismatch |
| Long-view XGBoost | AUC 0.5953 | LT -19.53%; long view +8.06% | Reject objective mismatch |
| Original multi-task MMoE | AUC 0.5918; candidate regret 0.0615 | LT -5.94%; stay -7.83% | Reject |
| Guarded original MMoE | rule tolerance 0.03 | LT -0.256% | Reject |
| Linear-stay MMoE V2 | candidate regret 0.0447 | LT +15.60%; stay +19.88% | Hold reward hacking |
| Guarded V2, alpha 0.05 / tolerance 0.03 | frozen artifact, rule-constrained | LT +3.24%; duration +3.18% | Pass |
| Guarded V2, alpha 0.10 / tolerance 0.05 | last accepted control | incremental LT +2.02%; duration +1.85% | Pass |
| Guarded V2, alpha 0.15 / tolerance 0.08 | last accepted control | incremental LT +2.11%; duration +2.06% | Pass and promote |

The small 20,000-user run incorrectly suggested +0.49% LT for the original
guarded model. The one-million-user run rejected it. Small runs are smoke and
power checks, not launch evidence.

## Root cause

The original ladder predicted long view while the launch metric was dominated
by stay and active-day value. It learned to improve threshold events while
reducing total stay. The first multi-task VT also used a next-session head with
AUC approximately 0.50 and a log-normalized stay target as if it were linear
seconds.

V2 removed return from online fusion and predicted `stay_seconds / 180`
directly. That exposed a second simulator defect: direct replacement raised
selected video duration more than 20%, mechanically inflating stay while quality
long view changed only slightly. The duration-distribution guardrail holds that
path. Restricting learned reranking to candidates within a bounded rule-score
tolerance keeps duration drift below 5%; three sequential million-user launches
then pass unified LT and preserve the last accepted control at every step.

## Next acceptance bar

The next challenger should consume the recorded 24-event sequence and train
with a request-aware ranking objective. It must beat the guarded V2 active
control, keep selected-duration drift within 5%, and show nonnegative LT with no
quality or negative-feedback regression. A predictive return head or causal
long-term exchange estimate is required before retention receives nonzero
online weight.

Evidence:

- `reports/training/2026-08-23-v3-model-ladder-gpu.json`
- `reports/launches/2026-08-23-v3-model-ladder-1m-gpu.json`

All values are synthetic simulator evidence, not company production metrics.
