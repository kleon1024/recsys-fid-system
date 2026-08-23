# L-V3-MULTITASK-001 — Primitive multi-task Feed ranker

Decision: `HOLD`. The V3 stay-aligned MMoE is not the active control.

## Scope

The request-level snapshot contains 1,099,046 mature exposures from 50,000
synthetic users. Every request retains 48 recalled items, 20 coarse survivors,
the exposed item and propensity, 28 point-in-time features, a 24-event behavior
sequence, lifecycle and region slices, and delayed labels. The snapshot binds
to the same V3 model/index/feature/behavior bundle as serving.

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
directly. That exposed a second simulator defect: selected video duration rose
more than 20%, mechanically inflating stay while quality long view changed only
slightly. A duration-distribution guardrail now turns that apparent LT pass into
`HOLD` and keeps `rule_personalized_v1` active.

## Next acceptance bar

The next challenger must consume the recorded 24-event sequence, train with a
request-aware ranking objective, keep selected-duration drift within 5%, and
show nonnegative LT with no quality or negative-feedback regression. A
predictive return head or causal long-term exchange estimate is required before
retention receives nonzero online weight.

Evidence:

- `reports/training/2026-08-23-v3-model-ladder-gpu.json`
- `reports/training/2026-08-23-v3-multitask-stay-v2-gpu.json`
- `reports/launches/2026-08-23-v3-model-ladder-1m-gpu.json`
- `reports/launches/2026-08-23-v3-multitask-stay-v2-duration-audit-1m-gpu.json`

All values are synthetic simulator evidence, not company production metrics.
