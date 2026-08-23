# L-STRATEGY-001 — Fatigue-aware ranking constraint

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `strategy`
- Owner: `feed-strategy`
- Hypothesis: Penalizing repeated affinity under fatigue protects HLT.
- Change: Enable a 0.12 fatigue-match penalty.
- Product dependency: none
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

No retraining; strategy-only change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | -0.0818% | 0.1568 | -0.0003% |
| long_view_rate | -0.3903% | 0.08156 | -0.0004% |
| quality_long_view_rate | -0.0159% | 0.9765 | +0.0000% |
| negative_rate | +0.2863% | 0.7495 | +0.0000% |
| lt_value_per_exposure | +0.0225% | 0.9105 | -0.0001% |
| local_value_tree_score_per_exposure | -1.7797% | 0.2675 | -0.0041% |

Primary metric `quality_long_view_rate`: -0.0159%,
p=0.9765. Absolute 95% confidence interval:
[-0.00009519,
+0.00009237].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,537,748 requests/s,
  170.3 MiB peak.
- Treatment: 4,352,499 requests/s,
  170.3 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
