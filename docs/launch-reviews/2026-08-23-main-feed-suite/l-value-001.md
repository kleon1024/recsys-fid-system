# L-VALUE-001 — Balanced engagement Value Tree

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `business_value`
- Owner: `feed-value`
- Hypothesis: More quality weight improves durable value without losing stay.
- Change: Shift affinity/quality weights from 1.0/0.45 to 0.85/0.60.
- Product dependency: none
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

No retraining; Value Tree-only change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | -0.1168% | 0.04288 | -0.0356% |
| long_view_rate | -0.5105% | 0.02257 | -0.1206% |
| quality_long_view_rate | -0.1208% | 0.8229 | -0.1117% |
| negative_rate | +0.2920% | 0.7447 | -0.0034% |
| lt_value_per_exposure | +0.0103% | 0.9588 | -0.0119% |
| local_value_tree_score_per_exposure | -2.0066% | 0.2109 | -0.2668% |

Primary metric `quality_long_view_rate`: -0.1208%,
p=0.8229. Absolute 95% confidence interval:
[-0.00010446,
+0.00008304].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,428,548 requests/s,
  170.3 MiB peak.
- Treatment: 4,526,311 requests/s,
  170.3 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
