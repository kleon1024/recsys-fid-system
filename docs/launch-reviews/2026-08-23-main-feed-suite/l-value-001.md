# L-VALUE-001 — Balanced engagement Value Tree

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `business_value`
- Owner: `feed-value`
- Hypothesis: More quality weight improves durable value without losing stay.
- Change: Shift affinity/quality weights from 1.0/0.45 to 0.85/0.60.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

No retraining; Value Tree-only change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | -0.0033% | 0.9482 | -0.0333% |
| lt_rate | -0.0161% | 0.9399 | -0.1112% |
| hlt_rate | +0.6020% | 0.2701 | -0.0743% |
| negative_rate | -0.2962% | 0.7393 | +0.0138% |

Primary metric `hlt_rate`: +0.6020%,
p=0.2701. Absolute 95% confidence interval:
[-0.00004013,
+0.00014340].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 12,596,333 requests/s,
  91.0 MiB peak.
- Treatment: 12,874,176 requests/s,
  91.0 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
