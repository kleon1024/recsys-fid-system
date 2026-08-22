# L-LONGTERM-001 — Stronger fatigue protection

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `long_term_value`
- Owner: `feed-longterm`
- Hypothesis: More aggressive fatigue control improves long-term quality.
- Change: Increase fatigue penalty from 0.12 to 0.24.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

No retraining; constraint-only change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.0302% | 0.5559 | -0.0002% |
| lt_rate | +0.0938% | 0.66 | -0.0005% |
| hlt_rate | +0.6487% | 0.2347 | +0.0000% |
| negative_rate | -0.3164% | 0.7222 | -0.0013% |

Primary metric `hlt_rate`: +0.6487%,
p=0.2347. Absolute 95% confidence interval:
[-0.00003614,
+0.00014742].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 12,969,594 requests/s,
  91.0 MiB peak.
- Treatment: 12,768,244 requests/s,
  91.0 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
