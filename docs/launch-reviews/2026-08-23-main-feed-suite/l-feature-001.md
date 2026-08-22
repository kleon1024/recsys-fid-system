# L-FEATURE-001 — Lower-noise sequence interest feature

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `feature`
- Owner: `feed-feature`
- Hypothesis: A better point-in-time interest estimate improves scoped ranking.
- Change: Reduce observable interest noise from 0.12 to 0.08.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

Reused the frozen control model to isolate feature-quality impact. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.0660% | 0.1983 | +0.0359% |
| lt_rate | +0.2109% | 0.3232 | +0.1248% |
| hlt_rate | +0.7509% | 0.1691 | +0.1041% |
| negative_rate | -0.3164% | 0.7222 | -0.0042% |

Primary metric `stay_per_exposure`: +0.0660%,
p=0.1983. Absolute 95% confidence interval:
[-0.00119905,
+0.00577959].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 10,644,650 requests/s,
  90.6 MiB peak.
- Treatment: 12,992,438 requests/s,
  91.0 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
