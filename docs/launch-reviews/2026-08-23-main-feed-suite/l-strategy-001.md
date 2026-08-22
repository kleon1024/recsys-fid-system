# L-STRATEGY-001 — Fatigue-aware ranking constraint

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `strategy`
- Owner: `feed-strategy`
- Hypothesis: Penalizing repeated affinity under fatigue protects HLT.
- Change: Enable a 0.12 fatigue-match penalty.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

No retraining; strategy-only change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.0302% | 0.5555 | -0.0002% |
| lt_rate | +0.0930% | 0.6628 | -0.0012% |
| hlt_rate | +0.6478% | 0.2354 | -0.0005% |
| negative_rate | -0.3164% | 0.7222 | +0.0000% |

Primary metric `hlt_rate`: +0.6478%,
p=0.2354. Absolute 95% confidence interval:
[-0.00003622,
+0.00014734].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 13,425,982 requests/s,
  91.0 MiB peak.
- Treatment: 12,949,239 requests/s,
  91.0 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
