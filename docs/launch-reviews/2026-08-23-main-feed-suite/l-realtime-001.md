# L-REALTIME-001 — Faster online interest refresh

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `realtime`
- Owner: `feed-realtime`
- Hypothesis: A fresher sequence state reacts faster without harming HLT.
- Change: Increase online interest update rate from 0.06 to 0.12.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

Reused the frozen control model to isolate state-freshness impact. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.0344% | 0.5024 | +0.0043% |
| lt_rate | +0.1096% | 0.6074 | +0.0171% |
| hlt_rate | +0.6650% | 0.2232 | +0.0132% |
| negative_rate | -0.3164% | 0.7222 | +0.0012% |

Primary metric `stay_per_exposure`: +0.0344%,
p=0.5024. Absolute 95% confidence interval:
[-0.00229153,
+0.00467535].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 12,831,826 requests/s,
  91.0 MiB peak.
- Treatment: 12,596,490 requests/s,
  91.0 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
