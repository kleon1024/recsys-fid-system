# L-REALTIME-001 — Faster online interest refresh

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `realtime`
- Owner: `feed-realtime`
- Hypothesis: A fresher sequence state reacts faster without harming HLT.
- Change: Increase online interest update rate from 0.06 to 0.12.
- Product dependency: none
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

Reused the frozen control model to isolate state-freshness impact. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | -0.0749% | 0.1952 | +0.0077% |
| long_view_rate | -0.3710% | 0.09791 | +0.0267% |
| quality_long_view_rate | +0.0138% | 0.9796 | +0.0349% |
| negative_rate | +0.2796% | 0.7552 | -0.0064% |
| lt_value_per_exposure | +0.0246% | 0.902 | +0.0021% |
| local_value_tree_score_per_exposure | -1.7853% | 0.2659 | +0.0029% |

Primary metric `stay_per_exposure`: -0.0749%,
p=0.1952. Absolute 95% confidence interval:
[-0.00659808,
+0.00134731].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,579,859 requests/s,
  170.3 MiB peak.
- Treatment: 4,482,037 requests/s,
  170.3 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
