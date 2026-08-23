# L-FEATURE-001 — Lower-noise sequence interest feature

Status: `hold_underpowered_or_neutral`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `feature`
- Owner: `feed-feature`
- Hypothesis: A better point-in-time interest estimate improves scoped ranking.
- Change: Reduce observable interest noise from 0.12 to 0.08.
- Product dependency: none
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

Reused the frozen control model to isolate feature-quality impact. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | -0.0461% | 0.4258 | +0.0369% |
| long_view_rate | -0.2641% | 0.2392 | +0.1282% |
| quality_long_view_rate | +0.0897% | 0.868 | +0.1127% |
| negative_rate | +0.2675% | 0.7654 | -0.0181% |
| lt_value_per_exposure | +0.0337% | 0.8663 | +0.0123% |
| local_value_tree_score_per_exposure | -1.7153% | 0.2853 | +0.0526% |

Primary metric `stay_per_exposure`: -0.0461%,
p=0.4258. Absolute 95% confidence interval:
[-0.00559360,
+0.00236124].

## Gate and review

The evidence does not justify rollout. Preserve control and revise or stop.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,409,003 requests/s,
  168.3 MiB peak.
- Treatment: 4,512,031 requests/s,
  170.3 MiB peak.

## Next action

Do not ramp. Increase evidence only if the hypothesis remains economically meaningful; otherwise close the launch.
