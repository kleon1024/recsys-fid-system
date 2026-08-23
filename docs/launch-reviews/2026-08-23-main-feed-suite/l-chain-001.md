# L-CHAIN-001 — Remove cold-user UID collision score

Status: `pass_primary_metric`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `chain_diagnosis`
- Owner: `feed-consistency`
- Hypothesis: Removing unrelated hashed UID noise restores candidate quality.
- Change: Remove a 0.35 random collision term from cold-user ranking.
- Product dependency: none
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

Reused the frozen model and removed only the diagnosed chain defect. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.1387% | 0.01564 | +0.2256% |
| long_view_rate | +0.3223% | 0.1508 | +0.7414% |
| quality_long_view_rate | +0.5632% | 0.2988 | +0.6197% |
| negative_rate | +0.2186% | 0.8073 | -0.0777% |
| lt_value_per_exposure | +0.0956% | 0.633 | +0.0735% |
| local_value_tree_score_per_exposure | -1.1370% | 0.4806 | +0.7064% |

Primary metric `stay_per_exposure`: +0.1387%,
p=0.01564. Absolute 95% confidence interval:
[+0.00091831,
+0.00879106].

## Gate and review

The primary metric cleared the randomized gate without a significant quality-view or negative-feedback regression.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,420,305 requests/s,
  170.3 MiB peak.
- Treatment: 4,669,696 requests/s,
  170.3 MiB peak.

## Next action

Ramp only through the next guarded stage and continue monitoring.
