# L-CHAIN-001 — Remove cold-user UID collision score

Status: `pass_primary_metric`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `chain_diagnosis`
- Owner: `feed-consistency`
- Hypothesis: Removing unrelated hashed UID noise restores candidate quality.
- Change: Remove a 0.35 random collision term from cold-user ranking.
- Product dependency: none
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

Reused the frozen model and removed only the diagnosed chain defect. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.2385% | 2.668e-06 | +0.2085% |
| lt_rate | +0.8109% | 0.0001443 | +0.6959% |
| hlt_rate | +1.1839% | 0.03077 | +0.5480% |
| negative_rate | -0.4375% | 0.6227 | -0.1229% |

Primary metric `stay_per_exposure`: +0.2385%,
p=2.668e-06. Absolute 95% confidence interval:
[+0.00480978,
+0.01170397].

## Gate and review

The primary metric cleared the randomized gate without a significant HLT or negative-feedback regression.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 11,967,160 requests/s,
  91.0 MiB peak.
- Treatment: 13,106,851 requests/s,
  91.0 MiB peak.

## Next action

Ramp only through the next guarded stage and continue monitoring.
