# L-PRODUCT-001 — Expand personalized Feed trigger

Status: `pass_primary_metric`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `product`
- Owner: `feed-product`
- Hypothesis: Product eligibility expansion converts model value into overall ITT.
- Change: Increase eligible trigger coverage from 0.5% to 1.0%.
- Product dependency: Feed trigger eligibility and exposure logging
- Short-term value: stay and LT
- Long-term value: HLT, negative feedback, and return behavior

## Training and artifacts

No retraining; trigger-only product change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.3870% | 2.737e-14 | +0.3577% |
| lt_rate | +1.0966% | 2.991e-07 | +0.9971% |
| hlt_rate | +1.4428% | 0.008612 | +0.8091% |
| negative_rate | -0.5165% | 0.5611 | -0.2279% |

Primary metric `stay_per_exposure`: +0.3870%,
p=2.737e-14. Absolute 95% confidence interval:
[+0.00993257,
+0.01682360].

## Gate and review

The primary metric cleared the randomized gate without a significant HLT or negative-feedback regression.

The gate checks the declared primary metric, HLT regression, and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 12,169,116 requests/s,
  91.0 MiB peak.
- Treatment: 12,948,561 requests/s,
  91.0 MiB peak.

## Next action

Ramp only through the next guarded stage and continue monitoring.
