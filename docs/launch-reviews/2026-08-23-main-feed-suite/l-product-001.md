# L-PRODUCT-001 — Expand personalized Feed trigger

Status: `pass_primary_metric`. Synthetic main-Feed experiment; not company production evidence.

## Change and ownership

- Category: `product`
- Owner: `feed-product`
- Hypothesis: Product eligibility expansion converts model value into overall ITT.
- Change: Increase eligible trigger coverage from 0.5% to 1.0%.
- Product dependency: Feed trigger eligibility and exposure logging
- Short-term value: stay and long-view behavior
- Long-term value: LT container, quality view, negative feedback, and return

## Training and artifacts

No retraining; trigger-only product change. Control and treatment use frozen,
versioned policy dataclasses. Common-random potential worlds act as shadow replay;
the randomized estimate uses stable user-level 50/50 assignment.

## A/B result

| Metric | Observed relative lift | p-value | Known DGP effect |
|---|---:|---:|---:|
| stay_per_exposure | +0.2707% | 2.477e-06 | +0.3583% |
| long_view_rate | +0.5874% | 0.009019 | +0.9981% |
| quality_long_view_rate | +0.7577% | 0.1628 | +0.8266% |
| negative_rate | +0.0970% | 0.9137 | -0.1971% |
| lt_value_per_exposure | +0.1401% | 0.4842 | +0.1158% |
| local_value_tree_score_per_exposure | -0.9953% | 0.5375 | +0.8698% |

Primary metric `stay_per_exposure`: +0.2707%,
p=2.477e-06. Absolute 95% confidence interval:
[+0.00552381,
+0.01339772].

## Gate and review

The primary metric cleared the randomized gate without a significant quality-view or negative-feedback regression.

The gate checks the declared primary metric, quality-view regression, LT value,
and negative feedback.
A low p-value alone is insufficient; effect size, DGP truth, product trigger rate,
and long-term guardrails remain part of the decision.

## Performance

- Control: 4,676,747 requests/s,
  170.3 MiB peak.
- Treatment: 4,781,019 requests/s,
  170.3 MiB peak.

## Next action

Ramp only through the next guarded stage and continue monitoring.
