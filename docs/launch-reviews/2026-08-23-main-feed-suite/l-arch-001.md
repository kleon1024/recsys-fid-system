# L-ARCH-001 — Increase GPU user batch

Status: `pass_parity_and_performance`. Synthetic performance launch.

## Change

Increase device-resident user batch from 10,000 to 25,000. No model, feature,
Value Tree, or product parameter changes. Training is not applicable.

## Shadow and A/B

| Metric | Full-world distribution delta | randomized p-value |
|---|---:|---:|
| stay_per_exposure | -0.0214% | 0.4183 |
| long_view_rate | -0.0190% | 0.2115 |
| quality_long_view_rate | +0.0126% | 0.7513 |
| lt_value_per_exposure | +0.0596% | 0.2458 |
| local_value_tree_score_per_exposure | +0.8882% | 0.06208 |
| negative_rate | -0.5985% | 0.2863 |

The stable user A/B is neutral on every business metric. Maximum business drift
excluding negative feedback is 0.888%; negative
feedback moved -0.599%, also without a significant
randomized effect.

## Performance and cost

- Control: 1,755,450 requests/s,
  82.1 MiB.
- Treatment: 4,504,427 requests/s,
  168.3 MiB.
- Throughput lift: +156.60%.

## Decision

Pass: distribution parity holds and throughput improves. The trade-off is higher
GPU memory, so a production ramp would retain memory and P99 latency guardrails.
