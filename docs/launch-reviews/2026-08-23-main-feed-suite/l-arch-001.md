# L-ARCH-001 — Increase GPU user batch

Status: `pass_parity_and_performance`. Synthetic performance launch.

## Change

Increase device-resident user batch from 10,000 to 25,000. No model, feature,
Value Tree, or product parameter changes. Training is not applicable.

## Shadow and A/B

| Metric | Full-world distribution delta | randomized p-value |
|---|---:|---:|
| stay_per_exposure | -0.0106% | 0.8148 |
| lt_rate | -0.0811% | 0.7985 |
| hlt_rate | +0.0627% | 0.2133 |
| negative_rate | +0.2520% | 0.6778 |

The stable user A/B is neutral on every business metric. Maximum non-negative
business-distribution drift is below 0.1%; negative feedback moved 0.252%, also
without a significant randomized effect.

## Performance and cost

- Control: 4,874,540 requests/s,
  40.2 MiB.
- Treatment: 13,136,748 requests/s,
  90.6 MiB.
- Throughput lift: +169.50%.

## Decision

Pass: distribution parity holds and throughput improves. The trade-off is higher
GPU memory, so a production ramp would retain memory and P99 latency guardrails.
