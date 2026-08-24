# L-GOVERNANCE-001 — Feed content governance V5 / Feed 内容治理 V5

Decision: `continue_powered_online_experiment`. The candidate is not declared
fully passed and does not replace the active simulator control.

结论：进入下一轮预注册扩量实验，不能写成“已上线”。150 万用户随机 A/B 中平台 LT
显著为正，但 stay 非劣下界比预注册门槛低 `0.0014` 秒，因此门禁保持失败。

## What changed / 改了什么

The accepted control is unchanged: Feed MMoE, Linear Local coarse rank, MMoE
Local fine rank, and the same Value Tree. The treatment adds only:

- predicted integrity-risk threshold `0.75`;
- repeated-cluster penalty `0.02`;
- repeated-author penalty `0.01`.

POI pacing and creator boost are disabled. The hidden V5 behavior world adds a
latent-trait-conditioned response to repeated clusters and authors. Novelty and
patience remain hidden from serving. Terminal next-day retention is matured once
per user after the request trajectory.

## Bugs found before the experiment / 实验前修复的问题

1. A governance fallback could select an item already rejected by coarse rank,
   then a second mask made `argmax` choose an arbitrary position. Fallback is
   now restricted to upstream-eligible candidates.
2. Candidate metrics continued accumulating after a user became inactive,
   producing an impossible control eligibility fraction above one. Ad load,
   duration, POI, risk, and governance metrics now share the valid-exposure
   denominator.
3. Governance reused the Local anchor launch gate. It now has its own
   observable-only gates and excludes hidden oracle quality.
4. The simulator treated an immediate within-session return as an active day.
   Better stay therefore created fewer label opportunities and mechanically
   reduced LT. Session returns remain runtime state; next-day retention now has
   one equal terminal observation opportunity per user.

## Independent dose screen / 独立剂量筛选

All 50k screens used the same corpus, candidates, models, seed, and eight-step
trajectory.

| Candidate | Candidate retention | Paired LT | Result |
|---|---:|---:|---|
| Risk threshold 0.55 | 92.29% | -0.01057 | reject; relevance loss |
| Risk threshold 0.65 | 98.25% | -0.00199 | reject |
| Risk threshold 0.75 | 99.80% | +0.00038 | advance |
| Diversity 0.02 / 0.01 | 100.00% | +0.00111 | advance for more power |
| Diversity 0.05 / 0.02 | 100.00% | -0.00093 | reject |
| POI cap 3, minimum gap 1 | 95.23% | -0.00672 | reject; POI exposure -4.98pp |

This screen prevents a common failure: tuning a risk or business proxy until it
looks cleaner while purchasing the result with relevance or Local traffic.

## Powered randomized A/B / 扩量随机实验

The final run contains 1.5 million users, disjoint stable user buckets, eight
request steps, 200k GPU batches, common model artifacts, and no warmup period.
Paired replay is the sensitive shadow diagnostic; disjoint A/B is the online
decision authority.

| Metric | Control | Disjoint A/B lift | 95% CI of absolute effect |
|---|---:|---:|---:|
| Platform LT per user | 3.25494 | +0.3006% | `[+0.00058, +0.01899]` |
| Next-day active | 0.44074 | +0.3757% | `[+0.00007, +0.00325]` |
| Stay per exposure | 8.20159s | +0.1878% | `[-0.02144s, +0.05224s]` |
| Long view | 0.05842 | +0.6228% | `[+0.00004, +0.00069]` |
| Quality long view | 0.03936 | +0.3672% | `[-0.00012, +0.00040]` |
| Predicted integrity risk | 0.34699 | -0.2460% | `[-0.00103, -0.00068]` |
| Consecutive duplicate | 0.01206 | -46.24% | `[-0.00577, -0.00538]` |
| Selected POI rate | 0.29588 | -0.9344% | `[-0.00335, -0.00218]` |

The LT, risk, duplicate, quality-view, and negative-feedback gates pass. Stay
does not: its lower bound is `-0.02144s`, below the `-0.02000s` noninferiority
margin. The report therefore remains `continue_powered_online_experiment`.
Changing the threshold after seeing the result would be p-hacking.

The RTX 4090 processed about 75.7k requests/s per world with 9.11 GB peak GPU
memory. Control and treatment together took about 306 seconds. These are
simulator throughput numbers, not production latency or cost.

## Product and supply boundary / 产品与供给边界

The immediate product lever is conservative distribution governance, not a new
content-understanding model. It can ship behind parameters quickly, but only as
an experiment. Content understanding and originality prediction need labels and
a model release. Creator exploration needs the publication and creator-retention
loop plus a creator-side switchback; it is disabled here because the unified
consumer run has no supply treatment.

Raw hash-bound evidence:
[`2026-08-24-content-governance-v5-1p5m.json`](../../reports/launches/2026-08-24-content-governance-v5-1p5m.json).

Evidence boundary: all results are synthetic. They demonstrate engineering,
causal accounting, and launch discipline; they do not claim TikTok, X, Binance,
or any employer's production uplift.
