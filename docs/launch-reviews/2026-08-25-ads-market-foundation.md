# A-LR-001: Ads market foundation

This is synthetic engineering evidence, not TikTok production evidence.

本文是合成数字孪生的工程证据，不代表 TikTok 生产指标。

## Root cause / 根因

Ads previously entered Feed accidentally through organic Popular, ANN and
Retarget routes. The hidden advertiser world had bids and budgets, but the
platform had no observable budget ledger, immutable billed-spend event,
auction eligibility or final-slate load constraint. Multiple requests in one
GPU microbatch could therefore pass a per-request budget check and collectively
overspend the same advertiser.

此前广告会从 Popular、ANN、Retarget 等自然召回意外进入 Feed。隐藏 advertiser
world 虽有 bid 和 budget，平台侧却没有可观测预算账本、immutable spend、auction
eligibility 和最终列表 load 约束。同一 microbatch 的多个请求可能分别通过预算检查，
却共同透支同一 advertiser。

## Implemented boundary / 已实现边界

- Route registration no longer activates a route. The stable default excludes
  both `ads_auction` and `search_semantic` until their Launch Review promotes.
- Only `ads_auction` may contribute `ContentKind.AD`; organic and Commerce routes
  exclude Ads before Top-K selection.
- Every tick publishes observable advertiser budget snapshots and bid updates.
- Every ad impression owns exactly one immutable `AD_SPEND` event.
- Advertiser capacity is split by factual A/B assignment probability. Excess
  ads are deterministically replaced with the highest remaining organic item.
- Final slate permits at most one ad. Randomized Ads that would need constrained
  propensity recomputation fail closed.
- Feed ad clicks can mature into delayed Pixel conversions with click lineage.

## Local mechanism replay / 本地机制回放

The frozen CPU smoke uses 1,024 users, 8,000 items and 24 ticks. It is a
mechanism/reconciliation test, not an online lift estimate.

| Metric | Result |
|---|---:|
| Ad impressions / spend events | 1,099 / 1,099 |
| Clicks / delayed Pixel conversions | 116 / 10 |
| Synthetic billed revenue | 4,012.13 |
| Unbudgeted / unpriced / over-bid spend | 0 / 0 / 0 |
| Overspend / partial billing | 0 / 0 |
| Maximum ads per request | 1 |
| Focused tests | 53 passed |
| Architecture lint | 0 errors |

Raw evidence:
[`reports/launches/ads-market-smoke.json`](../../reports/launches/ads-market-smoke.json).

## A-LR-001 protocol / 实验协议

Control has no Ads route. Treatment adds only budgeted auction candidates;
corpus, organic retrieval, rankers and Feed dedup stay fixed. Primary value is
billable spend. Feed dwell has a 1% noninferiority margin and negative feedback
must not significantly increase. Pixel conversion is reported only with its
maturity watermark.

The one-slot availability test can use user randomization because only Treatment
consumes Ads inventory and capacity is probability-scaled. Later Ads ranking or
pacing experiments share a market on both arms and require market-diverted or
switchback design; orthogonal UID hashes alone are not sufficient.

Decision: **implementation accepted for GPU validation; no launch decision**.
A-LR-001 has not run on the continuously evolving RTX 4090 factual world.

决策：**实现进入 GPU 验证，不作上线结论**。A-LR-001 尚未在 RTX 4090 连续 factual
world 上运行，因此不声明收入或 LT 增量。
