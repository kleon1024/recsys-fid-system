# L-SIMULATOR-003 — GPU request-level candidate graph

Change type: simulator architecture and experiment correctness  
Decision: accept simulator V2; hold the Local ranker launch  
Evidence boundary: synthetic RTX 4090 trajectory A/B, not production lift

## Invariant / 不变量

Every measured request must execute one auditable candidate graph: route recall,
RRF merge and deduplication, coarse truncation, fine rank, constrained mixing,
exposure, behavior transition, and mature-label trace. A high-throughput A/B may
sample bounded trace rows, but it cannot bypass the graph used by the policy.

每个被度量的请求必须执行同一张可审计候选图。GPU 扩量可以只保留有界 trace，不能
再用 20 个随机候选绕过召回、粗排和混排。

## Removed path / 删除的旧路径

The former tensor engine created 20 dense random candidates, forced a
post-search match for every user, and reported oracle coarse pass-through of
one. That path is removed from the accepted simulator. Search is now an
exogenous sparse event with a TTL; retarget eligibility is frozen before the
treatment period.

旧引擎的全员 post-search、固定 20 候选和粗排通过率 1 已被删除。当前 trigger cohort
在 treatment 前冻结，避免用处理后的行为选择实验人群。

## 1M GPU evidence / 百万级证据

The control processed 12,005,509 measured requests in 4.46 seconds, or 2.69M
requests/s, with 2.67GB peak GPU memory. Eight recall routes produce about 48.5
unique candidates per request before 48-to-20 coarse truncation. Stage
attribution is no longer degenerate: 19.08% recall miss, 0.016% coarse miss,
19.46% fine-rank miss, and 61.44% served audit oracle. A separate multi-queue
smoke produced a nonzero 3.71% mix-rank miss.

The post-search trigger rate is 4.001%. The Local intent treatment has unified
LT -0.0172% overall with an absolute 95% interval of [-0.0242, 0.0209], and
triggered LT +0.190% with an interval crossing zero. The launch is held. This is
the intended result: simulator repair must be able to reject a previously
plausible feature or policy.

Runtime tuning is isolated from behavior semantics. Counter-based random draws
are keyed by `user_id + request_step + stream + seed`; stage counts are exactly
equal at 25K, 100K, 200K, and 400K batch sizes. The maximum checked metric delta
at the selected 200K batch is 2.2e-8. Throughput improves from 1.36M to 2.88M
requests/s while peak memory grows from 376MB to 2.67GB. A 400K batch consumes
5.30GB but is slower, so 200K is the measured 4090 default.

百万级结果不再把每层写成全通过。Local intent treatment 的整体与 triggered LT
区间均跨零，因此 Hold，不能因为 Local 指标或模型复杂度直接上线。

## Request trace and acceptance / 请求级 trace 与验收

Each world preserves eight sampled requests and 384 candidate rows. Every row
contains route provenance, recall/coarse/fine/mix scores and ranks, stage pass,
exposure, audit-oracle identity, and mature labels only for the exposed item.
The trace is content-hashed. Unit tests require one exposure and one mature
label row per request.

Acceptance bars:

- stage-attribution counts close exactly to measured requests;
- search and retarget routes are sparse and pre-treatment cohorts match across worlds;
- coarse pass fraction is below one and route budgets are explicit;
- request traces close with one exposure and no unexposed false labels;
- architecture lint, focused tests, repository acceptance, and RTX 4090 scale pass.

## Evidence

- `reports/launches/2026-08-23-feed-digital-twin-v2-1m-gpu.json`
- `reports/benchmarks/2026-08-23-tensor-batch-scale-1m-gpu.json`
- `fid_lab/feed_loop/scale/graph/candidate.py`
- `fid_lab/feed_loop/scale/graph/trace.py`
- `fid_lab/feed_loop/scale/experiment/trigger.py`
