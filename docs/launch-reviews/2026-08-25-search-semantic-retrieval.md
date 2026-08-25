# S-LR-001: Search semantic retrieval

This is synthetic engineering evidence, not TikTok production evidence.

本文是合成数字孪生的工程证据，不代表 TikTok 生产指标。

## Changed owner / 单一变更

Control keeps exact topic retrieval. Treatment adds one FAISS semantic Search
route over the same item corpus, Top-K budget, coarse ranker, fine ranker and
final mixer. Feed, Commerce, Local, Posting and Live policies are unchanged.

Control 保留精确 topic 召回；Treatment 只增加一条 FAISS 语义 Search 召回。同一
item corpus、Top-K、粗排、精排和混排均冻结，其他业务场景不变。

## Closed Search loop / Search 闭环

The user world now emits a query, candidate-level Search success, at most two
request-level reformulations, an explicit abandonment, and a query-linked Feed
entry after a successful Search. Session end clears pending continuation.
Hidden intent remains in the user world; the platform sees only query topic,
served candidates and emitted events.

用户世界现在包含 query、候选级 Search success、最多两次 request 级改写、显式放弃，
以及成功搜索后的 query-linked Feed 承接。Session end 会清除待续状态，隐藏兴趣不暴露
给平台。

Experiment attribution is joined by `request_id`: `QUERY` occurs before online
assignment, while the served Search impression owns the factual A/B cell. This
prevents pre-treatment intent from being mislabeled as an experiment outcome.

实验归因通过 `request_id` 完成。`QUERY` 发生在在线分桶之前，实际 Search 曝光才
拥有 factual A/B cell，避免把 treatment 前的意图错误写成实验结果。

## Preregistered gate / 预注册门禁

- Primary: request-normalized Search-success rate, clustered by user.
- Guardrails: reformulation must not significantly increase; detail rate must
  pass a 3% noninferiority margin.
- Diagnostics: post-search Feed continuation and semantic candidate support.
- Integrity: Control semantic candidates must equal zero; Treatment must have
  positive support.

## Current evidence / 当前证据

| Evidence | Result |
|---|---:|
| Search behavior/audit tests | Pass |
| Search launch smoke | Pass |
| Control semantic candidates | 0 |
| Treatment semantic support | Positive |
| Focused tests | 38 passed |
| Architecture lint | 0 errors |
| RTX 4090 factual A/B | Pending; host offline |

Decision: **implementation accepted for GPU validation; no launch decision**.
No uplift is claimed until the continuously evolving factual checkpoint runs
S-LR-001 on RTX 4090 and the preregistered confidence gates finish.

决策：**实现进入 GPU 验证，不作上线结论**。在连续 factual checkpoint 上完成
S-LR-001 和预注册置信区间门禁之前，不声明业务增量。
