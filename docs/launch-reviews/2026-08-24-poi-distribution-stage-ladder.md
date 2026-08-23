# L-POI-STAGE-001 — POI distribution cascade ladder

Change type: recall, coarse, fine, and mix isolation  
Scale: 1,000,000 users × 24 sequential requests × 3 seeds  
Decision: promote retarget recall only; hold or reject every later proposal  
Evidence: `reports/launches/2026-08-24-poi-distribution-stage-ladder.json`

## What was fixed / 修复内容

The prior GPU candidate builder performed an implicit Top-20 coarse truncation
inside recall. A coarse experiment therefore received an already-truncated
pool and could not own its treatment. Request traces also reconstructed the
wrong coarse pass flag and logged a default score instead of the score that
selected survivors.

旧链路在召回候选构造器内部提前做了 Top-20，导致粗排实验拿到的已经是粗排结果；
request trace 还把 48 个候选全部当作通过粗排，并保存了错误的默认分数。

The repaired invariant is:

```text
8 recall routes -> RRF/dedupe 48 -> coarse score/mask Top 20
-> frozen fine choice -> independent mix -> one exposure
```

Recall emits no coarse score. The coarse stage alone owns
`candidate_coarse_scores`, `candidate_coarse_mask`, and its budget. Fine and mix
scores remain separate in the request-level dataset.

## Adjacent Launch Reviews / 相邻上线评审

| Stage | Control -> treatment | Primary effect | Platform LT/user | Decision |
|---|---|---:|---:|---|
| Retrieval | 6 routes -> +post-search | +0.00000789 anchor CTR | -0.00588 | Reject |
| Retrieval | +post-search -> +retarget | +0.00522658 anchor CTR | +0.06947 | Pass all 3 seeds |
| Coarse | quality -> linear | +0.00140307 anchor CTR | -0.00682 | Reject |
| Coarse | linear -> DCNv2-distilled score | -0.00000473 anchor CTR | -0.00223 | Reject |
| Fine | static -> post-search | -0.00003002 Local tree | -0.00178 | Reject |
| Fine | post-search -> retarget | +0.00138421 Local tree | +0.00557 | Hold: seed instability |
| Fine | retarget -> intent ranker | +0.00086099 Local tree | +0.00112 | Hold: seed instability |
| Mix | Feed guarded -> Local expansion | +0.00430956 Local tree | -0.01126 | Reject |

`DCNv2-distilled score` is a bounded-cross scoring proxy in the tensor world,
not a claim that a trained DCNv2 artifact served this experiment. An actual
trained coarse artifact still requires frozen-candidate training, serialization
parity, and a separate Launch Review.

这些结果说明模型容量和 Local 指标都不能代替 LT。线性粗排提高 anchor CTR 仍损伤
平台 LT；扩大 Local 混排提高 Local tree，却把 LT/user 拉低。因此只有 retarget 召回
具备稳定的业务和平台共同收益。

## Scale and orchestration / 规模与编排

The runner declares 12 policy arms but materializes only 9 unique semantic GPU
worlds per seed. Worlds are content-keyed by all policy fields except the display
name and reused by adjacent comparisons. Wall time fell from 394.9 seconds to
297.1 seconds, a 24.8% reduction, while preserving common random streams.

No Dagster runtime was added. The tensor kernel remains the hot path; a future
control plane may schedule the same semantic nodes only when retries,
multi-machine partitions, or scheduled materialization become a measured need.

## Evidence boundary / 证据边界

This is a V3 synthetic Local behavior simulation. It validates stage ownership,
effect recovery, guardrails, and GPU scale. It is not TikTok production evidence
and does not estimate live business lift.

