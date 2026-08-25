# v4 World Checkpoint and Retrieval Mainline Review

Decision: `PASS` for durable world checkpointing and `PROMOTE` for Popular.
Cold-start is `STOP_INCONCLUSIVE`; recent ANN remains `HOLD` after its first
window. These are synthetic-world decisions and do not claim TikTok production
lift.

## What changed

The simulator now saves a content-addressed ecosystem checkpoint rather than a
model-only snapshot. It covers hidden users, creator and market supply, trends,
delayed outcomes, the observable projection, graph and ANN state, append-only
events, the factual experiment plan and learning cursors. Restore fails closed
on catalog, runtime, feature, model or code mismatch. An explicit code migration
is allowed only when the runtime contract still matches and is recorded in the
next review.

Request traces now retain the complete recall universe, the actual coarse and
fine scorer inputs, stage admission, exact randomized propensity and factual
exposures. Unexposed items have masked behavior labels. Deterministic traffic no
longer pretends to provide counterfactual OPE support.

The Feed route registry adds separate `random` and `popular` authorities over
the same 30-day main corpus. Every retrieval LR freezes corpus, Top-K, coarse,
fine and mixing. The runner can execute one window, checkpoint, stop and resume
from the next tick.

## Restore evidence

After tick 1, two fresh runtimes restored from the same checkpoint. Their next
tick entry events, response events, request candidate traces, point-in-time
context, hidden user state, supply state and platform projection were tensor
equal. The full RTX 4090 repository gate passed with 310 tests and three
subtests; architecture lint passed with zero errors.

## Sequential Launch Reviews

| LR | Change | Evidence | Decision |
|---|---|---|---|
| R-LR-001 | Random → Random + Popular | 1,555/1,596 triggered users; stay +27.19%, absolute CI95 +8.35s to +13.23s; long-view +35.67%; negative -7.42% | Promote |
| R-LR-002 | Add cold-start | Three cumulative windows, 824/825 users; stay +3.80%, CI95 -3.36s to +8.10s; only 1.96% of route candidates exposed | Stop inconclusive |
| R-LR-003 | Add recent ANN | First window, 319/347 users; stay -4.25%, CI95 -7.82s to +3.67s; 24.66% exposed | Hold |

The large Popular lift is expected only for a deliberately weak random
bootstrap baseline. It is not a mature-system uplift. Cold-start demonstrates
why route Recall alone is insufficient: 100% entered merged recall, 57.14%
passed coarse, 9.85% reached fine and 1.96% was exposed. The fixed downstream
ranker removed most of the incremental supply.

## Reliability finding

The first resumed cold-start run exposed a state-machine bug: a `HOLD` result
advanced to the next route. Checkpoint
`aaecb8993916c2ba4d8058af88497a7c881743e13ab9c61d4c321cc86aba94a5`
is invalidated. The accepted runner retains a pending LR cursor, accumulates
events and cascade counts across windows and stops inconclusive after three
pre-registered windows. This is a correctness recovery, not an algorithm lift.

The compact evidence artifact is
[`reports/launches/2026-08-25-v4-world-checkpoint-retrieval-mainline.json`](../../reports/launches/2026-08-25-v4-world-checkpoint-retrieval-mainline.json).

## Current mainline

The active retrieval baseline remains `random + popular`. Cold-start was not
added. Recent ANN continues from checkpoint
`87f1fa6990584f2db57792d13c81a998ec4ec61dfc98e1f95991f2d54bb13d0c`.
Future ANN windows must use the same experiment assignment and accumulated
analysis start. No burn-in or earlier LR is replayed.
