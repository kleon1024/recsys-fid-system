# v4 World Checkpoint and Retrieval Mainline Review

Decision: `PASS` for durable world checkpointing and `PROMOTE` for Popular.
Cold-start and recent Graph are `STOP_INCONCLUSIVE`; recent ANN is `REJECT`.
These are synthetic-world decisions and do not claim TikTok production lift.

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

The checkpoint root now owns a locked branch registry. `main` is the sole
factual training authority. `shadow`, `replay` and `counterfactual` branches may
fork any checkpoint on their source lineage, but their events cannot become
later training data. Branch heads advance by compare-and-swap and only to a
direct child, so concurrent or cross-branch updates fail closed. When the
runner is restarted, it resolves the registered head automatically.

## Restore evidence

After tick 1, two fresh runtimes restored from the same checkpoint. Their next
tick entry events, response events, request candidate traces, point-in-time
context, hidden user state, supply state and platform projection were tensor
equal. The full RTX 4090 repository gate passed with 314 tests and three
subtests; architecture lint passed with zero errors.

## Sequential Launch Reviews

| LR | Change | Evidence | Decision |
|---|---|---|---|
| R-LR-001 | Random → Random + Popular | 1,555/1,596 triggered users; stay +27.19%, absolute CI95 +8.35s to +13.23s; long-view +35.67%; negative -7.42% | Promote |
| R-LR-002 | Add cold-start | Three cumulative windows, 824/825 users; stay +3.80%, CI95 -3.36s to +8.10s; only 1.96% of route candidates exposed | Stop inconclusive |
| R-LR-003 | Add recent ANN | Two cumulative windows, 537/589 users; stay -11.72%, CI95 -14.14s to -1.95s; 26.10% exposed | Reject |
| R-LR-004 | Add recent Graph | Three cumulative windows, 837/809 users; stay -3.70%, CI95 -8.96s to +3.04s; 29.33% exposed | Stop inconclusive |
| R-LR-005 | Add Following | Three cumulative windows, 814/784 users; stay +7.01%, CI95 -0.94s to +10.98s; long-view +9.64% with positive CI; 5.59% exposed | Stop inconclusive |
| R-LR-006 | Add Hot | Three cumulative windows, 885/819 users; stay +1.27%, CI95 -5.11s to +7.20s; 27.14% exposed | Stop inconclusive |
| R-LR-007 | Add Evergreen | Three cumulative windows, 594/605 users; stay -4.69%, CI95 -11.50s to +3.31s; only 3.75% exposed | Stop inconclusive |

The large Popular lift is expected only for a deliberately weak random
bootstrap baseline. It is not a mature-system uplift. Cold-start demonstrates
why route Recall alone is insufficient: 100% entered merged recall, 57.14%
passed coarse, 9.85% reached fine and 1.96% was exposed. The fixed downstream
ranker removed most of the incremental supply.

Recent ANN is not a downstream attrition failure: all candidates entered
recall and coarse, 58.29% reached fine and 26.10% were exposed. Its candidate
semantics reduced stay and long-view significantly, so it was rejected. Graph
also reached the final slate at a meaningful rate, but three windows did not
establish either benefit or harm.

Following improved the auxiliary long-view metric but did not establish the
pre-registered stay outcome, so it was not promoted. Hot reached the slate but
did not move behavior. Evergreen was mostly removed by the fixed downstream
ranker. These distinct outcomes separate candidate quality from cascade
attrition instead of calling every retrieval miss a model failure.

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

The handcrafted retrieval ladder is complete. The active baseline remains
`random + popular`; no later route passed the stay gate. The factual head is
checkpoint
`e31caeb662860379bf40e9d597069163d2023eb2cdf462bc31005b039eaeca1e`
at tick 147. The next retrieval work is a learned Two-Tower challenger, not
another manual route. It must start from this registered head without replaying
burn-in or any earlier LR.
