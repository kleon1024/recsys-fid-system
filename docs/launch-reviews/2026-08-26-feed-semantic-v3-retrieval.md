# Feed semantic v3 retrieval Launch Review

Status: Popular promoted; personalized routes held

## Initial learned fine ranker

The first Long-view-only Dense LR proved that observable learning could beat
RRF, but its three A/B seeds all moved Negative upward. One of those seeds also
failed A/A, while the runner still allowed Promote. That result is superseded.

The corrected runner fails closed on A/A and serves `feed-engagement-v1`, a
prior-corrected multi-task value tree with positive Play-3S, Long View,
Complete, Like, Comment, Share and Follow value and explicit negative-feedback
penalty. Seed 1809 passed A/A. Against the frozen Random+Popular RRF baseline it
produced:

| Metric | Relative delta | Absolute 95% CI |
| --- | ---: | ---: |
| Stay | +9.60% | [+3.60s, +12.39s] |
| Play-3S | +7.10% | [+0.22, +1.58] |
| Long View | +12.13% | [+0.35, +1.00] |
| Complete | +14.58% | [+0.23, +0.59] |
| Follow | +10.06% | [+0.03, +0.20] |
| Negative | -0.98% | [-0.080, +0.057] |

The decision is Promote for the requested single-seed protocol. The artifact is
content-bound at SHA-256
`d1477ed48c667feecc6e724856269b262b587e77ca8de22b2750fa76744757d0`.
Offline long-view AUC is `0.5893` and request GAUC is `0.5937`.

This does not erase the support warning: 296,368 of 1,247,007 training rows had
mature labels, and none came from randomized exposure. Online A/B establishes
the factual launch result for this policy; deterministic offline evaluation
alone cannot authorize a different policy.

## Changed world authority

The previous 64-dimensional Gaussian residuals were not normalized before
mixing. Their norm scaled with vector width and overwhelmed public content,
topic and user-interest directions. `feed-semantic-v3` normalizes every
semantic residual before mixing, versions the semantic authority, and makes
persistent need episodes affect response utility. Hidden state remains private.

Independent structural checks now report public-to-private content cosine
`0.923`, primary-interest cosine `0.776`, unrelated-topic cosine `0.001`, and
same-topic public cosine `0.843`. The old logged exposed-slate semantic and
sequence pair AUCs were `0.502` and `0.499`; the v3 run reports `0.534` and
`0.528`.

## Fixed protocol

All reviews used 10,000 users, 100,000 items, 512 topics, 64-dimensional
embeddings, 112 burn-in ticks, 32 A/A ticks, 64 A/B ticks, 50/50 triggered
traffic and the same formula coarse/fine/mix policy. The GPU jobs ran on the
RTX 4090. Each change owned retrieval only.

## Decisions

| Review | Control | Treatment | Stay | Long view | Negative | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Popular | Random | Random + Popular | +18.98% | +30.05% | -1.11% | Promote |
| Interest Popular | Random + Popular | + Interest Popular | -3.92% | -3.03% | -5.71% | Hold |
| Recent ANN | Random + Popular | + Recent ANN | -4.73% | -5.98% | -4.47% | Hold |

Popular had a positive stay confidence interval of `[+8.69s, +16.61s]` and
passed the negative-feedback guardrail. Interest Popular had no stay lift and
significantly reduced Follow by 10.78%. Recent ANN significantly reduced
Play-3S, Long View, Like and Comment.

## Stage diagnosis

The v3 Recent ANN route is no longer random: its same-request strong-outcome
rate is `1.57%`, versus Random at `0.36%`. It nevertheless trails Popular at
`7.98%`. Recent ANN average public quality is `0.572`, versus Popular at
`0.628`. The fixed formula cascade admits 58.9% of ANN candidates to fine rank
and exposes 14.7%, so incremental semantic coverage displaces higher-value
Popular candidates.

The next Launch Review must therefore change ranking ownership, not the User
World or another retrieval route. Train the initial observable ranker on
request-level candidates and mature multi-objective labels, calibrate scores
across route provenance, and require Popular preservation plus candidate-level
value lift before A/B. Retrieval remains frozen at Random + Popular until that
gate passes.

## Evidence

- `reports/launches/2026-08-26-feed-semantic-v3-popular-lr.json`
- `reports/launches/2026-08-26-feed-semantic-v3-interest-popular-lr.json`
- `reports/launches/2026-08-26-feed-semantic-v3-recent-ann-lr.json`
- `reports/launches/2026-08-26-feed-semantic-v3-vt-lr.json`

## Incremental Stay target review

The next ranker inherited all 20 task heads from the accepted value-tree
checkpoint, initialized only the new bounded `stay_value` target, and trained
for eight epochs on 1,642,124 newly returned candidate rows plus 371,321
historical replay rows. Feature-normalization changes were algebraically folded
into the inherited LR weights so the warm-start scorer preserved the accepted
model before its first optimizer step. Per-launch A/A was removed; platform
A/A is a separate periodic experiment-health responsibility.

Against the accepted value-tree ranker, Stay increased 1.64%, Long View 3.99%
and Negative decreased 3.47%. The Stay confidence interval was
`[-3.48s, +7.16s]`, while the experiment could only detect a 6.77% relative
change. The decision is Hold for insufficient power, not Reject. The accepted
ranker remains active.

## Recent ANN on the accepted ranker

Recent ANN was then tested as the only treatment change on top of Random,
Popular and the accepted value-tree ranker. It hit 15,720 treatment requests,
produced 32,858 candidates, and 53.5% of its candidates reached exposure, so
coverage was not the failure. Stay changed -2.48%, Long View -2.77%, and
Negative -3.64%; confidence intervals crossed zero. The route is not promoted.
Its next iteration must improve candidate value or lower the RRF weight rather
than merely increase retrieval volume.

- `reports/launches/2026-08-26-feed-semantic-v3-stay-vt-lr.json`
- `reports/launches/2026-08-26-feed-semantic-v3-vt-recent-ann-lr.json`
