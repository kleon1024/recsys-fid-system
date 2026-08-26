# Feed semantic v3 retrieval Launch Review

Status: Popular promoted; personalized routes held

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
