# Digital Twin P1 Feed Loop Launch Review — 2026-08-25

Decision: accept P1 as the Feed supply-consumption and lifecycle foundation.
This accepts simulator mechanics and replay evidence only. It does not promote
a recommendation model or claim real-user lift.

## Changed authority

P1 replaces the former route aliases and reserved-item publication behavior:

- Posting selection emits an intent over a POI, product or prompt. Successful
  publication allocates one immutable `post_id`; the source candidate remains
  separately traceable.
- The observable platform owns reserved, cold-start, recent, hot, evergreen and
  expired lifecycle states. Recent membership is bounded to 30 days.
- Core Feed owns Recent ANN, Recent Graph/I2I, Following, Cold-start, Hot and
  Evergreen. Local, Posting, Commerce, Live, Search and Retarget are separate
  business routes.
- Publish capacity, cooldown and creator-exit failures are typed. Moderation,
  creator deletion and creator exit remove future supply and force ANN refresh.
- Route and candidate logs persist request-time lifecycle. They never reconstruct
  lifecycle from the post-response projection.

## Failures found during the review

The first analytical implementation read lifecycle after response ingestion.
One request could make an item hot, causing its earlier Recent route admission
to look invalid. The ClickHouse/DuckDB admission query found 120 false
violations. Request-time lifecycle is now part of `RequestCandidateTrace`.

The first Cold-start route used one global Top-K for every user. In the 100K-user
run, 32,363 posts were published but only 18 distinct posts reached a future
Feed candidate set. This was a severe supply-feedback starvation bug, not a
model-quality result. Request-level deterministic rotating exploration raised
future candidate coverage to 31,646 posts, or 97.78%.

## Acceptance evidence

| Gate | Result |
|---|---|
| Repository acceptance | 202 legacy tests plus 49 v4 pytest tests pass on the RTX environment |
| Architecture | zero errors; the declared non-Dagster asset warning remains |
| ClickHouse | all diagnostics execute on 25.8.32.4; lifecycle admission violations = 0 |
| Scale | 100K users, 2M items, two ticks, 55,997,705 durable rows |
| Runtime | 106.06 s total; 7.52 GiB peak CUDA; 5.15 GiB peak RSS |
| Replay | two content-bound partitions with exact dataset and partition hashes |
| Feed loop | publish→future candidate lineage exists; 97.78% next-tick new-post coverage |
| Failure semantics | capacity, cooldown, creator exit, moderation and deletion are executable |

The content-bound reports are
[`2026-08-25-digital-twin-p1-feed-loop-100k-2m-4090.json`](../../reports/benchmarks/2026-08-25-digital-twin-p1-feed-loop-100k-2m-4090.json)
and
[`2026-08-25-digital-twin-p1-multitick-clickhouse.json`](../../reports/benchmarks/2026-08-25-digital-twin-p1-multitick-clickhouse.json).

## Evidence boundary and next phase

P1 proves that factual posting can create supply that enters later Feed requests,
that lifecycle and removal decisions remain auditable, and that the mechanism
scales on one RTX 4090. The response world remains a handwritten partially
observed SCM. Therefore P1 cannot decide whether LR, XGBoost, W&D, DCNv2 or MMoE
should launch. P2 must calibrate population, sessions, response, retention and
creator dynamics before the learned cascade begins.
