# L-RETRIEVAL-003 — v4 Fixed-Budget Retrieval Review

Decision: `PASS` for the P3-06 implementation and evidence contract. Retain the
lifecycle-rule retriever as control. Reject Graph, RRF, Two-Tower and
Multi-interest challengers for launch on this factual window.

Evidence boundary: synthetic offline and serving-shadow evidence only. No model
was promoted, and this review makes no A/B, production or TikTok claim.

## What changed

- Added one `learning/retrieval` authority that consumes persisted v4
  `RecallExample` rows and request-time context instead of creating another
  synthetic world or negative sampler.
- Reused the accepted source-specific expected-count correction and
  false-negative mask. Positive targets remain unchanged because they are always
  present in sampled softmax.
- Added a versioned retrieval feature contract, observable Two-Tower and
  Multi-interest models, FAISS IVF-Flat index, registry round-trip and the real
  platform `recent_ann` injection boundary.
- Compared lifecycle rules, co-visit Graph, their fixed-budget RRF merge,
  Two-Tower and Multi-interest on one corpus, query set, Top-50 and 10 ms/query
  budget. A fixed observable ranker diagnoses downstream candidate value.
- Fixed two infrastructure defects found by the new path: Hive-style
  `event_time=...` paths collided with the request table's `event_time` column,
  and lifecycle fallback performed quadratic whole-corpus list membership.

## Frozen run

The RTX 4090 artifact covers 100,000 users, 1,000,000 items, four event-time
partitions, 255,046 training queries and 10,000 deterministic evaluation queries.
It completed in 190.227 seconds and peaked at 14.840 GiB CUDA. Both neural models
ran four equal-budget epochs over 20 source-corrected negatives per query.

| Candidate policy | Recall@50 | Fixed-ranker Recall@20 | Coverage | ms/query | Decision |
|---|---:|---:|---:|---:|---|
| Lifecycle rules | 0.7854 | 0.2484 | 0.000050 | 0.0246 | Control |
| Co-visit Graph | 0.7532 | 0.2560 | 0.000114 | 0.0117 | Reject |
| Lifecycle + Graph RRF | 0.7741 | 0.2601 | 0.000101 | 0.0631 | Reject |
| Two-Tower | 0.6291 | 0.1845 | 0.000468 | 0.0098 | Reject |
| Multi-interest | 0.6691 | 0.2120 | 0.000438 | 0.0738 | Reject |

The IVF candidate sets retain 98.06% and 97.94% of exact Top-50 respectively;
ANN approximation and latency are not the reason for model rejection. Training
loss also decreases each epoch, so a broken optimizer is not the diagnosis.

## Root cause and boundary

The factual target is conditioned on the old platform's exposure. Only 530
different positive items occur in 255,046 training queries; the 10,000-query
evaluation set contains 89 different positives, 72.34% of which are already in
the train-frequency Top-50. The rule retriever therefore has a large logging
policy advantage and extremely low catalog coverage.

This review does not modify the DGP or weaken Recall@50 to make a neural model
win. The learned models expose more catalog items, but the logged data cannot
identify whether those novel candidates improve user outcomes. P3-09 must test
that question with supported randomized retrieval traffic or a factual treatment
A/B. Until then, all learned retrieval artifacts remain rejected and Semantic-ID
retrieval remains blocked behind an accepted dense baseline.

## Evidence

- Raw report:
  `reports/launches/2026-08-25-p3-fixed-budget-retrieval-100k-gpu.json`
- SHA-256:
  `fd0a1d52ad06ff3972aad2c41bb01acf1b5af3b2fc821e6c818f1da705b311e2`
- The report's `learning_source_sha256` matches the synchronized source closure.
- Architecture lint: zero errors and the declared custom-AssetGraph warning.

Next: P3-07 coarse ranking must consume the same factual candidates and feature
contracts. P3-09 owns the randomized retrieval truth gap; it must not be hidden
inside another offline metric.
