# P3 Request Sample Authorities Review

Decision: `PASS` for the P3-01..03 **sample-contract slice**.

This decision accepts the v4 Recall, Coarse and Fine materialization contracts.
It does not promote a retrieval or ranking model, claim an A/B lift, or complete
P3. Trainer consumption, the feature/FID manifest and all model ladders remain
gated by P3-04 onward.

## Authority changes

- Recall uses fixed source budgets for in-batch, exposed, mined semantic and
  catalog draws. Every valid negative retains its source-conditional q,
  expected draw count, observation status and false-negative risk. Corrected
  sampled softmax subtracts log expected count only from sampled negatives; the
  always-present positive is unchanged.
- In-batch sampling is O(requests × draws), not O(requests squared). Duplicate
  positive items use their exact peer frequency when computing q.
- Coarse examples retain recall score, coarse rank, fine teacher score/rank and
  a candidate-level order-conflict mask. Hard labels remain exposure-derived;
  unexposed candidates are not assigned behavior zeros.
- Fine examples retain recall/coarse/fine lineage, separate task applicability
  and maturity, exact maturity time, joint assignment/exposure probability and
  an explicit support mask.
- The point-in-time history is chronological and heterogeneous: item, event
  type, surface, duration, occurrence time and ingestion time. Short sequence is
  a declared suffix of the same long-sequence authority.
- Full-flow schema v3 persists the new lineage in request, mature-label and
  training-example tables.

## Correctness evidence

The focused tests independently cover source allocation, q and expected-count
semantics, false-negative masking, exhaustive-softmax equivalence, teacher order
conflicts, applicability versus maturity, joint propensity and chronological
point-in-time history. The complete v4 suite passes with 62 tests.

The source audit also found and removed a scale defect before acceptance: the
first in-batch implementation materialized a quadratic peer matrix. The accepted
implementation samples peer indices directly and calculates q from batch item
frequency without allocating that matrix.

## RTX 4090 scale evidence

The content-bound report is
`reports/launches/2026-08-25-p3-request-sample-authorities-100k-gpu.json`
with SHA-256
`bc95d694c39ca173cb6c7f9750887f7cad15b84b6593863eb980f866e7a97d90`.

| Measure | Result |
|---|---:|
| Users / catalog items / ticks | 100,000 / 1,000,000 / 2 |
| Factual requests | 191,710 |
| Cascade width | 96 → 48 → 16 → 8 |
| Recall-positive requests in final partition | 80,765 |
| Negative draw budget | 12 in-batch + 3 exposed + 2 mined + 3 catalog |
| Valid negative rate | 99.4374% |
| False-negative risk rate | 17.3775% |
| Observed-negative rate | 15.0826% |
| Fine logging support | 100% of valid impressions |
| Historical events / event types | 1,286,014 / 19 |
| Cascade / Joiner time | 91.389 s / 0.086 s |
| Peak CUDA memory | 6.217 GiB |

The false-negative rate is evidence for retaining the mask, not a quality
regression: sampled peer/catalog items often occur in the user's prior history.
P3-04 learners must exclude masked rows and consume the stored expected count.

## Remaining gates

P3-04 must connect these immutable partitions to the active/candidate trainer
lanes and prove idempotent watermark/resume plus checkpoint compatibility.
P3-05 must compile the feature/FID manifest to both training and serving.
Teacher Top-K pass-through, pairwise/listwise training and identified DR remain
model/evaluator gates in P3-07..09; this review does not pre-accept them.
