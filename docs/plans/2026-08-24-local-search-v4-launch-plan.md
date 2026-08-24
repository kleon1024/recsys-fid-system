# Local Search V4 Launch Plan

Status: deferred P5 design input. Execution order and acceptance are owned by
`docs/plans/digital-twin-v4-execution-plan.md`; this file is not a backlog authority.

## Scope

Add the missing request-level Local Search retrieval, fine-rank, and end-to-end
Launch Reviews without changing accepted Feed, POI distribution, POI posting,
or Feed posting evidence.

## Current findings

- V4 composite authority promotes the external Feed behavior kernel only.
- The Launch ledger has five missing cells: three Local Search and two POI Detail.
- Architecture lint has zero errors; no directory move is required before work.
- The shared paired A/B statistics, tensor gather, artifact replay, and release
  verification are already canonical and must be reused.

## Implementation order

1. Define a Local Search request contract and teacher-hidden query/session world.
2. Materialize lexical, geo, semantic, graph/history, and retarget candidates with
   a fixed corpus and candidate budget.
3. Preserve request-level candidates, actual exposure, point-in-time features,
   cascaded click/detail/order labels, and open-loop observability masks.
4. Run retrieval Launch Reviews with a frozen baseline ranker.
5. Rematerialize exposure after the accepted retrieval policy, retrain Linear,
   boosted-tree, Wide & Deep, and sequence-aware rankers, then run fine-rank reviews.
6. Run end-to-end LT and transaction guardrails, bind artifacts, publish a
   simulator-only authority, and update the Launch ledger.

## Invariants

- No latent teacher field is available to retrieval or ranking models.
- No audit oracle is injected into the candidate set.
- Search model labels come only from actual exposed results.
- Retrieval promotion precedes fresh logging and fine-rank retraining.
- Local transaction or Pixel value is not directly relabeled as LT.
- Existing release reports and accepted metrics remain byte-identical except for
  manifests or authorities whose source closure intentionally changes.

## Verification

- `maestro-dataeng lint-architecture`
- focused Local Search request, label, artifact, and stage-isolation tests
- RTX 4090 multi-seed Launch Review
- `python -m fid_lab.check` on the complete remote environment
- GitHub Reference acceptance
