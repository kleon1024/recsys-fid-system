# P3 Streaming Learning and Feature Authority Review

Decision: `PASS` for the P3-04/05 infrastructure slice. The LR probe remains
`HOLD` as a recommendation challenger.

This review accepts the persistent sample bus, feature/FID authority, checkpoint
registry and learned-serving adapter. It does not select LR, XGBoost or a deep
model, claim recommendation lift, or complete P3.

## Authority changes

- One manifest declares all 11 dense and 13 sparse fields, including observable
  source, transform, group, dtype, default, TTL, namespace, slot, bucket count
  and vocabulary. It covers user, item, creator, context, route, counter,
  sequence, content and explicit crosses without hidden-world fields.
- The platform computes features once. The exact fine-candidate dense tensor,
  FID and bucket tensor enter `RequestCandidateTrace`, the Joiner and full-flow
  schema v4. Offline replay no longer reconstructs a second feature formula.
- Fine training rows persist the complete task-label, applicability and maturity
  vectors. The previous unweighted sum across positive and negative tasks was
  removed as a training target.
- Dataset schema v3 includes the full feature manifest, sample contract, table
  schemas and partition hashes. Active and candidate lanes maintain independent,
  event-time ordered cursors over the same immutable partitions.
- The persistent registry binds dataset, feature, FID, catalog, index, corpus,
  code and artifact hashes. Incompatible snapshots fail before load or score;
  rejection does not mutate active; corrupt active content uses only the last
  compatible fallback.
- The platform owns a scorer protocol and does not import learning. An artifact
  changes ranking only after explicit validated installation, and every request
  logs its numeric `fine_version_id`.

## Correctness and failure evidence

Focused tests prove exact trace→Joiner→Parquet tensor replay, exact learned score
replay, FID collision reporting, feature-drift reporting, lane independence,
ordered/idempotent commits, restart, compatibility rejection, snapshot hold,
artifact corruption fallback and checkpoint/version joins. The LR probe exists
only to exercise these paths; its report explicitly sets
`purpose=infrastructure_only_not_model_launch`.

The complete synchronized repository gate passes 202 historical tests and the
v4 suite. Architecture lint has zero errors and the declared `AssetGraph`
decorator warning only.

## RTX 4090 scale evidence

The content-bound report is
`reports/launches/2026-08-25-p3-streaming-feature-authority-100k-gpu.json`
with SHA-256
`99e687187b99a27461a6f5ecdb256c3dce4eef915e182b36a53e4880c6c1a449`.

| Measure | Result |
|---|---:|
| Users / catalog items / ticks | 100,000 / 1,000,000 / 2 |
| Cascade width | 96 → 48 → 16 → 8 |
| Persisted fine rows | 1,267,664 |
| Mature task labels | 7,808,576 |
| Dense / sparse fields | 11 / 13 |
| Active / candidate consumed partitions | 2 / 2 |
| Generation / materialization | 46.994 s / 29.561 s |
| Load / two-lane probe training | 2.171 s / 0.722 s |
| Total | 88.266 s |
| Peak CUDA memory | 8.538 GiB |

Tick 0 is Posting and tick 1 is Feed, so the drift report correctly detects a
large surface and candidate-distribution shift. This is detector evidence, not
a quality regression or a frozen launch threshold. The active bootstrap probe
exercises publication; the candidate remains shadow `HOLD` because P3-09 has not
evaluated ranking or A/B value.

## Remaining gates

P3-06 must train the equal-budget retrieval ladder from the accepted recall
samples. P3-07 and P3-08 must train coarse and fine challengers from the same
feature bytes. P3-09 owns request-aware evaluation, identified OPE and the model
pass/hold/reject decision. Only after those gates may a probe architecture become
a recommendation model claim.
