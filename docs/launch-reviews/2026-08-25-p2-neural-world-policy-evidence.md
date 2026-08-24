# P2 Neural Feed World Launch Review

Decision: `PASS` for manual promotion as the v4 **Feed response authority**.

Formula remains the deterministic invariant oracle. Retention, creator supply,
Search, Ads, Commerce, Local, Posting, Live and unified LT are outside this
promotion and remain independently gated.

## Exact authority closure

| Artifact | SHA-256 |
|---|---|
| v23 artifact manifest | `fc8fa5fd9bb81a3e29e06ecf1ecc69acaaa4fbc8d78cdf4790474d60035ad465` |
| frozen model weights | `8d61966e5472a649773453540a48c175c7c73f26ec5b75c20f9baa25779b9726` |
| feature contract | `f8cde7b032f2f0ad4d3cd7b91a85f85a3a1fb9355ea9f0a7a7ba05e402646082` |
| unified review file | `b2d0480baa37ed684233b90c9c3c3fe73ff9817a918f34f895fe83d69b280880` |
| randomized-policy file | `639b6319fc88f069d10d7336b6233acc46514c788432c5085a41b012a8e3d72e` |
| 100K shadow file | `b7a2064f60462b891a6b002f939ad99f7ce0f52c142cced9bb7b0ad0bb7c263c` |
| structural manifest | `c76f30d08614302040fb7f9dc7308a8886681baedf7cf3f6bb8acb5857800a31` |

The v23 support-only repackage retained the exact v20-v22 weights. It did not
retrain the model or move an acceptance threshold.

## Why the prior HOLD is closed

The earlier external-only model used incompatible feature semantics and failed
held-out structural directions. The real-cascade bridge now uses family 1 and 2
for training, family 3 for validation, and untouched family 9 for the final test.
The final model recovers all three intervention signs with normalized magnitude
MAE `0.27419`, below the frozen `0.30` threshold.

The former multi-domain support profile merged two structurally different train
families and covered only `93.069%` of a 20K factual shadow. v23 preserves three
separate support components: external source, structural family 1 and structural
family 2. Their union covers `98.9835%` of 412,006 factual Feed requests in the
100K-user shadow. Every component rejects the eight-scale adversarial attack;
the aggregate rejection rate is `100%`.

## External and held-out evidence

| Gate | Result |
|---|---:|
| Mean binary ECE | `0.01725` |
| Joint action-correlation MAE | `0.02801` |
| Stay P50 / P90 relative error | `3.96% / 1.61%` |
| Free-running lag-1 MAE | `0.07649` |
| Ensemble probability std P99 | `0.05510` |
| Policy Kendall tau | `0.66667` |
| Policy-value normalized MAE | `3.26%` |
| Identified randomized-policy pairs | `6 / 6` |
| Minimum ESS fraction | `74.76%` |
| Maximum importance weight | `9.41` |
| Held-out structural sign accuracy | `3 / 3` |
| Held-out structural normalized MAE | `0.27419` |

The randomized evidence identifies immediate Feed utility only inside the
declared 7,388-item KuaiRand pool. It does not identify long-term retention,
creator response, business conversion or LT.

## Factual-cascade shadow

The final shadow runs 100K users, one million catalog items and eight ticks using
the real reference cascade. Formula responses alone mutate the factual world;
NeuralSCM is evaluated as the non-committing challenger.

- 412,006 factual Feed requests and 4,467,336 neural impressions.
- Support rate `98.9835%`, above the frozen `97%` gate.
- Exact request, item, post, creator, event and schema lineage across both runs.
- Maximum float delta `0`; maximum duration quantization delta `1 ms`.
- Neural and Formula replay, factual slate replay and partition invariance pass.
- 4,096-request inference micro-batches are part of the recorded configuration.
- Wall time `7m25.64s`; peak RSS `2.46 GiB` on RTX 4090.

Streaming replay evidence reduced the 20K peak RSS from `4.51 GiB` to `1.80 GiB`.
The 100K run exposed and closed two real scale defects before acceptance:

1. Whole-batch attention exceeded a CUDA launch limit. The authority now owns a
   bounded inference micro-batch contract and proves partition invariance.
2. The former linear event hash admitted collisions. Event schema v5 now uses an
   injective 45-bit request, 6-bit event type and 12-bit request-local ordinal
   encoding and rejects out-of-contract values.

## Verification and transition

The exact synchronized source passed 202 historical tests, 57 v4 digital-twin
tests and the repository acceptance report. Local architecture lint reports zero
errors and the documented asset-wrapper warning.

Promotion means P3 may use `NeuralFeedResponseAuthority` as its Feed behavior
world when the exact artifact is explicitly supplied. The generic world
constructor keeps Formula as its dependency-free invariant-test default because
weights are not committed to the public repository. No non-Feed head is silently
enabled.
