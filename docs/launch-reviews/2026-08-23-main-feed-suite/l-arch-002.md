# L-ARCH-002 — Tensor runtime responsibility split

Status: `pass_semantic_parity`. Architecture-only synthetic release.

## Change

Split the 795-line tensor engine into a 437-line orchestrator plus contracts,
response-kernel, and state-transition modules. No model, feature, policy,
Value Tree coefficient, RNG key, candidate budget, or A/B assignment changed.

The active and rollback release bundles now bind all four behavior-source files.
The historical request dataset retains its original logging bundle and authority
ID; it is not relabeled as data produced by the refactored code.

## Verification

| Gate | Result |
|---|---|
| Stateful tensor tests | 15/15 pass |
| Repository acceptance | 99/99 pass |
| Counter RNG batch invariance | pass |
| Candidate-stage attrition | pass |
| Unified LT gate | pass |
| Historical external evidence hashes | unchanged |
| Architecture linter | 0 errors |

## Scale

On RTX 4090, 10 million users over 24 steps completed in 43.98 seconds at 5.01
million simulated requests per second. Peak allocated memory was 2.49 GiB with a
200k-user batch, confirming memory remains batch-bounded.

## Decision

Pass the code-only source-closure refresh. This does not promote V4, change the
active model, claim production QPS, or claim business lift.
