# Refactor Plan

## Scope

- `fid_lab/feed_loop/world_model/external/kuairand/`
- `fid_lab/feed_loop/world_model/external/ope.py`
- `fid_lab/feed_loop/world_model/external/replay.py`
- `fid_lab/evolution/data/`
- `fid_lab/evolution/models/esmm.py`
- related CLI entry points, tests, architecture configuration, and documentation

## Findings

- Structure smell: `kuairand` contains 13 modules in one directory and mixes data materialization, models, training, evaluation, launch decisions, and CLI parsing.
- Code smell: benchmark and adapter CLI modules own model fitting, calibration, pair mining, artifact publication, and report construction.
- Contract smell: dataset, catalog, feature schema, model, calibration, and policy configuration are not validated as one immutable artifact closure.
- Evaluation smell: shadow replay and randomized OPE use separate utility and calibration implementations, so launch decisions can disagree for implementation reasons.
- Dependency smell: evaluation imports private data helpers; hashing is duplicated; ESMM is exported but has no executing owner.
- Semantic risk: V4 artifacts trained against the previous catalog can be loaded with the rebuilt catalog without a fail-closed compatibility check.
- Baseline: V3 remains the active simulator authority and all current V4 external evaluations remain `hold`.

## Moves

1. Capture hashes and decision fields for the current external capacity, shadow, dataset, and calibration reports as the golden behavior baseline.
2. Create one shared contract layer for streaming hashes, dataset/artifact manifests, `PolicySpec`, calibration rules, and compatibility validation.
3. Split KuaiRand into cohesive `data`, `modeling`, `evaluation`, `launch`, and `cli` subpackages; keep compatibility re-exports only at the public package boundary.
4. Move randomized dataset materialization/loading into `data`; remove imports of private helpers from other modules.
5. Move neural/tabular fitting, pairwise adaptation, and artifact serialization into `modeling`; make CLI modules argument-only adapters.
6. Make calibrated slate scoring and utility computation one evaluation authority consumed by both shadow replay and randomized OPE.
7. Add a launch pipeline/state machine that executes and records `offline -> calibration -> OPE -> shadow -> simulated A/B -> review`, with fail-closed artifact compatibility gates.
8. Centralize recall negative proposal semantics and expected-count correction so Joiner and retrieval training consume one sampling contract.
9. Wire ESMM into the model registry and focused tests instead of exempting an orphan module.
10. Update imports, entry points, architecture declarations, README diagrams, and launch documentation; remove superseded duplicate implementations after parity is proven.

## Invariants Preserved

- V3 remains the active authority; no V4 artifact is promoted by this refactor.
- Current CLI names remain callable through thin compatibility entry points.
- Existing report schemas and decision meanings remain readable.
- Current V4 randomized and shadow outcomes remain `hold` unless a later, separately reviewed experiment changes the evidence.
- Sampling probabilities remain per-draw conditional probabilities and loss correction uses expected source counts.
- LT remains an A/B outcome container; model utility scores are not renamed or presented as LT.

## Verification

- `maestro-dataeng lint-architecture --format json`
- `python3 -m fid_lab.check`
- focused model, data, OPE, replay, artifact-compatibility, and CLI tests
- golden report hash/decision verification
- import smoke for every retained CLI entry point
- `ruff check` on changed Python modules

## Acceptance Bar

- No business logic in CLI modules.
- No cross-package import of private symbols.
- One owner for hashing, manifests, calibration, policy utility, and launch state transitions.
- Dataset/artifact/catalog/schema mismatch fails before scoring.
- Architecture linter has no errors in the refactored scope.
- The frozen V4 launch decision remains `hold`; no production or simulator-authority claim is made.
