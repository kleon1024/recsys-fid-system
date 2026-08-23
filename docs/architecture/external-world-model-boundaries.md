# External World-Model Architecture

This boundary keeps model capacity, causal evaluation, and launch authority
separate. A higher AUC can enter evaluation, but it cannot promote the simulator
or be called LT without passing the launch state machine.

```mermaid
flowchart LR
    SRC["KuaiRand source logs"] --> DATA["data\npoint-in-time histories\nstandard/random splits\ncatalog"]
    DATA --> MANIFEST["immutable dataset manifest\nsplit + catalog hashes\nschema + vocabulary"]
    MANIFEST --> TRAIN["modeling\nW&D / Transformer / MMoE\npairwise adaptation"]
    TRAIN --> ARTIFACT["model artifact\nstate + training manifest\nparent lineage"]

    MANIFEST --> COMPAT{"artifact compatibility"}
    ARTIFACT --> COMPAT
    COMPAT -->|mismatch| FAIL["fail before scoring"]
    COMPAT -->|closed| CAL["evaluation\nrandomized calibration"]
    CAL --> OPE["randomized DR/OPE"]
    CAL --> SHADOW["stateful shadow replay"]

    POLICY["PolicySpec\nutility semantics\ntemperature\nexploration support"] --> OPE
    POLICY --> SHADOW
    OPE --> REVIEW["launch review"]
    SHADOW --> REVIEW
    REVIEW --> HOLD["hold / reject"]
    REVIEW --> RAMP["simulated A/B / staged ramp"]
    RAMP --> LT["LT measured only as final A/B outcome"]
```

## Ownership

| Package | Owns | Must not own |
|---|---|---|
| `data` | source parsing, point-in-time history, split and catalog materialization | model fitting or launch decisions |
| `modeling` | architectures, losses, training, adaptation, artifact production | causal claims or LT conversion |
| `evaluation` | calibration, policy scoring, randomized OPE, replay estimands | artifact promotion |
| `launch` | hashes, compatibility, policy contract, gate transitions | feature generation or training |
| `cli` | argument parsing and invocation | business logic |

The invariant is fail-closed compatibility. Dataset split bytes, catalog bytes,
feature schema, vocabulary, model state, calibration rules, and policy parameters
form one evidence closure. A rebuilt catalog invalidates old artifacts even when
the training split hashes are unchanged.

## Composite authority after randomized evaluation

V4 does not replace every behavior surface with one neural model. It composes
task kernels with independent evidence scopes:

```mermaid
flowchart TB
    Feed["Feed behavior\nexternal randomized V4"] --> World["Composite simulator world"]
    Local["POI and Local response\nsynthetic V3"] --> World
    Supply["Posting and supply\nsynthetic V3"] --> World
    Measure["Retention and commercialization\nmeasurement only"] --> World
    World --> Replay["request-level replay"]
    Replay --> Experiment["stage-specific simulated A/B"]
    Experiment --> Review["Launch Review"]
```

The Feed component is eligible because one frozen treatment artifact passes
randomized DR/OPE, two independent stateful shadow worlds, behavior guardrails,
and a one-million-user power simulation. Shadow stay effects are 3.1--4.0 times
the randomized OPE estimate, so the review preserves magnitude disagreement and
uses agreement only for primary direction and guardrails.

The unified NeuralSCM remains a challenger. Its external bridge has request-level
slates, point-in-time features, histories, observed labels, and masks, but the
source has no POI, supply, retention, or commercialization outcomes. Missing
tasks have `label_mask=0`; they are never written as negatives or proxy-mapped
to unrelated Feed actions.

`artifacts/releases/simulator-world.json` owns world-kernel selection.
`artifacts/releases/simulated-feed-control.json` owns the active ranking policy.
Neither file may mutate the other. V3 is the executable rollback epoch.
