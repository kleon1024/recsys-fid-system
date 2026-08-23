# L-SIMULATOR-008 — External stateful shadow A/B

Change type: W&D control to sequence-Transformer treatment under an independent world kernel

Decision: Hold; negative-feedback uncertainty remains

Hardware: RTX 4090 24GB, CUDA 12.6

Evidence:

- `reports/world-model/2026-08-24-kuairand-shadow-ab-200k.json`
- `reports/world-model/2026-08-24-kuairand-external-capacity-seed26-4090.json`

## Experiment contract

The frozen control is the seed-20260824 W&D artifact. The frozen treatment is
the seed-20260824 sequence Transformer. A seed-20260826 Transformer, never used
during strategy iteration, owns potential outcomes. All three artifacts are
SHA-256 bound in the report.

The final replay uses 200,000 users, eight sequential request/feedback updates,
20 candidates per request, identical candidate slates, identical initial
histories, and paired structural random numbers. This represents 1.6 million
state transitions. Selected items and sampled feedback are appended to each
arm's separate point-in-time history before the next request.

The treatment is control-feasible: candidates must preserve control-predicted
click, long view, like, and stay within declared tolerances and must improve
treatment-predicted hate by at least 0.005. The ranking utility combines
primitive outcomes and is explicitly not unified LT.

## Iteration record

An unconstrained sequence treatment improved long view and stay but reduced
like by about four percentage points in the 1K smoke. Increasing the like weight
reversed that metric but caused large click and long-view regressions. A
control-feasible constrained ranker removed the large trade-offs and reduced
effects to the industrially plausible ten-thousandth range.

The guardrails were frozen before the independent seed-20260826 world and final
200K run. The 100K result was used only for a predeclared power calculation;
the estimated hate effect required roughly 192K users, so the final sample was
fixed at 200K. No gate or coefficient changed after that calculation.

## Final result

| Metric | Absolute delta | 95% confidence interval | Gate |
|---|---:|---:|---|
| Click | +0.00491 pp | [+0.00417, +0.00564] pp | Pass |
| Long view | +0.00533 pp | [+0.00453, +0.00613] pp | Pass |
| Like | -0.00095 pp | [-0.00359, +0.00170] pp | Pass guardrail |
| Hate | -0.00423 pp | [-0.00950, +0.00103] pp | **Fail** |
| Normalized stay | +0.00156 pp | [+0.00111, +0.00201] pp | Pass |

Hate improves on average, but its upper confidence bound remains positive. The
experiment therefore holds. Increasing the sample again would use nearly the
entire incomplete Pure test set and still would not provide randomized causal
evidence, so sample-size escalation stops here.

## Decision boundary

The request/slate adapter, state transition, artifact replay, model-capacity
gates, and large-scale shadow A/B are operational. V4 is not promoted because:

- the hate hard guardrail has not cleared uncertainty;
- KuaiRand-Pure lacks the random-exposure log required for causal falsification;
- external LT exchange rates and real frozen-policy outcomes remain unavailable.

下一步不是继续调 simulator 让结果通过，而是接入 KuaiRand-1K random log 或等价
随机流量，训练跨 seed uncertainty ensemble，并用真实 random exposure 校准 hate 和
policy ordering。V3 继续作为可执行 synthetic authority，外部 V4 kernel 保持 challenger。
