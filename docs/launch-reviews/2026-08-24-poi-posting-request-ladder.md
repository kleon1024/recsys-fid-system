# L-POI-POST-REQUEST-001 — POI posting request and model ladder

Change type: posting candidate retrieval, fine rank, and supply-to-Feed value  
Scale: 200,000 posting requests × 3 seeds  
Decision: promote History recall and Wide & Deep in the simulated authority  
Evidence: `reports/launches/2026-08-24-poi-posting-request-launch-review.json`

## Root cause / 根因

The historical demo forced the latent target POI into every candidate slate and
generated selection from features visible to the model. Its trained MMoE and
the downstream supply switchback were separate worlds. It could demonstrate a
model API but could not support a candidate, model, or end-to-end Launch Review.

历史 demo 把 target POI 强制放入候选集，行为公式又直接读取模型可见特征；训练模型和
供给 switchback 还是两条断链。因此它只能证明接口可运行，不能证明召回、精排或端到端
业务增量。

The new request authority uses a hidden teacher and noisy observed features:

```text
latent creator intent + hidden preference + nonlinear interaction
    -> Popular / Geo / Semantic / History candidate routes
    -> 20 recalled candidates
    -> 8 actually exposed candidates
    -> select -> publish -> quality-adjusted supply
    -> fixed-load Feed content risk + stay/active-day LT components
```

The audit oracle is never forced into candidates. Only exposed candidates enter
behavioral training. Publish is the entire-space `selected and published`
label. Time splits are 70/15/15 by request order. Model inputs receive noisy
draft/history/location signals; the teacher retains latent intent and hidden
nonlinear interactions.

## Candidate Launch Reviews / 投稿候选实验

| Proposal | Oracle recall | Publish/request | Platform LT/request | Content risk | Decision |
|---|---:|---:|---:|---:|---|
| Popular+Geo → +Semantic | +0.03977 | +0.00749 | +0.000877 | -0.000273 | Hold; 2/3 seeds pass |
| Popular+Geo → +History | +0.06083 | +0.04726 | +0.003085 | -0.000716 | Pass all seeds |

Semantic retrieval improves opportunity and average value, but one seed cannot
prove a positive publication effect. History retrieval improves the complete
funnel and clears every seed, so it becomes candidate authority.

Semantic 召回提高了 oracle recall 和平均价值，但一个 seed 的发布增量不显著，因此
不能晋级。History 召回在三个 seed 都提高发布、相关供给与 LT，因此成为候选 authority。

## Fine-rank model evolution / 精排模型演进

| Control → treatment | Publish/request | Platform LT/request | Content risk | Decision |
|---|---:|---:|---:|---|
| Rule → Linear | +0.04947 | +0.002969 | -0.001755 | Pass all seeds |
| Linear → Wide & Deep | +0.00604 | +0.000434 | -0.000289 | Pass all seeds |
| Wide & Deep → MMoE | +0.00062 | +0.000069 | -0.000094 | Hold; 1/3 seeds pass |

Validation AUC also evolves in the expected direction. Across the three seeds,
Linear publish AUC is 0.7865–0.8049, W&D is 0.8088–0.8133, and MMoE is
0.8098–0.8135. MMoE has slightly greater offline capacity but no stable
incremental online effect over W&D, so complexity is not promoted by name.

## End-to-end business effect / 端到端业务结果

The accepted stack is:

```text
Popular+Geo + Rule
    -> History recall + Wide & Deep
```

Across three seeds it changes:

- publish/request: `+0.14611`;
- relevant supply/request: `+0.13877`;
- platform LT/request from stay and active-day components: `+0.008737`;
- selected-content negative risk at fixed Feed load: `-0.004024`.

The early-stage synthetic lift is intentionally much larger than a mature Feed
experiment. It demonstrates the evolution mechanism, not a forecast for a
large production platform.

## Artifact and serving evidence / 模型与服务证据

All nine artifacts (three model families × three seeds) are hash-bound under
`artifacts/models/poi-posting-request-v1/`. Save/load replay has exactly zero
score delta. The simulated authority is
`artifacts/releases/simulated-poi-posting-control.json`; its rollback is the
Popular+Geo rule policy.

## V4 boundary / V4 边界

The external V4 authority validates Feed behavior only. It explicitly does not
authorize POI, supply, transaction, retention, or commercialization effects.
This posting world follows the multi-agent latent-state structure motivated by
[RecSim NG](https://arxiv.org/abs/2103.08057) and the request/session/retention
separation in [KuaiSim](https://arxiv.org/abs/2309.12645), but it remains a
teacher-hidden synthetic world.

Production promotion requires creator-side impression/action logs, artifact-
bound randomized posting interventions, and a city-period or creator-cluster
test for downstream supply interference. Until then the release state is
`hold_external_creator_and_supply_validation`.

