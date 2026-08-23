# DGP architecture decision research / DGP 架构决策研究

Status: research complete; neural-SCM challenger implemented and evaluated

Decision scope: the successor to the V1–V3 synthetic behavior world, not the ranking models

## Executive conclusion / 核心结论

V1–V3 did not produce three materially different user worlds. V1 is a mostly linear hand-authored response function. V2 adds heavy-tailed attributes and a feature-derived nonlinear stay adjustment. V3 fixes random-stream correlation, changes long-view semantics, and calibrates several aggregate rates to KuaiRand. V3 is a useful bug-fix and calibration epoch, but it is not a learned behavioral digital twin.

V1–V3 并不是三个实质不同的用户世界。V1 是近线性的人工响应函数；V2 加入重尾属性和由显式特征计算的非线性 stay 修正；V3 修复随机流相关性、调整 long-view 口径，并对齐若干 KuaiRand 总体比例。V3 是必要的正确性修复，但不能称为 learned behavioral digital twin。

The next authority should be a **partially observed neural structural causal world model**, trained as an ensemble and surrounded by explicit causal and validation contracts. It should combine a learned latent sequence model with explicit slate, session, provider, and intervention structure. A pure formula simulator remains only a unit-test oracle. A pure LLM-agent simulator is not the high-throughput numeric authority.

下一代 authority 应采用**部分可观测的神经结构因果 world model**：以 ensemble 形式训练，把 learned latent sequence model 与 slate、session、provider 和 intervention 的显式结构结合。纯公式模拟器只保留为单元测试 oracle；纯 LLM agent 不作为高吞吐数值模拟 authority。

## Why V3 still favors XGBoost / 为什么 V3 仍偏向 XGBoost

The current behavior kernel constructs watch-time truth from affinity, quality, satisfaction, fatigue, duration, plus a nonlinear adjustment made from affinity×quality, a hard threshold, hashed segment match, short/long sequence match, and a sine term. The evaluated models receive those same 28 scalar fields. The teacher therefore exposes nearly all ingredients of its own rule.

当前 behavior kernel 用 affinity、quality、satisfaction、fatigue、duration 生成 watch time，再叠加 affinity×quality、硬阈值、hash segment match、短长序列 match 和正弦项。被评估模型又直接获得同一组 28 个标量字段。也就是说，teacher 几乎把自己的公式原料全部泄漏给 student。

This creates three structural biases:

1. Axis-aligned threshold and piecewise effects are especially easy for boosted trees.
2. Most ladder models train only the long-view binary label, which is a thresholded derivative of watch time; they do not learn the joint play, stay, engagement, leave, and return process.
3. External calibration matches marginal rates, not conditional or interventional behavior. Matching overall play, like, and stay does not prove that two user cohorts, two slates, or two policies induce the right response difference.

The V3 benchmark confirms the distinction. XGBoost improves pointwise AUC only from 0.5934 to 0.5953 over logistic regression, but its request-level audit regret is 0.0922 versus 0.0393 for logistic regression; the hand-authored personalized rule reaches 0.0160. Calling this an XGBoost win conflates a small classification gain with worse candidate choice.

V3 benchmark 也验证了这个问题。XGBoost 相对 logistic regression 的 pointwise AUC 仅从 0.5934 提高到 0.5953，但 request-level audit regret 从 0.0393 恶化到 0.0922；人工 personalized rule 是 0.0160。因此，“XGBoost 赢了”只在一个分类指标上成立，并不代表 request-level 排序更好。

## What the literature establishes / 文献给出的边界

### Configurable simulators

[RecSim](https://arxiv.org/abs/1909.04847) makes user latent state, preference dynamics, familiarity, choice, and response models configurable. [RecSim NG](https://arxiv.org/abs/2103.08057) extends this into a differentiable probabilistic multi-agent ecosystem with latent-variable learning and accelerated execution. [SARDINE](https://arxiv.org/abs/2311.16586) emphasizes dynamic users, recommendation-induced change, and feedback from biased logged data. [T-RECS](https://arxiv.org/abs/2107.08959) demonstrates why ecosystem and societal feedback require more than one-step accuracy.

These systems supply the correct abstraction and intervention semantics. They do not by themselves make a hand-authored world realistic. 可配置框架解决“怎么表达世界”，不自动解决“这个世界是否像真实用户”。

### Learned and data-driven simulators

[Virtual-Taobao](https://ojs.aaai.org/index.php/AAAI/article/view/4402) learns a virtual retail environment from real interactions and addresses the risk that a policy exploits simulator defects. [KuaiSim](https://arxiv.org/abs/2309.12645) models request-level list response, session behavior, multi-behavior feedback, and cross-session retention. [RL4RS](https://arxiv.org/abs/2110.11073) explicitly calls out the reality gap and separates simulator evaluation, policy evaluation inside the simulator, and counterfactual evaluation. [Sim2Rec](https://doi.org/10.1109/ICDE55515.2023.00260) treats the simulator as an uncertain set of possible environments rather than a single trusted truth.

The common lesson is that a learned response model is necessary but insufficient. A single maximum-likelihood simulator can still be exploited, drift outside logging-policy support, and compound one-step error over long trajectories.

共同结论是 learned response model 必要但不充分。单一最大似然 simulator 仍可能被 policy reward-hack，在 logging-policy support 之外外推失败，并在长轨迹中累积误差。

### Causal and off-policy evaluation

Slate actions create a combinatorial support problem. [Slate OPE](https://arxiv.org/abs/1605.04812) derives estimators using slate structure; [sequential slate reward evaluation](https://doi.org/10.1145/3394486.3403229) relaxes reward-independence assumptions; [distributional slate OPE](https://ojs.aaai.org/index.php/AAAI/article/view/28667) estimates outcome distributions rather than only the mean. Counterfactual risk minimization, SNIPS, and doubly robust estimators reduce specific biases or variance, but none identifies policies that have no overlap with logged actions.

因此 simulator 与 OPE 不能互相替代。Simulator 用于机制压力测试和 trajectory rollout；随机/探索日志上的 IPS、SNIPS、DR 用于约束 simulator 的因果外推。没有 overlap 时，两者都不能诚实地产生“可信线上增量”。

### LLM and generative agents

[Agent4Rec](https://doi.org/10.1145/3626772.3657844) equips LLM agents with profiles, memory, emotion, and actions; [LLM-Powered User Simulator](https://doi.org/10.1609/aaai.v39i12.33456) combines explicit logic with statistical modeling. These approaches are valuable for semantic intent, explanations, conversational behavior, and rare scenario generation.

They are currently a poor primary engine for tens of millions of reproducible numeric Feed events: token inference is expensive, exact probability calibration is difficult, model upgrades change the DGP, paired-counterfactual random numbers are awkward, and the agent can import world knowledge unavailable to real users. LLM agents should generate content/intent scenarios or challenge the numeric simulator, not own the A/B truth.

LLM agent 适合生成语义意图、解释、对话和罕见场景；不适合直接承担千万级可复现 Feed 行为和 A/B 真值。它应当是 scenario generator 或 adversarial challenger，而非数值 authority。

## Architecture alternatives / 方案比较

| Alternative | Strength | Fatal limitation for this project | Decision |
|---|---|---|---|
| More hand-authored nonlinear rules | Transparent, fast, exact counterfactual truth | Continues feature/formula leakage and designer bias | Keep only as deterministic test world |
| One-step learned response model | Easy to train and calibrate | Misses state evolution, slate competition, compounding error | Component only |
| Pure autoregressive sequence model | Learns joint temporal patterns | Correlation is mistaken for intervention; weak outside support | Component with causal scaffold |
| Pure LLM agents | Rich semantic reasoning and rare scenarios | Slow, unstable, poorly calibrated numeric behavior | Challenger/scenario lane only |
| Pure agent-based mechanistic model | Explicit users/providers and interference | Parameters remain hand-authored and hard to calibrate | Structural scaffold only |
| Hybrid neural SCM ensemble | Learned distributions plus explicit interventions and uncertainty | More complex training and validation | Selected research direction |

## Selected design / 选定设计

The selected design has six separable authorities:

1. **Exogenous population and catalog process.** Heavy-tailed users, creators, items, regions, lifecycles, supply arrival, quality, deletion, and inventory. Correlated latent factors are learned with a copula/normalizing-flow or variational population model, not independent uniform draws.
2. **Partially observed user state.** A latent state-space model carries stable preferences, multiple interests, novelty seeking, fatigue, satisfaction, commercial intent, and session intent. The policy sees only noisy point-in-time projections and logged behavior, never the latent state.
3. **Slate response world model.** A set/attention encoder represents position, neighboring items, duplication, author/category concentration, organic/ad/live competition, and opportunity cost. An autoregressive multi-action decoder generates play start, censored watch time, slide, like, favorite, share, comment, negative feedback, POI anchor/detail, and transaction events with logical masks.
4. **State transition and survival.** A recurrent or Transformer state transition consumes the chosen slate and sampled actions. Separate survival/hazard heads model session exit, return delay, active day, and creator/supply response. This makes long-term effects an emergent outcome rather than a direct LT formula.
5. **Causal noise and uncertainty.** Every stochastic structural equation accepts versioned exogenous noise so control and treatment can run paired potential-outcome trajectories. An ensemble or posterior over worlds represents epistemic uncertainty; policies must pass across plausible worlds, not exploit one simulator.
6. **Evidence boundary.** Randomized/exploratory logs calibrate propensities and intervention effects; standard logs train conditional dynamics; held-out policies test policy-order agreement; synthetic stress worlds test invariants. None alone is called production truth.

这六层必须物理隔离：population、latent user state、slate response、state transition、causal noise、evidence boundary。模型训练只能读取 serving-observable features；simulator audit 可以读取 latent truth。这样才能真正消除 teacher 公式泄漏。

## Is public data required? / 是否必须公开数据

No. Public data is not an architectural requirement. The preferred evidence order is:

1. proprietary randomized or exploration traffic with exact propensities;
2. proprietary standard logs plus orthogonal A/B results and mature labels;
3. public randomized datasets such as KuaiRand for external falsification;
4. public observational datasets for shape and pipeline tests;
5. fully synthetic teacher worlds for mechanisms and stress tests only.

不要求公开数据。真实内部随机流量和历史 A/B 才是最高价值证据。公开数据的作用是让开源项目可复现、提供外部反证，并避免 simulator 完全迎合自己生成的样本。若没有任何真实或随机日志，仍可构造复杂 neural DGP，但只能称为“复杂合成世界”，不能称为“拟合真实 TikTok 用户”。

## Acceptance protocol / 验收协议

V4 is accepted only if it passes all five gates below. Matching marginal rates is not sufficient.

| Gate | Required evidence |
|---|---|
| Distribution | Joint and conditional calibration for watch-time quantiles, action correlations, session length, return delay, head/tail users/items, and Local funnels |
| Sequence | Multi-step rollouts reproduce transition matrices, autocorrelation, novelty/fatigue curves, and cross-session retention without teacher forcing |
| Intervention | Recover the sign and approximate magnitude of held-out randomized product/model/strategy treatments by cohort and trigger |
| Policy | Rank multiple frozen policies in the same order as replay/OPE/A/B where support exists; show uncertainty when it does not |
| Anti-exploitation | No candidate policy may win only in one simulator member, by accessing latent state, or by driving trajectories outside empirical support |

Additional negative controls are mandatory: shuffled treatment, impossible future feature, duplicated event, removed propensity, and an intentionally reward-hacking policy must all fail. The simulator must report coverage, support distance, ensemble disagreement, and rollout drift beside every synthetic lift.

## Decision and next step / 决策与下一步

Do not add another nonlinear formula and call it V4. Preserve V1–V3 as historical control epochs. The neural-SCM research challenger is now implemented on the frozen request-level dataset. It passes distribution, free-running sequence, ensemble uncertainty, frozen-V3 intervention recovery, and synthetic policy-order gates. External randomized intervention recovery and real frozen-policy ordering are still unavailable, so it has not replaced V3 as the experiment authority.

不要继续给 V3 加公式。V1–V3 保留为历史 control。受控 neural-SCM prototype 已完成并通过合成世界门禁；下一步必须接入内部随机实验与历史 A/B。只有恢复外部 held-out intervention 并正确排列真实 frozen policy，才允许升级为新的 experiment authority。
