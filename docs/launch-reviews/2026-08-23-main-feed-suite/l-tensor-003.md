# L-TENSOR-003 — Published model migration to the GPU tensor engine

Change type: simulator architecture and model-serving parity
Decision: tensor migration accepted; staged model launch passes unified LT gate
Evidence date: 2026-08-23
Hardware: RTX 4090 24GB, CUDA 12.6

Evidence boundary: this is a synthetic launch simulation. Production rollout
requires organization-approved causal LT exchange rates and live holdout evidence.

## Why this change was required

The semantic Gymnasium simulator was the causal contract authority, but 10,000
fresh users and two policy worlds took about nine minutes after GPU training.
The bottleneck was Python user state, candidate construction, and repeated
world execution, not neural or tree training. It could not support million-user
small-effect tests or many orthogonal parameter layers.

语义模拟器能够验证因果链路，但 GPU 训练结束后，1 万用户、两个策略世界仍需约 9 分钟。
瓶颈是 Python 用户状态、候选构建和重复 world 执行，无法支持百万用户小增量与大量正交
实验。因此本次改造的是模拟执行层，不是继续调模型。

## Published artifact contract

The same published policy is used by both engines:

- logistic-regression base artifact
  `sha256:af4ca7536ddc0080c834f8d9b3cc7262f6d946fb67ebcdf3e3b715c563ab2e5e`;
- expected-stay XGBoost artifact
  `sha256:6f3bb23dd1646d6a35d97befc576037903f7064cb52448ea74e13efb208cd4e3`;
- composite guarded-policy artifact
  `sha256:c3286e2b19f1d35b0fc0a85fc466ba58dd51256a36236f599730920383c6f939`;
- canonical 24-field feature schema
  `f9b7f08ad49b7db0317beed4a07d2155ff4bb909369ebb01e17e614f4c42d010`.

LR coefficients are converted to Torch tensors. XGBoost 3.2 loads its JSON
artifact and uses CUDA `inplace_predict`; CuPy provides zero-copy DLPack transfer
back to Torch. Candidate features, scoring, responses, and state transitions
remain device-resident. Hash or schema mismatch fails before simulation.

## Parity repair

The first tensor connection was invalid: stay was 4.39 seconds and quality view
2.9%, far from the semantic authority. The tensor world used signed Gaussian
interests, uniform quality and duration, and lacked session limits. V2 now uses
positive long-tail interest vectors, beta-like quality, log-normal duration,
the same 24 feature fields, nonlinear response terms, eight-request sessions,
and four-session termination. A declared `+0.25` stay-log intercept calibrates
the remaining candidate-distribution difference.

最终百万用户 control parity 为：stay gap -0.13%、long-view gap +0.20%、
quality-view gap -9.81%，全部通过 10% 分布门槛。stay 与 quality treatment effect
相对 semantic true ITT 的绝对差分别为 0.88 和 0.16 个百分点，均通过 2pp 门槛。

![Tensor migration evidence](../../assets/tensor-migration.svg)

## Performance and A/B result

One million users, 24 steps, and two common-random-number worlds completed in
about 13 seconds of measured simulation time:

| World | Requests/s | Users/s | Peak GPU memory |
|---|---:|---:|---:|
| LR control | 3.06M | 168.5K | 267MB |
| Guarded XGBoost | 2.51M | 138.2K | 401MB |

The tensor A/B estimates stay/exposure +0.74%, long-view +0.32%, unified
platform LT value/user +0.265%, and quality-long-view -1.49%. The absolute LT
increment is +0.03308 per user with a 95% confidence interval of
[+0.00393, +0.06223]. Its lower bound is nonnegative, so the model passes the
unified LT gate and may enter a simulated staged rollout with a retained holdout.
The simulator decision is `pass_unified_lt_nonnegative`; production readiness is
`hold_synthetic_rates` because this public reconstruction has no organization-approved
exchange manifest.

百万用户实验说明算法确实能改变业务核心，也暴露了 trade-off：stay 与统一 LT 为正，
quality-long-view 显著下降 1.49%。统一 LT 增量为每用户 +0.03308，95% 置信区间下界
为 +0.00393，因此 tensor engine 与模型上线门禁均通过，可进入分阶段放量并保留长期
holdout。

Quality-long-view is not ignored. It remains a diagnostic because the current
LT contract has no measured exchange coefficient for it. If long-term holdout
evidence proves a causal DAU, stay, or commercialization effect, that effect
must be calibrated into the LT container; the system must not hard-code a
second objective that double counts or overrides LT. Safety, legal, privacy,
and integrity constraints remain independently fail-closed.

## Remaining boundary

The tensor runner now owns scale and power; the semantic runner remains the
contract oracle. Tensor results are admissible only when distribution and
treatment-effect parity pass. The next iteration should investigate the quality
trade-off and estimate whether it has incremental long-horizon LT impact, while
the staged launch and retained holdout verify the synthetic result does not drift.
