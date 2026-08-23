# L-SIMULATOR-002 — Nonlinear model-capacity audit

Change type: simulator, sample scale, and model-capacity diagnosis  
Decision: accept the V2 DGP as an offline research lane; no serving change  
Evidence date: 2026-08-23  
Evidence boundary: deterministic synthetic RTX 4090 runs

## Hypothesis

Advanced rankers failed because the V1 data-generating process exposed nearly
all useful signal as linear dense/cross features and the one-million-impression
sample was too small. If this is the cause, a versioned nonlinear DGP plus a
larger sample should create measurable capacity headroom without changing the
evaluation split or model zoo.

复杂模型失败的假设原因是：V1 DGP 将主要信号近似线性地暴露出来，且 100 万主曝光
只产生约 2 万条 anchor 样本。若根因成立，加入版本化非线性机制并扩大样本后，复杂
模型应在相同切分和模型集合中出现可测增量。

## Train and offline evidence

V1 signal diagnosis reached oracle AUC 0.6716. Logistic regression with the
known cross and sequence diagnostics reached 0.6719, leaving no positive
observable headroom. V2 adds unexposed threshold, periodic, segment,
recency-weighted sequence, and nonlinear intent/quality effects.

At one million main impressions, V2 produced 19,964 anchor examples and only a
0.0009 W&D-over-LR AUC difference. At ten million impressions, it produced
199,883 examples. Logistic regression reached AUC 0.6048; XGBoost 0.6161,
DCNv2 0.6111, MMoE 0.6115, and PLE 0.6120. The 10M training wall times were
0.14 seconds for CPU logistic regression, 0.50 for CUDA XGBoost, 9.33 for CUDA
DCNv2, 4.52 for CUDA MMoE, and 6.68 for CUDA PLE. XGBoost was switched to CPU
for the serving-style prediction timing, and every model card records both
devices explicitly.

V1 的已知交叉加序列逻辑回归已经触及 oracle，证明旧模拟器没有给复杂模型留下有效
空间。V2 在约 20 万样本时让 XGBoost、DCNv2、MMoE、PLE 全部超过逻辑回归，支持
“DGP 与样本规模共同限制模型”的根因判断。

## Shadow, A/B, and gate

Shadow/replay: not run for the V2 artifacts.  
Stateful A/B: not run for the V2 artifacts.  
Business metrics: unavailable by construction.  
Gate: offline research acceptance only.

The stateful Feed authority remains logistic regression. The V2 offline result
cannot be copied into the existing A/B report because that would compare a new
DGP offline model with an old-DGP policy environment. The next valid launch
must publish one V2 model artifact, score the real frozen candidate cascade,
replay the same feature manifest, and run fresh-user stateful A/B with Feed,
Local, and platform LT gates.

V2 artifact 尚未经过 shadow/replay 与有状态 A/B，因此不能上线，也不能把离线 AUC
差值冒充业务增量。下一条有效上线记录必须让同一 artifact 贯穿候选级联、特征回放、
fresh-user A/B 和 Feed/Local/platform LT 门禁。

## Evidence

- `reports/benchmarks/2026-08-23-signal-diagnosis.json`
- `reports/benchmarks/2026-08-23-model-v2-1m-gpu.json`
- `reports/benchmarks/2026-08-23-model-v2-10m-gpu.json`
- `docs/research/model-simulator-root-cause.md`
