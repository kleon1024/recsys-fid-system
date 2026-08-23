# L-COARSE-001 — Equal-pool coarse pass-through ladder

All launches use 100 recalled candidates and the same
`local_intent_quality_rank_v4` fine ranker. Only coarse scoring or Top-K changes.

所有实验固定 100 个召回候选和同一个 `local_intent_quality_rank_v4` 精排模型，
只改变粗排打分或 Top-K。

| Change / 变更 | Oracle recall | Fine regret | Platform LT | Stay | Local tree | Decision |
|---|---:|---:|---:|---:|---:|---|
| Quality-only Top-20 → LR-style Top-20 | 65.3% → 99.9% | 0.0685 → 0.0361 | +3.79% | +7.26% | +33.57% | Pass weak-baseline repair |
| LR-style → Local-aware cross Top-20 | -0.002% relative | +0.067% | +0.040%, p=.628 | -0.002% | +0.025% | Hold |
| Local-aware Top-20 → Top-40 | +0.080% relative | +0.066% | +0.040%, p=.628 | -0.002% | +0.018% | Hold |

The first gain repairs a deliberately weak quality-only baseline and must not be
presented as mature-system lift. Once oracle pass-through reaches 99.9%, Local
crosses and twice the coarse budget do not improve platform LT. Top-40 doubles
pass fraction from 20% to 40% while exposing more candidates to fine-rank error.

第一次收益来自修复刻意设置的弱 quality-only baseline，不能冒充成熟系统收益。
当 oracle 通过率达到 99.9% 后，Local cross 与两倍粗排预算都没有提升平台 LT；
Top-40 将通过率从 20% 提高到 40%，同时让更多候选暴露给精排误差。

RTX 4090 throughput is 1.77M–1.84M simulated requests/s for 100 candidates,
with 279 MiB peak allocated memory. This is simulation throughput, not online
C++ service latency.

Numeric authority / 数值权威：
`reports/launches/2026-08-23-coarse-cascade-ladder.json`.
