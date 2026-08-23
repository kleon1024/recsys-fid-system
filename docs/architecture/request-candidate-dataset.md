# Request-level candidate dataset / 请求级候选数据集

The training and Launch Review authority is now a closed request graph, not a
flat impression row. Each request retains the complete recalled pool, coarse
decision, fine score, current mixing decision, point-in-time history, and mature
labels. The semantic audit exports three partitionable JSONL tables; the GPU
training snapshot stores the same closure as time-split tensor partitions.

训练与上线复盘的 authority 已从扁平曝光样本改为闭合的 request graph。每次请求保留
完整召回池、粗排决策、精排分数、当前混排决策、时点历史和成熟标签；物理上仍拆成
三张逻辑表。语义审计导出 JSONL，GPU 训练快照保存为按时间切分的 tensor partition；
两者不能互相伪装成同一个物理格式。

| Table | Grain | Required closure |
|---|---|---|
| `requests` | one row per request | user, time, experiment snapshot, manifest, PIT sequence |
| `candidate_decisions` | request + candidate | routes, recall/coarse/fine/mix scores and ranks, exposed position or filter reason |
| `mature_labels` | request + candidate + POI | labels, task masks, exchanged LT components |

The GPU candidate row now stores the actual stage outputs, not reconstructed
proxies: `recall_scores`, `candidate_coarse_scores`,
`candidate_coarse_mask`, `candidate_fine_scores`, `candidate_mix_scores`, and
the exposed index. The invariant is `48 recalled -> 20 coarse survivors -> 1
exposure` under the default budget. Recall/RRF cannot truncate the pool or
write a coarse score.

GPU 候选行直接保存各阶段真实输出，默认闭包是 `48 个召回合并候选 → 20 个粗排通过
候选 → 1 个曝光`。召回/RRF 不允许截断粗排预算，也不允许生成一份伪粗排分数。

Unexposed or immature outcomes have `label_mask=false`; they are never written
as negatives. Fine features are hydrated only for coarse survivors. This is
intentional stage behavior, not missing data.

未曝光或未成熟行为统一写 `label_mask=false`，不得伪造负样本。只有通过粗排的候选才
拥有精排特征，这是阶段化 hydration，而不是数据缺失。

```text
oracle item absent from recalled pool  -> recall_miss
present after recall but removed       -> coarse_miss
coarse survivor ranked below exposure  -> fine_rank_miss
fine rank changed before exposure      -> mix_rank_miss
oracle item exposed                    -> served_oracle
```

The historical 500-user semantic run produced 1,590 requests, 159,000
candidate decisions, and 159,000 label rows. Attribution was 1,269 recall misses,
174 coarse misses, 119 fine-rank misses, zero mix misses, and 28 served-oracle
requests. That historical run had no independent mixing stage. The current GPU
cascade does: fine choice is frozen before Local/Live/Ads mixing, and a changed
final choice is attributed to `mix_rank_miss`.

历史 500 用户验证产生 1,590 个请求、159,000 条候选决策和同数 label 行。归因结果为
召回缺失 1,269、粗排丢失 174、精排错误 119、混排错误 0、正确命中 28。当前尚未接入
独立混排，因此当时 mix 为 0。当前 GPU 级联已先冻结精排选择，再执行 Local、Live、
Ads 混排；最终候选被改变时单独归因到 `mix_rank_miss`。

The audit also found and fixed a real cross-request logging defect: route and
recall metadata were read after `environment.step()`, so they could belong to
the next request while item/features/scores belonged to the current request.
The dataset now fails closed unless every coarse candidate belongs to the same
request's recalled pool.

This dataset is necessary but not sufficient for launch attribution. The
release controller must also compare every proposal with the last accepted
control. A held or rejected proposal cannot become the control of the next A/B.
The checked sequence is now:

```text
Basic --Sequence/Hold--> Basic
Basic --Realtime/Pass--> Basic + Realtime
Basic + Realtime --Local/Pass--> Basic + Realtime + Local
Basic + Realtime + Local --Hash/Reject--> Basic + Realtime + Local
```

`artifacts/releases/simulated-feed-control.json` binds the final active model,
rollback model, exact Launch Review hash, feature schema, and model artifact
hash. It records simulator state only; synthetic LT exchange rates still block
a production-readiness claim.

```bash
python3 -m fid_lab.evolution.data.request_dataset_cli \
  --users 100 --items 2000 --candidates 20 \
  --output /tmp/request-candidate-v1
```
