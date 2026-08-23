# Request-level candidate dataset / 请求级候选数据集

The training and Launch Review authority is now a closed request graph, not a
flat impression row. Each request retains the complete recalled pool, coarse
decision, fine score, current mixing decision, point-in-time history, and mature
labels. Physical storage remains three partitionable JSONL tables.

训练与上线复盘的 authority 已从扁平曝光样本改为闭合的 request graph。每次请求保留
完整召回池、粗排决策、精排分数、当前混排决策、时点历史和成熟标签；物理上仍拆成
三张可分区 JSONL 表。

| Table | Grain | Required closure |
|---|---|---|
| `requests` | one row per request | user, time, experiment snapshot, manifest, PIT sequence |
| `candidate_decisions` | request + candidate | routes, recall/coarse/fine/mix scores and ranks, exposed position or filter reason |
| `mature_labels` | request + candidate + POI | labels, task masks, exchanged LT components |

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

The first checked 500-user stateful run produced 1,590 requests, 159,000
candidate decisions, and 159,000 label rows. Attribution was 1,269 recall misses,
174 coarse misses, 119 fine-rank misses, zero mix misses, and 28 served-oracle
requests. Zero mix misses is expected because this run has no independent
mixing stage yet; the schema prevents that missing stage from being blamed on
fine rank.

首个 500 用户验证产生 1,590 个请求、159,000 条候选决策和同数 label 行。归因结果为
召回缺失 1,269、粗排丢失 174、精排错误 119、混排错误 0、正确命中 28。当前尚未接入
独立混排，因此 mix 必须为 0，不能把不存在的阶段伪装成精排问题。

The audit also found and fixed a real cross-request logging defect: route and
recall metadata were read after `environment.step()`, so they could belong to
the next request while item/features/scores belonged to the current request.
The dataset now fails closed unless every coarse candidate belongs to the same
request's recalled pool.

```bash
python3 -m fid_lab.evolution.data.request_dataset_cli \
  --users 100 --items 2000 --candidates 20 \
  --output /tmp/request-candidate-v1
```
