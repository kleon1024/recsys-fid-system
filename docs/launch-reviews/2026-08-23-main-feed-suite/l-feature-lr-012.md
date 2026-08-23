# F-LR-012 — Remove geo and Local-interest signals

Decision: `reject_unified_lt_negative`

Control: Basic + Realtime + Local + Category

Removing `distance_score` and `local_interest_affinity` reduced offline AUC
from 0.74459 to 0.73307. Unified LT fell 3.073%; its absolute confidence
interval was [-0.41916, -0.36064]. Stay fell 7.884%, quality-long-view fell
14.53%, negative feedback rose 17.21%, and online oracle regret rose 190.15%.

Geo and Local-interest are the load-bearing Local features in this DGP. 去掉后
离线、候选选择和最终 LT 同时恶化，因此明确 Reject；active 与 rollback 均保持不变。
