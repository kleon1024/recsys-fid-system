# L-SIMULATOR-009 — Randomized V4 composite world authority

Change type: simulator architecture and evidence authority  
Decision: promote the Feed behavior kernel inside the simulator; hold the
unified NeuralSCM and every unsupported business surface  
Rollback: V3 simulator epoch

## What changed

The system no longer assumes that one DGP artifact can authorize Feed, POI,
posting supply, retention, and commercialization. World selection is now a
component manifest independent from the serving-policy manifest.

The external Feed lane uses 1,010,285 standard-policy training requests,
204,541 validation requests, and 43,027 random-exposure requests. Random users
are split between calibration and evaluation. The same candidate corpus,
Top-K, policy contract, and treatment artifact are used by DR/OPE and both
shadow worlds.

## Evidence

| Gate | Result |
|---|---:|
| Randomized DR/OPE stay delta | +0.000509; 95% CI [0.000360, 0.000659] |
| Shadow world seed 25 stay delta | +0.001602; 95% CI [0.001383, 0.001822] |
| Shadow world seed 27 stay delta | +0.002030; 95% CI [0.001763, 0.002297] |
| Click, long-view, like, hate guardrails | Pass in OPE and both shadows |
| Independent shadow worlds | Two distinct artifact hashes |
| Simulated A/B power | 1,000,000 users; paired truth recovered |
| Failed-seed evidence | Seed 31 adapter retained as Reject |

The shadows estimate 3.1--4.0 times the OPE magnitude. That discrepancy is not
hidden. Promotion uses consistent primary direction and bounded guardrails, not
exact agreement on effect size. Ranking utility is behavior value, not LT.

## Unified NeuralSCM failure review

The request-level Kuai bridge materializes eight candidates per request,
point-in-time histories, 28 features, explicit label masks, and separate random
calibration/evaluation users. On the RTX 4090, three ensemble members train for
eight epochs in 78.7 seconds.

Held-out calibration reduces stay p50 relative error from 158% to 8.5% and p90
error from 466% to 4.6%. Mask-aware uncertainty p99 falls from 0.115 to 0.055.
The challenger still fails joint-action calibration and sequence rollout. The
source history lacks historical stay, POI, supply, return, and commercialization
state, so mapping those fields to unrelated actions would be false evidence.

## Boundary

This is a simulator authority transition, not a production launch or live A/B.
Only Feed click, long view, like, comment, forward, follow, hate, and normalized
stay have external evidence. Local response and posting supply remain synthetic
V3 components. Retention and commercialization remain measurement-only. Unified
LT can be reported only by the final experiment using accepted exchange rates.

中文口径：V4 只上线到模拟器的 Feed 行为内核，不代表统一世界模型上线。POI、投稿供给、
留存和商业化没有外部随机证据，因此继续使用旧模拟组件或只做测量，不能补零、不能借标签、
也不能拿行为 utility 冒充 LT。
