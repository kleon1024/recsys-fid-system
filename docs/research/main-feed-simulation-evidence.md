# Main Feed Simulation Evidence and Scale Design

This repository simulates a public, TikTok-style short-video Feed. It does not
claim access to TikTok's proprietary labels, ranking weights, or production
architecture. Synthetic LT and HLT are explicit benchmark proxies.

## Evidence boundary

TikTok publicly describes watch completion, likes, shares, follows, comments,
content information, and lower-weight device/account signals as recommendation
inputs. It also describes diversity constraints and cold-start/popular content.
Those observations justify the simulated action vocabulary and a multi-route
cascade, but not any proprietary weight or threshold.

Monolith motivates dynamic sparse embeddings, online training, expiry/frequency
filtering, and fault-tolerant real-time learning. KuaiSim motivates three
distinct simulation levels: request/list response, within-session sequential
response, and cross-session retention. SARDINE motivates explicit latent-state
dynamics, biased logged data, slate effects, and uncertainty. RecSim NG further
shows why population state should be expressed as batched tensors on accelerated
hardware rather than one Python object per user.

Simulation results are pre-online evidence only. KuaiSim explicitly notes that
simulator superiority is useful only when simulator behavior is consistent with
real data. Therefore every launch record must separate known DGP truth,
estimated offline metrics, shadow/replay parity, and randomized A/B estimates.

## Execution architecture

The simulator has three execution tiers with one shared contract:

1. Semantic tier: 100-10,000 stateful Gymnasium/SARDINE-style trajectories.
   It validates action order, feature point-in-time semantics, delayed labels,
   session exit/return, candidate closure, and debuggable failure cases.
2. Throughput tier: 100,000-10,000,000 tensorized trajectories on RTX 4090.
   User state, candidate features, policy scores, response draws, and state
   transitions remain in GPU memory. No per-user Python loop is allowed.
3. Experiment tier: 1,000,000+ vectorized potential outcomes. It validates
   stable assignment, overlapping experiment layers, interaction audits,
   confidence intervals, CUPED, minimum detectable effect, and 0.1%-1% ITT.

The GPU tier should follow WarpDrive's core performance principle: simulations
and agents run in parallel against one device-resident state store, avoiding
CPU/GPU copies. Rust is conditional, not foundational. Add a Rust/PyO3 kernel
only when profiling shows stable hashing, RRF merge, event encoding, or Joiner
work consumes more than 30% of end-to-end runtime after NumPy/Torch batching.

## Orthogonal A/B parameters

Google's overlapping-experiment design partitions mutually dependent parameters
into layers. Experiments are mutually exclusive inside a layer and orthogonal
across layers. This repository follows that rule. Suggested layers are recall,
coarse rank, fine rank, Value Tree/calibration, product policy, realtime/feature,
and infrastructure. One user may join one experiment in every layer, but never
two conflicting experiments in the same layer.

Every request log must carry the assigned layer/variant map plus the resolved
full-chain parameter snapshot. Analysis must include marginal effects,
predeclared interaction slices for overlapping launches, Sample Ratio Mismatch,
trigger rate, exposure rate, model/feature manifest parity, and guardrails.
Changing a model name without freezing all other parameters is not a model A/B.

## Acceptance bars

- Semantic and tensor tiers match action rates and transition distributions by
  sliced two-sample checks; fixed seeds must replay the same manifest.
- Synthetic generator parameters are fitted or calibrated to an explicit public
  dataset or a clearly labeled scenario, never tuned to make a candidate win.
- Offline AUC/GAUC gains are not launch evidence. A model must pass candidate
  replay, calibration, Top-K overlap/value, and randomized trajectory A/B.
- Per-mille results require a reported confidence interval, power/MDE, CUPED
  variance reduction, SRM, and effect heterogeneity. A true DGP effect is an
  audit oracle and must never be exposed to the policy as a feature.
- Performance is accepted only with measured users/s, requests/s, GPU memory,
  and CPU/GPU time split. Rust is adopted only against this benchmark.

## Primary sources

- TikTok, [How TikTok recommends videos for you](https://newsroom.tiktok.com/how-tiktok-recommends-videos-for-you?lang=en)
- ByteDance, [Monolith](https://arxiv.org/abs/2209.07663)
- Kuaishou et al., [KuaiSim](https://openreview.net/pdf?id=dJEjgQcbOt)
- NAVER, [SARDINE](https://arxiv.org/abs/2311.16586)
- Google, [RecSim NG](https://arxiv.org/abs/2103.08057)
- Google, [Overlapping Experiment Infrastructure](https://research.google.com/pubs/archive/36500.pdf)
- Salesforce, [WarpDrive](https://jmlr.org/papers/volume23/22-0185/22-0185.pdf)
- Microsoft, [CUPED variance reduction](https://www.microsoft.com/en-us/research/articles/deep-dive-into-variance-reduction/)
