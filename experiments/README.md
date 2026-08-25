# Registered recommendation experiments

This directory is the pre-treatment authority for every new Launch Review.
Running a CLI with an arbitrary user count is prohibited. The current factual
authority is one continuous `feed-standard-v1` world on the RTX 4090. Diagnostic
smoke and scale runs cannot advance its `main` branch.

```text
registered LaunchSpec
→ restore the current main checkpoint
→ stable user assignment with default traffic on the active policy
→ append factual requests, events and mature samples
→ request-aware replay, support, SRM and clustered A/B
→ pass / hold / reject
→ advance the world; change active policy only on pass
```

Each plan names its predecessor and is hash-bound before evidence. If a model,
feature, candidate budget, DGP, label horizon or Value Tree changes, it is a new
launch id rather than a continuation. Historical multi-salt plans remain legacy
diagnostics and cannot advance the continuous v4 world.

中文口径：规模不是越大越可信。正式 LR 只在同一个连续世界累计流量和时间，性能
benchmark 不能推进事实基线。任何看完 treatment effect 才修改 MDE、流量或窗口的
做法都不能产生上线结论。
