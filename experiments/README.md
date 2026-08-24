# Registered recommendation experiments

This directory is the pre-treatment authority for every new Launch Review.
Running a CLI with an arbitrary user count is prohibited.

```text
registered smoke plan
→ one 100k-user salt
→ immutable smoke report
→ registered three-salt screen plan bound to that report
→ screen aggregate
→ registered powered plan sized from control variance and business MDE
→ powered aggregate
→ pass / hold / reject
```

Each later phase contains the path and SHA-256 of its predecessor. It must keep
the same control, treatment and scenario fingerprints. If a model, feature,
candidate budget, DGP, label horizon or Value Tree changes, it is a new launch
id rather than a continuation.

中文口径：规模不是越大越可信。Smoke、screen、powered A/B 和性能 benchmark
拥有不同证据权限。任何单盐结果都是 `partial_evidence`；任何看完 treatment effect
才决定扩到 1000 万或 1 亿的做法都不能产生上线结论。
