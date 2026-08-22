# Main Feed Foundation Launch Review — 2026-08-23

Status: main-Feed simulation foundation accepted; candidate model launches are
accepted or rejected independently. Local Service is out of scope for this gate.
All numbers below are deterministic synthetic evidence, not TikTok metrics.

## Launch contract

The evaluated loop is six-route recall, RRF merge, coarse Top 20, rank policy,
exposure, multi-action response, within-session state update, leave/return,
point-in-time Joiner, retraining, shadow replay, and user-level A/B. Actions are
play, 3-second play, slide, stay, LT, HLT, like, favorite, comment, share, and
negative feedback. POI events are logged but do not decide this review.

The semantic run used 300 logging users, 2,000 items, 4,345 main-Feed exposures,
fresh experiment users, and common-random-number potential trajectories. Its
main distribution was 94.84% play, 81.77% 3-second play, 13.07% slide, 31.14%
LT, 9.80% HLT, 9.11% like, 4.14% favorite, and 0.35% negative feedback. The
analytic/sampled LT probability gap was 0.68 percentage points.

## Iteration decisions

| Iteration | Offline/online result | Decision | Interpretation |
|---|---|---|---|
| Popular → quality/affinity rule | true stay/exposure +5.74%; observed p=0.027 | Pass initial baseline | A deliberately weak cold-start baseline leaves large algorithmic headroom. |
| Rule → basic LR | true stay/exposure +3.18%; observed underpowered; HLT risk | Hold | LR demonstrates algorithmic impact, but this sample cannot clear the HLT guardrail. |
| Basic LR → LR + sequence | true stay/exposure -1.57%; long-term value negative | Hold | Adding a feature is not automatically useful; current short history is noisy. |
| Sequence LR → full LR | true stay/exposure -7.04% | Hold | Extra feature surface changes candidate ordering without improving value. |
| LR → XGBoost | true stay/exposure -2.73%; observed p=0.046 | Reject | Nonlinear offline fit does not compensate for policy/state distribution shift. |

The first model-ladder implementation incorrectly treated total per-user stay as
the sole launch metric. W&D/DCNv2 appeared to gain roughly 15%-17% total stay
while HLT and long-term value regressed. The gate now uses exposure-normalized
stay, LT/HLT rates, negative feedback, and long-term value. Re-evaluation rejects
W&D, DeepFM, and DCNv2 on the HLT guardrail.

The first deep model also embedded `user_id % 1024`. Because the experiment uses
fresh users, this mapped unseen users onto unrelated learned embeddings. Removing
raw UID from the cold-user model improved deep-model AUC but did not clear the
business gate. The durable follow-up is separate warm-user temporal evaluation
and cold-user OOV handling, not another hash tweak.

The initial MMoE run used a record-index split that was incorrectly described as
temporal and was rejected. After correcting the authority to session 0 train,
session 1 validation, sessions 2-3 warm-user test, plus a disjoint fresh-user
A/B, MMoE warm GAUC reached 0.668. In fresh-user DGP truth it produced
stay/exposure +2.89%, LT +5.17%, HLT +7.66%, long-term Value +14.25%, and lower
negative feedback. The observed stay p-value was 0.0038 and shadow replay delta
was zero. Decision: pass the synthetic initial-model gate. This is not a claim
that an advanced model should generate a multi-percent mature production lift.

## Scale and experiment evidence

On the Windows RTX 4090, the tensor engine ran 1,000,000 users for 24 sequential
steps. Depending on policy it produced 21.83-21.94 million exposures in
1.61-1.94 seconds: 11.24-13.52 million requests/second with about 95 MB peak
allocated GPU memory. User state, 20 candidates, response draws, and transitions
remain device-resident. This is a throughput/DGP benchmark, not proof of ranking
quality or C++ serving latency.

A scoped industrial-size launch then enabled personalized ranking for only 1%
of users inside treatment. With stable 50/50 user assignment over one million
users, the observed ITT was stay/exposure +0.745% (absolute 95% CI +0.0223 to
+0.0291 seconds), LT rate +2.12%, HLT rate +2.31%, and negative rate -0.72%
(not significant). The model effect comes from changed candidate choices and
state transitions; it was not added to the outcome. The tiny stay p-value is a
consequence of scale and must not replace effect-size and guardrail review.

The vectorized A/B layer uses one million users, stable 50/50 assignment, known
potential outcomes, and CUPED with a 0.65 pre/post correlation. CUPED reduced
variance by 42.1%. In the fixed-seed run, a 0.1% effect remained underpowered;
0.3% was marginally detectable (p=0.038), while 0.5% and 1.0% were detected.
This layer verifies estimator power. It does not manufacture a model effect.

The experiment authority supports mutually exclusive experiments within one
layer and orthogonal assignment across recall, coarse, fine, Value Tree,
product, realtime-feature, and infrastructure layers. Each assignment resolves
to one complete parameter snapshot including route budgets, model ids,
calibration, Value Tree weights, diversity, exploration, feature snapshot, and
model manifest.

The vectorized stable-hash implementation assigned 10 million users in 0.259
seconds on the local CPU (38.6 million assignments/second). The two 25% cells
received 24.997% and 24.996%; the remaining 50.007% stayed on layer default.

## Failures and next launch

Resolved simulator failures:

- hidden true affinity was removed from online features;
- LT probability was aligned with the sampled stay distribution;
- total-stay-only gating was replaced by exposure-normalized multi-metric gates;
- inactive tensor users no longer count as plays;
- the CUPED target correlation no longer gets squared accidentally;
- a refactor variable no longer shadows the randomization-audit function;
- cold-user UID collision leakage was removed;
- candidate audit scoring was vectorized instead of issuing thousands of tiny GPU calls.

Open before any advanced model can pass:

- build repeated-user chronological train/validation/test periods and report warm/cold slices;
- calibrate LT, HLT, negative, and retention heads separately;
- handle the sparse negative task with an explicit sampling/masking protocol;
- connect orthogonal parameter snapshots to every actual cascade stage and log them per request;
- validate semantic and tensor-engine distributions with sliced statistical parity tests;
- add a small-effect launch whose 0.1%-1% ITT comes from a scoped policy change, not an injected outcome shift.

No Local Service launch should resume until those main-Feed items pass.
