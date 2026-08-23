# Main Feed Foundation Launch Review — 2026-08-23

Gate note: the model decisions below preserve the original pre-unified-LT
review and are historical evidence only. Current launches use exchanged unified
LT plus independent safety, legal, privacy, and integrity constraints.

Status: main-Feed simulation foundation accepted; candidate model launches are
accepted or rejected independently. Local Service is out of scope for this gate.
All numbers below are deterministic synthetic evidence, not TikTok metrics.

## Launch contract

The evaluated loop is six-route recall, RRF merge, coarse Top 20, rank policy,
exposure, multi-action response, within-session state update, leave/return,
point-in-time Joiner, retraining, shadow replay, and user-level A/B. Actions are
play, 3-second play, slide, stay, long view, quality long view, like, favorite,
comment, share, and negative feedback. Platform LT is computed separately from
stay, active-day, and accepted commercialization. POI events are logged but do
not enter LT directly.

The semantic run used 300 logging users, 2,000 items, 4,345 main-Feed exposures,
fresh experiment users, and common-random-number potential trajectories. Its
main distribution was 94.84% play, 81.77% 3-second play, 13.07% slide, 31.14%
long view, 9.80% quality long view, 9.11% like, 4.14% favorite, and 0.35%
negative feedback. The analytic/sampled long-view probability gap was 0.68
percentage points.

## Iteration decisions

| Iteration | Offline/online result | Decision | Interpretation |
|---|---|---|---|
| LR fine rank | AUC 0.7175; candidate regret 0.0625 | Keep authority | The simplest model generalizes best on the current DGP and sample. |
| LR → W&D | stay truth -4.49%; quality view -24.11% | Reject | More capacity worsens held-out candidate ordering. |
| LR → DeepFM | stay truth -4.20%; quality view -23.26% | Reject | Automatic second-order crosses do not match this sample regime. |
| LR → DCNv2 | stay truth -4.23%; quality view -22.31% | Historical reject | Re-evaluate under unified LT. |
| LR → MMoE | quality view +5.40%, but stay -5.58% and platform LT -1.46% | Reject | One task head improves while overall user value regresses. |

The first model-ladder implementation also overloaded LT as long view and is
superseded. The corrected ladder uses exposure-normalized stay, long view,
quality long view, negative feedback, and the separate platform LT container.
W&D, DeepFM, and DCNv2 failed the historical quality-view gate in this run.
Those proxy metrics are now diagnostics unless their causal effect is exchanged
into LT.

The first deep model also embedded `user_id % 1024`. Because the experiment uses
fresh users, this mapped unseen users onto unrelated learned embeddings. Removing
raw UID from the cold-user model improved deep-model AUC but did not clear the
business gate. The durable follow-up is separate warm-user temporal evaluation
and cold-user OOV handling, not another hash tweak.

After correcting the authority to session 0 train, session 1 validation,
sessions 2–3 warm-user test, and a disjoint fresh-user A/B, the current MMoE
reaches GAUC 0.6823. It improves quality long view but has known stay -5.58%
and platform LT -1.46%, so it is rejected. LR remains the fine-rank authority
for this DGP and sample size.

## Scale and experiment evidence

On the Windows RTX 4090, the tensor engine ran 1,000,000 users for 24 sequential
steps. Depending on policy it produced 21.83-21.94 million exposures in
1.61-1.94 seconds: 11.24-13.52 million requests/second with about 95 MB peak
allocated GPU memory. User state, 20 candidates, response draws, and transitions
remain device-resident. This is a throughput/DGP benchmark, not proof of ranking
quality or C++ serving latency.

A scoped industrial-size tensor launch changes candidate choices and state
transitions for a small eligible population; it never injects an outcome lift.
Current scale evidence is reported in the Local, coarse, queue, and reverse
holdout Launch Reviews with long-view terminology separated from platform LT.

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
- long-view probability was aligned with the sampled stay distribution;
- total-stay-only gating was replaced by exposure-normalized multi-metric gates;
- inactive tensor users no longer count as plays;
- the CUPED target correlation no longer gets squared accidentally;
- a refactor variable no longer shadows the randomization-audit function;
- cold-user UID collision leakage was removed;
- candidate audit scoring was vectorized instead of issuing thousands of tiny GPU calls.

Open before any advanced model can pass:

- build repeated-user chronological train/validation/test periods and report warm/cold slices;
- calibrate long-view, quality-view, negative, and retention heads separately;
- handle the sparse negative task with an explicit sampling/masking protocol;
- connect orthogonal parameter snapshots to every actual cascade stage and log them per request;
- validate semantic and tensor-engine distributions with sliced statistical parity tests;
- add a small-effect launch whose 0.1%-1% ITT comes from a scoped policy change, not an injected outcome shift.

No Local Service launch should resume until those main-Feed items pass.
