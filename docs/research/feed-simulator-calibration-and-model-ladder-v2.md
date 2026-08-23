# Feed simulator calibration and model-ladder V2

## Finding

The main reason advanced models could not displace LR was not proven to be model
capacity. `L-MODEL-001` trained on 45,794 impressions from the legacy CPU world,
while the current Feed simulator uses an eight-route request-level candidate
graph. More importantly, the former GPU behavior RNG coupled independent event
streams and the long-view label used the wrong threshold. Model ranking under
that world is not transferable evidence.

The audited feature LR sequence contains 14 launches: three Pass, three Reject,
and eight Hold. The accepted historical feature chain is Basic + Realtime +
Local context + Category hash. Large early lifts of +1.2% and +3.2% came from
the old synthetic epoch; after candidate-graph repair, sparse post-search and
retarget effects shrank to basis-point scale and remained uncertain. These
numbers must not be placed on one continuous production-growth curve.

## External reference design

[KuaiSim](https://arxiv.org/abs/2309.12645) separates request-level
multi-behavior response, whole-session leave, and cross-session retention.
[RecSim NG](https://google-research.github.io/recsim_ng/) treats ecosystem state
as a probabilistic, learnable system and uses vectorized accelerator execution.
[KuaiRand](https://github.com/chongminggao/KuaiRand) provides 12 feedback
signals and random interventions; its Pure standard logs are used here only for
calibration because the accessible KuaiSim snapshot omits `log_random`.
[TorchRec](https://meta-pytorch.org/torchrec/overview.html) is the mature path
for jagged sparse features, embedding sharding, and distributed model training
when the single-4090 model ladder outgrows dense tensors.

## Current epoch and invariant

V3 is the first externally calibrated research epoch. Its invariant is that
control and treatment share the same response, leave, and return kernels. The
policy may change candidate ordering but cannot inject ground-truth response
logic. Raw public data stays outside Git; calibration reports bind hashes,
schema, license, and causal limitations.

## Next iteration

The next executable milestone is a V3 request-level logging dataset with real
exposures, propensities, all candidate-stage scores, point-in-time features,
behavior sequence, and mature multi-task labels. LR and all advanced models
must share that frozen dataset, split, candidate budget, and LT A/B world.

Model promotion requires all four layers:

1. Offline AUC, GAUC, PR-AUC, calibration, and slice coverage.
2. Shadow replay parity and stage-attribution closure.
3. Common-random V3 trajectory A/B with unified LT.
4. Runtime latency, throughput, memory, and rollback evidence.

HSTU or generative recommendation remains a research lane until the calibrated
short-sequence ladder proves that additional capacity is useful. Scaling model
complexity before fixing labels and the user-response kernel would only fit the
simulator defects more accurately.
