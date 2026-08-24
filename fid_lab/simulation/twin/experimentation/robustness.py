"""One fixed model artifact evaluated across unseen hidden environments."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from ..contracts import BASELINE_POLICY, TwinConfig
from ..kernel import DigitalTwinKernel
from ..serving.models import ServingStack
from ..training.materialize import join_training_authorities, materialize_events
from ..training.ranker import RankerArtifact, train_fine_ranker
from .campaign import launch_decision
from .experiment import evaluate_from_preperiod


def _artifact_fingerprint(artifact: RankerArtifact) -> str:
    digest = sha256()
    for name, value in sorted(artifact.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_source_artifact(
    config: TwinConfig,
    architecture: str,
    source_steps: int,
    maximum_requests: int,
) -> tuple[RankerArtifact, dict[str, object]]:
    kernel = DigitalTwinKernel(config)
    logging = replace(
        BASELINE_POLICY,
        name="heldout-source-exploration",
        exploration_rate=0.10,
    )
    run = kernel.run(
        kernel.initialize(), logging, source_steps,
        trace_limit=config.training_trace_users,
    )
    trace = run.trace.sampled(maximum_requests, salt=config.environment_seed)
    events = materialize_events(
        trace,
        world_version=config.version,
        served_policy=logging.name,
        experiment_cell="source-training-world",
        watermark_step=run.snapshot.step,
    )
    samples = join_training_authorities(events)
    artifact = train_fine_ranker(
        samples.fine,
        model_id=f"heldout-{architecture}-{config.environment_seed}",
        architecture=architecture,
        epochs=3,
        microbatch_rows=4_096,
        device=kernel.device,
        seed=config.seed,
    )
    return artifact, samples.manifest()


def _evaluate_environment(
    base_config: TwinConfig,
    environment_seed: int,
    artifact: RankerArtifact,
    model_weight: float,
    salt: int,
) -> dict[str, object]:
    config = replace(base_config, environment_seed=environment_seed)
    kernel = DigitalTwinKernel(config)
    control = ServingStack(BASELINE_POLICY)
    candidate = ServingStack(
        BASELINE_POLICY,
        fine_model=artifact,
        fine_model_weight=model_weight,
    )
    preperiod = kernel.preperiod(control)
    experiment = evaluate_from_preperiod(
        config, kernel, preperiod, control, candidate, salt,
        trace_limit=config.audit_users,
    )
    decision, gates = launch_decision(experiment)
    return {
        "environment_seed": environment_seed,
        "decision": decision,
        "gates": gates,
        "primary": experiment.report["cuped_ab"][
            "synthetic_lt_measurement"
        ],
        "stay": experiment.report["cuped_ab"]["stay_seconds"],
        "negative": experiment.report["cuped_ab"]["negative"],
        "requests": experiment.report["cuped_ab"]["requests"],
        "trace_gates": experiment.report["trace"]["gates"],
    }


def run_heldout_environment_gate(
    config: TwinConfig,
    *,
    architecture: str,
    heldout_environment_seeds: tuple[int, ...],
    source_steps: int = 20,
    maximum_training_requests: int = 65_536,
    model_weight: float = 0.30,
    salt: int = 0x4CF5AD43,
) -> dict[str, object]:
    if not heldout_environment_seeds:
        raise ValueError("at least one held-out environment is required")
    if config.environment_seed in heldout_environment_seeds:
        raise ValueError("source environment cannot be a held-out environment")
    artifact, samples = _train_source_artifact(
        config, architecture, source_steps, maximum_training_requests
    )
    evaluations = [
        _evaluate_environment(
            config, seed, artifact, model_weight,
            salt + index * 104_729,
        )
        for index, seed in enumerate(heldout_environment_seeds)
    ]
    decisions = [row["decision"] for row in evaluations]
    aggregate = (
        "pass" if all(value == "pass" for value in decisions)
        else "reject" if any(value == "reject" for value in decisions)
        else "hold"
    )
    return {
        "schema": "heldout-hidden-environment-gate-v1",
        "source_environment_seed": config.environment_seed,
        "heldout_environment_seeds": list(heldout_environment_seeds),
        "artifact_fingerprint": _artifact_fingerprint(artifact),
        "artifact": artifact.manifest(),
        "source_samples": samples,
        "model_weight": model_weight,
        "evaluations": evaluations,
        "aggregate_decision": aggregate,
        "invariant": (
            "The exact source-trained checkpoint is replayed without fitting "
            "or calibration in every held-out hidden environment."
        ),
        "evidence_boundary": "Synthetic robustness gate; not production lift.",
    }
