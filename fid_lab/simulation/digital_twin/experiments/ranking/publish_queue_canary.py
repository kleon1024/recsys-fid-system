"""Run a memory-bounded expanded canary for a frozen Publish Queue scorer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import torch

from ...contracts import Surface
from ...engine import ExperimentPlan
from ...learning import ProbeArtifact, SparseLinearArtifact
from ...observability.store import replace_json_atomic
from ..launch_review.metrics import StreamingExperimentMetrics
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy
from .publish_queue_launch import _publish_decision


@dataclass(frozen=True)
class PublishQueueCanaryConfig:
    publish_checkpoint: str
    control_fine_checkpoint: str
    output: str
    users: int = 100_000
    items: int = 100_000
    burn_in_steps: int = 112
    experiment_steps: int = 96
    ticks_per_day: int = 16
    seed: int = 1_809
    device: str = "cuda"
    publish_weight: float = 0.12
    minimum_triggered_users: int = 30_000


def _load_artifact(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema = checkpoint.get("schema")
    if schema == "v4-hashed-sparse-fid-lr-v1":
        return SparseLinearArtifact.from_checkpoint(checkpoint)
    return ProbeArtifact.from_checkpoint(checkpoint)


def _run_steps(kernel, plan, logical_time, steps, accumulator=None) -> int:
    for _ in range(steps):
        tick = kernel.step(logical_time, plan)
        if accumulator is not None:
            accumulator.append(tick.entry_events)
            accumulator.append(tick.response_events)
        logical_time += 1
    return logical_time


def run_publish_queue_canary(
    config: PublishQueueCanaryConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    output = Path(config.output)
    runtime = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        ticks_per_day=config.ticks_per_day,
        seed=config.seed,
        device=config.device,
        response_authority_mode="formula_oracle",
        event_log_root=str(output / "event-log"),
    )
    _, kernel = _build_kernel(runtime)
    fine = ProbeArtifact.from_checkpoint(torch.load(
        config.control_fine_checkpoint, map_location="cpu", weights_only=True,
    ))
    publish = _load_artifact(config.publish_checkpoint)
    kernel.platform.install_fine_scorer(1, fine)
    kernel.platform.install_publish_scorer(1, publish)
    control = replace(
        _policy(
            "feed-random-popular-accepted-vt", 1, ("random", "popular"),
            config.ticks_per_day,
        ),
        fine_version_id=1,
    )
    treatment = replace(
        control,
        name="feed-publish-sparse-fid-canary",
        mix_version_id=control.mix_version_id + 1,
        publish_version_id=1,
        publish_weight=config.publish_weight,
    )
    baseline = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=control,
        experiment_seed=config.seed + 401,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    logical_time = _run_steps(
        kernel, baseline, 0, config.burn_in_steps,
    )
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=config.seed + 409,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    metrics = StreamingExperimentMetrics(config.users, config.device)
    analysis_start = logical_time
    logical_time = _run_steps(
        kernel, experiment, logical_time, config.experiment_steps, metrics,
    )
    estimates, sample = metrics.analyze()
    decision, reason = _publish_decision(
        estimates, sample, config.minimum_triggered_users,
    )
    report = {
        "schema": "feed-publish-queue-expanded-canary/v1",
        "quality_claim": "synthetic factual-world evidence only",
        "config": asdict(config),
        "artifact": {
            "model": publish.model_name,
            "source": config.publish_checkpoint,
        },
        "review": {
            "launch_review": "PUBLISH-LR-002",
            "analysis_start_time": analysis_start,
            "analysis_end_time": logical_time - 1,
            "changed_owner": "Feed Publish Queue score and mixer weight only",
            "sample": sample,
            "metrics_per_triggered_user": estimates,
            "decision": decision,
            "reason": reason,
        },
        "event_authority": kernel.event_log.manifest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(output / "report.json", report)
    return report
