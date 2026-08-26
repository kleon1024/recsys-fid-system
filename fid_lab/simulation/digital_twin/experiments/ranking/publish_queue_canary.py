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
from ...profile import STANDARD_FEED_PROFILE
from ..launch_review.metrics import StreamingExperimentMetrics
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy
from .publish_queue_launch import _publish_decision


@dataclass(frozen=True)
class PublishQueueCanaryConfig:
    publish_checkpoint: str
    control_fine_checkpoint: str
    output: str
    users: int = 100_000
    items: int = 1_000_000
    burn_in_steps: int = 112
    experiment_steps: int = 96
    ticks_per_day: int = 16
    seed: int = 1_809
    device: str = "cuda"
    publish_weight: float = 0.12
    minimum_triggered_users: int = 30_000
    cuda_memory_fraction: float = 0.60
    minimum_wsl_available_gib: float = 4.0
    minimum_cuda_free_gib: float = 2.0

    def __post_init__(self) -> None:
        if self.items * STANDARD_FEED_PROFILE.users < (
            self.users * STANDARD_FEED_PROFILE.items
        ):
            raise ValueError(
                "expanded canary must preserve the standard item/user ratio"
            )
        if not 0.0 < self.cuda_memory_fraction <= 1.0:
            raise ValueError("CUDA memory fraction must be in (0, 1]")
        if min(
            self.minimum_wsl_available_gib,
            self.minimum_cuda_free_gib,
        ) <= 0.0:
            raise ValueError("runtime memory guards must be positive")


def _memory_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("WSL MemAvailable is unavailable")


def _check_runtime_pressure(config: PublishQueueCanaryConfig) -> None:
    available = _memory_available_gib()
    if available < config.minimum_wsl_available_gib:
        raise RuntimeError(
            f"WSL memory guard tripped at {available:.2f} GiB available"
        )
    if torch.device(config.device).type != "cuda":
        return
    device = torch.device(config.device)
    device_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    free, _ = torch.cuda.mem_get_info(device_index)
    free_gib = free / 2**30
    if free_gib < config.minimum_cuda_free_gib:
        raise RuntimeError(
            f"CUDA memory guard tripped at {free_gib:.2f} GiB free"
        )


def _load_artifact(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema = checkpoint.get("schema")
    if schema == "v4-hashed-sparse-fid-lr-v1":
        return SparseLinearArtifact.from_checkpoint(checkpoint)
    return ProbeArtifact.from_checkpoint(checkpoint)


def _run_steps(
    kernel, plan, logical_time, steps, config, accumulator=None,
) -> int:
    for _ in range(steps):
        _check_runtime_pressure(config)
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
    device = torch.device(config.device)
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if device.index is None else device.index
        )
        torch.cuda.set_per_process_memory_fraction(
            config.cuda_memory_fraction, device_index,
        )
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
        kernel, baseline, 0, config.burn_in_steps, config,
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
        kernel, experiment, logical_time, config.experiment_steps,
        config, metrics,
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
