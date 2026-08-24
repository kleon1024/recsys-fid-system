"""Continuous events-to-training-to-shadow-to-A/B learning loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import torch

from ..contracts import BASELINE_POLICY, TwinConfig, TwinPolicy
from ..experimentation.campaign import launch_decision
from ..experimentation.experiment import evaluate_from_preperiod
from ..kernel import DigitalTwinKernel
from ..serving.models import ServingStack
from .buffer import TraceWindow
from .materialize import join_training_authorities, materialize_events
from .ranker import train_fine_ranker
from .registry import ModelRegistry


@dataclass(frozen=True)
class ContinuousLearningConfig:
    iterations: int = 3
    logging_steps: int = 4
    sample_lookback_steps: int = 32
    architectures: tuple[str, ...] = (
        "lr", "wide_deep", "deepfm", "dcnv2", "mmoe"
    )
    fine_model_weights: tuple[float, ...] = (0.10, 0.20, 0.30)
    train_epochs: int = 3
    microbatch_rows: int = 4_096
    max_training_window_requests: int = 65_536
    exploration_rate: float = 0.10

    def __post_init__(self):
        if self.iterations < 1 or self.logging_steps < 1:
            raise ValueError("continuous learning requires positive windows")
        if len(self.architectures) < self.iterations:
            raise ValueError("one candidate architecture is required per iteration")
        if len(self.fine_model_weights) < self.iterations:
            raise ValueError("one model blend weight is required per iteration")
        if not 0.0 < self.exploration_rate <= 0.25:
            raise ValueError("logging exploration rate must be in (0, 0.25]")
        if self.max_training_window_requests < 1:
            raise ValueError("training window request budget must be positive")


def _add_observed_experiment_traces(store, experiment):
    store.add(experiment.preperiod.trace)
    store.add(experiment.mixed.traces["control"])
    store.add(experiment.mixed.traces["treatment"])


def _logging_stack(active, exploration_rate):
    strategy = replace(
        active.strategy,
        name=f"{active.strategy.name}+exploration-log",
        exploration_rate=exploration_rate,
    )
    return ServingStack(
        strategy=strategy,
        coarse_model=active.coarse_model,
        fine_model=active.fine_model,
        coarse_model_weight=active.coarse_model_weight,
        fine_model_weight=active.fine_model_weight,
    )


def _iteration_report(
    index, world_before, watermark, world, artifact, registered,
    decision, gates, active, samples, store, experiment,
):
    return {
        "iteration": index + 1,
        "world_step_before": world_before,
        "training_watermark": watermark,
        "world_step_after": world.step,
        "candidate_model": artifact.manifest(),
        "registry_version": registered.version,
        "decision": decision,
        "gates": gates,
        "active_after": active.name,
        "samples": samples.manifest(),
        "event_window": store.manifest(world.step),
        "ab_primary": experiment.report["cuped_ab"][
            "synthetic_lt_measurement"
        ],
        "ab_stay": experiment.report["cuped_ab"]["stay_seconds"],
        "ab_negative": experiment.report["cuped_ab"]["negative"],
        "ab_requests": experiment.report["cuped_ab"]["requests"],
        "sample_evolution": experiment.report["sample_evolution"],
        "ecosystem_interference": experiment.report["ecosystem_interference"],
    }


def run_continuous_learning(
    twin_config: TwinConfig,
    learning: ContinuousLearningConfig = ContinuousLearningConfig(),
    initial_strategy: TwinPolicy = BASELINE_POLICY,
    salt: int = 0x6A09E667,
) -> dict[str, object]:
    started = perf_counter()
    kernel = DigitalTwinKernel(twin_config)
    if kernel.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(kernel.device)
    world = kernel.initialize()
    active = ServingStack(initial_strategy)
    registry = ModelRegistry()
    store = TraceWindow(learning.sample_lookback_steps)
    iterations = []
    for index in range(learning.iterations):
        world_before = world.step
        logging_run = kernel.run(
            world,
            _logging_stack(active, learning.exploration_rate),
            learning.logging_steps,
            trace_limit=twin_config.training_trace_users,
        )
        store.add(logging_run.trace)
        watermark = logging_run.snapshot.step
        events = materialize_events(
            store.at(
                watermark, learning.max_training_window_requests
            ),
            world_version=twin_config.version,
            served_policy="multi-policy-event-window",
            experiment_cell="observed-online-traffic",
            watermark_step=watermark,
        )
        samples = join_training_authorities(events)
        architecture = learning.architectures[index]
        artifact = train_fine_ranker(
            samples.fine,
            model_id=f"fine-{architecture}-iteration-{index + 1}",
            architecture=architecture,
            epochs=learning.train_epochs,
            microbatch_rows=learning.microbatch_rows,
            device=kernel.device,
            seed=twin_config.seed + index,
        )
        registered = registry.register("fine", artifact)
        registry.shadow(registered.version)
        candidate = ServingStack(
            strategy=active.strategy,
            coarse_model=active.coarse_model,
            fine_model=artifact,
            fine_model_weight=learning.fine_model_weights[index],
        )
        preperiod = kernel.preperiod_from(logging_run.snapshot, active)
        experiment = evaluate_from_preperiod(
            twin_config,
            kernel,
            preperiod,
            active,
            candidate,
            salt + index * 104_729,
            trace_limit=twin_config.training_trace_users,
        )
        decision, gates = launch_decision(experiment)
        if decision == "pass":
            registry.promote(registered.version)
            active = candidate
        elif decision == "reject":
            registry.reject(registered.version)
        world = experiment.mixed.snapshot
        _add_observed_experiment_traces(store, experiment)
        store.prune(world.step)
        iterations.append(_iteration_report(
            index, world_before, watermark, world, artifact, registered,
            decision, gates, active, samples, store, experiment,
        ))
    return {
        "schema": "continuous-learning-digital-twin-v1",
        "invariant": (
            "Only observed mixed-world traffic enters later training windows; "
            "shadow full-rollout trajectories never become factual samples."
        ),
        "config": twin_config.manifest(),
        "learning": {
            "iterations": learning.iterations,
            "logging_steps": learning.logging_steps,
            "sample_lookback_steps": learning.sample_lookback_steps,
            "architectures": list(learning.architectures),
            "fine_model_weights": list(learning.fine_model_weights),
            "exploration_rate": learning.exploration_rate,
            "max_training_window_requests": (
                learning.max_training_window_requests
            ),
        },
        "iterations": iterations,
        "registry": registry.manifest(),
        "active_stack": active.manifest(),
        "final_world_step": world.step,
        "performance": {
            "seconds": perf_counter() - started,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(kernel.device)
                if kernel.device.type == "cuda" else 0
            ),
        },
        "evidence_boundary": "Synthetic online learning; not production lift.",
    }
