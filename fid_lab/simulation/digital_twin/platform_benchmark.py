"""Scale screen for the real observable retrieval/ranking/Joiner cascade."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch

from .catalog import build_public_catalog
from .engine import AtomicSimulationKernel
from .event_log import ObservableEventLog
from .experiments.layered import LayeredExperimentPlan, PolicyLayer
from .platform import (
    ROUTE_NAMES,
    CascadePolicy,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
)
from .samples.joiner import JoinerConfig, RequestLevelJoiner
from .samples.negative_sampling import NegativeSource, negative_source_counts
from .world import UserEcosystemWorld, UserWorldConfig


@dataclass(frozen=True)
class ReferenceBenchmarkConfig:
    users: int = 5_000
    items: int = 100_000
    steps: int = 2
    device: str = "cuda"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_experiment() -> LayeredExperimentPlan:
    control = CascadePolicy("observable-linear-v1", 1, 1, 1)
    return LayeredExperimentPlan(
        control,
        (PolicyLayer(
            "fine-rank",
            719,
            {
                "coarse_version_id": 2,
                "fine_version_id": 2,
                "mix_version_id": 2,
                "cross_weight": 0.20,
                "sequence_weight": 0.28,
            },
            control_fraction=0.05,
            treatment_fraction=0.05,
        ),),
    )


def _sample_authority_metrics(joined) -> dict[str, object]:
    recall = joined.recall
    negative_valid = recall.negative_item_id >= 0
    valid_count = int(negative_valid.sum())
    source_count = {
        source.name.lower(): int(
            (recall.negative_source == int(source)).sum()
        )
        for source in NegativeSource
    }
    expected = recall.negative_expected_count[negative_valid]
    fine = joined.fine
    fine_valid = fine.item_id >= 0
    history_valid = fine.context.history_item_id >= 0
    event_types = fine.context.history_event_type[history_valid]
    return {
        "recall_positive_requests": len(recall.request_id),
        "negative_draw_budget": recall.negative_item_id.shape[1],
        "negative_source_draws_per_request": dict(zip(
            tuple(source.name.lower() for source in NegativeSource),
            negative_source_counts(recall.negative_item_id.shape[1]),
        )),
        "negative_rows_by_source": source_count,
        "negative_valid_rate": valid_count / max(negative_valid.numel(), 1),
        "false_negative_rate": (
            float(recall.negative_false_negative_mask.sum())
            / max(valid_count, 1)
        ),
        "observed_negative_rate": (
            float(recall.negative_observed.sum()) / max(valid_count, 1)
        ),
        "expected_count_range": (
            [float(expected.min()), float(expected.max())]
            if len(expected) else [0.0, 0.0]
        ),
        "coarse_teacher_rows": int(joined.coarse.teacher_mask.sum()),
        "coarse_conflict_rows": int(joined.coarse.conflict_mask.sum()),
        "fine_applicable_labels": int(fine.label_applicable.sum()),
        "fine_mature_labels": int(fine.label_mature.sum()),
        "fine_training_labels": int(fine.label_mask.sum()),
        "fine_ope_support_rate": float(
            fine.ope_supported[fine_valid].float().mean()
        ),
        "history_valid_events": int(history_valid.sum()),
        "history_distinct_event_types": int(torch.unique(event_types).numel()),
    }


def run_reference_benchmark(
    config: ReferenceBenchmarkConfig,
) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    catalog = build_public_catalog(
        items=config.items,
        creators=max(config.items // 20, 1),
        merchants=max(config.items // 100, 1),
        advertisers=max(config.items // 200, 1),
        topics=64,
        countries=12,
        regions_per_country=16,
        embedding_dim=32,
        platform_seed=701,
        device=device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users,
        topics=64,
        embedding_dim=32,
        countries=12,
        regions_per_country=16,
        environment_seed=709,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=config.users, history_length=32),
        catalog,
        RetrievalConfig(
            route_k=24,
            merged_k=96,
            graph_neighbors=24,
            refresh_interval=1,
        ),
        RankingConfig(coarse_k=48, fine_k=16, expose_k=8),
    )
    event_log = ObservableEventLog(
        allowed_lateness=world.max_reporting_lag,
    )
    kernel = AtomicSimulationKernel(world, platform, event_log)
    experiment = _benchmark_experiment()
    _sync(device)
    started = time.perf_counter()
    requests = 0
    last = None
    for logical_time in range(config.steps):
        last = kernel.step(logical_time, experiment)
        requests += last.rendered_requests
    _sync(device)
    cascade_elapsed = time.perf_counter() - started
    if last is None or last.candidate_trace is None:
        raise RuntimeError("reference cascade did not emit a serving trace")
    trace = last.candidate_trace
    joiner_started = time.perf_counter()
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=20), catalog,
    ).materialize(
        trace,
        last.request_context,
        last.response_events,
        event_watermark=config.steps - 1,
    )
    _sync(device)
    joiner_elapsed = time.perf_counter() - joiner_started
    route_coverage = {
        route: int((trace.recall_route_id & (1 << index)).any(dim=1).sum())
        for index, route in enumerate(ROUTE_NAMES)
    }
    return {
        "scope": "observable-reference-cascade-screen",
        "quality_claim": False,
        "device": str(device),
        "ann_backend": platform.retriever.faiss.backend,
        "users": config.users,
        "catalog_items": config.items,
        "steps": config.steps,
        "requests": requests,
        "cascade_seconds": cascade_elapsed,
        "joiner_seconds": joiner_elapsed,
        "total_seconds": cascade_elapsed + joiner_elapsed,
        "requests_per_second": requests / cascade_elapsed,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        ),
        "stage_widths": {
            "recall": trace.recall_item_id.shape[1],
            "coarse": trace.coarse_item_id.shape[1],
            "fine": trace.fine_item_id.shape[1],
            "exposed": trace.exposed_item_id.shape[1],
        },
        "route_request_coverage": route_coverage,
        "version_ids": {
            "recall": sorted(set(trace.recall_version_id.tolist())),
            "coarse": sorted(set(trace.coarse_version_id.tolist())),
            "fine": sorted(set(trace.fine_version_id.tolist())),
            "mix": sorted(set(trace.mix_version_id.tolist())),
        },
        "sample_authorities": _sample_authority_metrics(joined),
        "evidence_boundary": (
            "FAISS, sparse co-visit, observable routes, coarse/fine/rerank and "
            "Joiner throughput only; no recommendation lift or production QPS claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=5_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_reference_benchmark(ReferenceBenchmarkConfig(
        users=args.users,
        items=args.items,
        steps=args.steps,
        device=args.device,
    )), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
