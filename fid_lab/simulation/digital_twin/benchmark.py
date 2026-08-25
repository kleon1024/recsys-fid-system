"""GPU throughput benchmark for the world/kernel boundary, not model quality."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch

from .catalog import PublicCatalog, build_public_catalog
from .contracts import (
    AppEventBatch,
    EventType,
    PlatformRequestBatch,
    RenderedSlateBatch,
)
from .engine import AtomicSimulationKernel, ExperimentPlan
from .event_log import ObservableEventLog
from .platform import ObservableProjection, open_platform_requests
from .world import UserEcosystemWorld, UserWorldConfig


@dataclass(frozen=True)
class WorldBenchmarkConfig:
    users: int = 100_000
    items: int = 500_000
    slate_width: int = 32
    embedding_dim: int = 32
    steps: int = 1
    device: str = "cuda"


class TransportBenchmarkPlatform:
    """Deterministic slate transport; deliberately not a retrieval model."""

    def __init__(self, catalog: PublicCatalog, width: int, users: int):
        self.catalog = catalog
        self.width = width
        self.ingested_events = 0
        self.projection = ObservableProjection(
            users, catalog, history_length=32,
        )

    def ingest(self, events: AppEventBatch) -> None:
        self.ingested_events += len(events.event_id)
        self.projection.ingest(events)

    def snapshot(self) -> object:
        return self.projection.state

    def open_requests(
        self, entry_events: AppEventBatch,
    ) -> PlatformRequestBatch:
        return open_platform_requests(entry_events)

    def render(
        self,
        snapshot: object,
        requests: PlatformRequestBatch,
        policy: object,
        experiment_cell: int,
        assignment_probability: torch.Tensor,
    ) -> RenderedSlateBatch:
        del snapshot
        position = torch.arange(
            self.width, device=requests.user_id.device,
        )[None].expand(len(requests.user_id), -1)
        item = torch.remainder(
            requests.request_id[:, None] * 503
            + position * 1_009
            + int(policy),
            len(self.catalog.item_id),
        )
        return RenderedSlateBatch(
            request_id=requests.request_id,
            user_id=requests.user_id,
            surface=requests.surface,
            event_time=requests.event_time,
            item_ids=item,
            positions=position,
            valid=torch.ones_like(item, dtype=torch.bool),
            ui_variant=torch.full_like(requests.user_id, experiment_cell),
            exposure_probability=torch.ones_like(item, dtype=torch.float),
            selection_policy_kind=torch.zeros_like(requests.user_id),
            exploration_rate=torch.zeros_like(requests.user_id, dtype=torch.float),
            slate_log_probability=torch.zeros_like(
                requests.user_id, dtype=torch.float,
            ),
            assignment_probability=assignment_probability,
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_world_benchmark(config: WorldBenchmarkConfig) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested without an available GPU")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    catalog = build_public_catalog(
        items=config.items,
        creators=max(config.items // 20, 1),
        merchants=max(config.items // 100, 1),
        topics=64,
        countries=12,
        regions_per_country=16,
        embedding_dim=config.embedding_dim,
        platform_seed=101,
        device=device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users,
        topics=64,
        embedding_dim=config.embedding_dim,
        countries=12,
        regions_per_country=16,
        environment_seed=103,
        future_signup_fraction=0.0,
    ), catalog)
    platform = TransportBenchmarkPlatform(
        catalog, config.slate_width, config.users,
    )
    event_log = ObservableEventLog(
        allowed_lateness=world.max_reporting_lag,
    )
    kernel = AtomicSimulationKernel(
        world,
        platform,
        event_log,
    )
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=0,
        treatment_policy=7,
        experiment_seed=107,
        control_fraction=0.05,
        treatment_fraction=0.05,
    )
    _synchronize(device)
    started = time.perf_counter()
    requests = events = 0
    delayed_counts = {
        event_type.name.lower(): 0
        for event_type in (
            EventType.ORDER,
            EventType.PAYMENT,
            EventType.REFUND,
            EventType.PIXEL_CONVERSION,
        )
    }
    for logical_time in range(config.steps):
        result = kernel.step(logical_time, experiment)
        requests += result.rendered_requests
        events += len(result.entry_events.event_id)
        events += len(result.response_events.event_id)
        for event_type in (
            EventType.ORDER,
            EventType.PAYMENT,
            EventType.REFUND,
            EventType.PIXEL_CONVERSION,
        ):
            delayed_counts[event_type.name.lower()] += int(
                result.entry_events.event(event_type).sum()
            )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    candidates = requests * config.slate_width
    return {
        "scope": "world-projection-kernel-throughput-only",
        "device": str(device),
        "users": config.users,
        "catalog_items": config.items,
        "steps": config.steps,
        "requests": requests,
        "candidates": candidates,
        "observable_events": events,
        "delayed_events": delayed_counts,
        "pending_delayed_events": world.delayed.pending_events,
        "world_authorities": world.manifest(),
        "event_watermark": event_log.watermark,
        "ingest_watermark": event_log.ingest_watermark,
        "elapsed_seconds": elapsed,
        "requests_per_second": requests / elapsed,
        "candidates_per_second": candidates / elapsed,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        ),
        "quality_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--slate-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_world_benchmark(WorldBenchmarkConfig(
        users=args.users,
        items=args.items,
        slate_width=args.slate_width,
        steps=args.steps,
        device=args.device,
    ))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
