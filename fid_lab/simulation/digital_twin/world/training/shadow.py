"""Factual v4 cascade replay with NeuralSCM as a non-committing shadow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .....feed_loop.world_model.training import load_world_ensemble
from .....feed_loop.world_model.validation.support import request_support_mask
from ...catalog import build_public_catalog
from ...contracts import AppEventBatch, EventType, Surface
from ...platform import (
    CascadePolicy,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
)
from ..authority import FormulaResponseAuthority, NeuralFeedResponseAuthority
from ..neural_features import build_neural_scm_batch
from ..runtime import UserEcosystemWorld
from ..state import UserWorldConfig


FLOAT_EVENT_FIELDS = frozenset({
    "value", "logging_probability", "assignment_probability",
})
REPLAY_FLOAT_TOLERANCE = 1e-6
REPLAY_DURATION_TOLERANCE_MS = 1


@dataclass(frozen=True)
class AuthorityShadowConfig:
    users: int = 20_000
    items: int = 200_000
    ticks: int = 8
    topics: int = 64
    countries: int = 12
    regions_per_country: int = 16
    embedding_dim: int = 32
    platform_seed: int = 811
    environment_seed: int = 821
    member_index: int = 0
    inference_batch_size: int = 4_096
    device: str = "cuda:0"


def _tensor_hash(hasher, value):
    tensor = value.detach().cpu().contiguous()
    hasher.update(str(tensor.dtype).encode())
    hasher.update(str(tuple(tensor.shape)).encode())
    hasher.update(tensor.numpy().tobytes())


def _event_hash(events: AppEventBatch):
    order = torch.argsort(events.event_id, stable=True)
    hasher = sha256()
    for field in fields(events):
        _tensor_hash(hasher, getattr(events, field.name)[order])
    return hasher.hexdigest()


def _event_snapshot(events: AppEventBatch):
    order = torch.argsort(events.event_id, stable=True)
    metadata = {}
    values = {}
    for field in fields(events):
        value = getattr(events, field.name)[order].detach().cpu().contiguous()
        hasher = sha256()
        _tensor_hash(hasher, value)
        metadata[field.name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hasher.hexdigest(),
        }
        if field.name in FLOAT_EVENT_FIELDS or field.name == "duration_ms":
            values[field.name] = value
    return {
        "schema": "semantic-event-evidence-v1",
        "metadata": metadata,
        "values": values,
    }


def _nonfinite_equal(left, right):
    return (
        torch.equal(torch.isnan(left), torch.isnan(right))
        and torch.equal(torch.isposinf(left), torch.isposinf(right))
        and torch.equal(torch.isneginf(left), torch.isneginf(right))
    )


def _semantic_event_comparison(left, right_events):
    right = _event_snapshot(right_events)
    left_metadata = left["metadata"]
    right_metadata = right["metadata"]
    discrete_equal = True
    discrete_mismatches = {}
    discrete_max_deltas = {}
    float_finite_pattern_equal = True
    float_deltas = {}
    duration_delta_ms = 0
    shape_equal = set(left_metadata) == set(right_metadata)
    for name in sorted(set(left_metadata) & set(right_metadata)):
        left_meta, right_meta = left_metadata[name], right_metadata[name]
        if (
            left_meta["shape"] != right_meta["shape"]
            or left_meta["dtype"] != right_meta["dtype"]
        ):
            shape_equal = False
            continue
        if name == "duration_ms":
            left_value = left["values"][name]
            right_value = right["values"][name]
            duration_delta_ms = int(
                (left_value.long() - right_value.long()).abs().max()
            ) if len(left_value) else 0
            continue
        if name not in FLOAT_EVENT_FIELDS:
            mismatch = int(left_meta["sha256"] != right_meta["sha256"])
            discrete_mismatches[name] = mismatch
            discrete_max_deltas[name] = None if mismatch else 0
            discrete_equal &= mismatch == 0
            continue
        left_value = left["values"][name]
        right_value = right["values"][name]
        pattern_equal = _nonfinite_equal(left_value, right_value)
        float_finite_pattern_equal &= pattern_equal
        finite = torch.isfinite(left_value) & torch.isfinite(right_value)
        float_deltas[name] = (
            float((left_value[finite] - right_value[finite]).abs().max())
            if finite.any() else 0.0
        )
    maximum_float_delta = max(float_deltas.values(), default=0.0)
    passed = (
        shape_equal and discrete_equal and float_finite_pattern_equal
        and maximum_float_delta <= REPLAY_FLOAT_TOLERANCE
        and duration_delta_ms <= REPLAY_DURATION_TOLERANCE_MS
    )
    return {
        "pass": bool(passed),
        "shape_equal": bool(shape_equal),
        "discrete_fields_exact": bool(discrete_equal),
        "discrete_mismatch_unit": "field_hash_flag",
        "discrete_field_mismatches": discrete_mismatches,
        "discrete_field_max_deltas": discrete_max_deltas,
        "float_nonfinite_patterns_exact": bool(float_finite_pattern_equal),
        "float_field_max_deltas": float_deltas,
        "maximum_float_delta": maximum_float_delta,
        "float_tolerance": REPLAY_FLOAT_TOLERANCE,
        "duration_max_delta_ms": duration_delta_ms,
        "duration_tolerance_ms": REPLAY_DURATION_TOLERANCE_MS,
    }


def _semantic_replay(comparisons):
    discrete_fields = set().union(*(
        set(row["discrete_field_mismatches"]) for row in comparisons
    )) if comparisons else set()
    return {
        "pass": all(row["pass"] for row in comparisons),
        "tick_count_equal": True,
        "ticks": len(comparisons),
        "discrete_fields_exact": all(
            row["discrete_fields_exact"] for row in comparisons
        ),
        "discrete_field_mismatches": {
            name: sum(
                row["discrete_field_mismatches"].get(name, 0)
                for row in comparisons
            )
            for name in sorted(discrete_fields)
        },
        "discrete_field_max_deltas": {
            name: None if any(
                row["discrete_field_max_deltas"].get(name, 0) is None
                for row in comparisons
            ) else 0
            for name in sorted(discrete_fields)
        },
        "float_nonfinite_patterns_exact": all(
            row["float_nonfinite_patterns_exact"] for row in comparisons
        ),
        "maximum_float_delta": max(
            (row["maximum_float_delta"] for row in comparisons), default=0.0,
        ),
        "float_tolerance": REPLAY_FLOAT_TOLERANCE,
        "duration_max_delta_ms": max(
            (row["duration_max_delta_ms"] for row in comparisons), default=0,
        ),
        "duration_tolerance_ms": REPLAY_DURATION_TOLERANCE_MS,
    }


def _slate_hash(slate):
    hasher = sha256()
    for name in ("request_id", "user_id", "surface", "item_ids", "positions"):
        _tensor_hash(hasher, getattr(slate, name))
    return hasher.hexdigest()


def _event_counts(events):
    return {
        event_type.name.lower(): int(events.event(event_type).sum())
        for event_type in (
            EventType.IMPRESSION, EventType.PLAY, EventType.PLAY_3S,
            EventType.LONG_VIEW, EventType.COMPLETE, EventType.LIKE,
            EventType.NEGATIVE, EventType.SESSION_END,
        )
    }


def _load_authority(artifact_dir, device, member_index, inference_batch_size):
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ensemble = load_world_ensemble(artifact_dir, device)
    authority = NeuralFeedResponseAuthority(
        ensemble,
        member_index=member_index,
        artifact_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        feature_contract_sha256=manifest["feature_contract_sha256"],
        feature_coverage=manifest["feature_coverage"],
        support_profile=manifest["support_profile"],
        inference_batch_size=inference_batch_size,
    )
    return authority, manifest


def _build_runtime(config, authority):
    catalog = build_public_catalog(
        items=config.items, creators=max(config.items // 20, 1),
        merchants=max(config.items // 100, 1), topics=config.topics,
        countries=config.countries,
        regions_per_country=config.regions_per_country,
        embedding_dim=config.embedding_dim,
        platform_seed=config.platform_seed, device=config.device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users, topics=config.topics,
        embedding_dim=config.embedding_dim, countries=config.countries,
        regions_per_country=config.regions_per_country,
        environment_seed=config.environment_seed, future_signup_fraction=0.0,
        initialization_mode="bootstrap",
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=config.users), catalog,
    )
    return catalog, world, platform


def _partition_hash(authority, snapshot, catalog, slate, seed):
    full = authority.respond(snapshot, catalog, slate, seed)
    midpoint = max(len(slate.request_id) // 2, 1)
    parts = (
        authority.respond(snapshot, catalog, slate.select(slice(0, midpoint)), seed),
        authority.respond(snapshot, catalog, slate.select(slice(midpoint, None)), seed),
    )
    partitioned = AppEventBatch.concatenate(parts)
    return _event_hash(full), _event_hash(partitioned)


def _replay_path(root, authority_name, logical_time):
    return root / f"{authority_name}-{logical_time:05d}.pt"


def _replay_evidence(root, logical_time, neural, factual, capture):
    comparisons = {}
    for name, events in (("neural", neural), ("formula", factual)):
        path = _replay_path(root, name, logical_time)
        if capture:
            torch.save(_event_snapshot(events), path)
            continue
        reference = torch.load(path, map_location="cpu", weights_only=True)
        comparisons[name] = _semantic_event_comparison(reference, events)
        path.unlink()
    return comparisons


def _run_once(config, authority, replay_root, capture):
    catalog, world, platform = _build_runtime(config, authority)
    formula = FormulaResponseAuthority()
    policy = CascadePolicy(
        name="p2-shadow-reference", coarse_version_id=1,
        fine_version_id=1, mix_version_id=1,
    )
    trace_hashes = []
    neural_hashes = []
    formula_hashes = []
    partition_pairs = []
    feed_requests = supported_requests = 0
    neural_counts = {}
    formula_counts = {}
    replay_comparisons = {"neural": [], "formula": []}
    for logical_time in range(config.ticks):
        entry = world.schedule(logical_time)
        world.commit(entry)
        platform.ingest(entry)
        requests = platform.open_requests(entry)
        serving = platform.render(
            platform.snapshot(), requests, policy, 0,
            torch.ones_like(requests.user_id, dtype=torch.float),
        )
        trace_hashes.append(_slate_hash(serving.slate))
        snapshot = world.snapshot()
        feed = serving.slate.surface == int(Surface.FEED)
        if feed.any():
            feed_slate = serving.slate.select(feed)
            batch = build_neural_scm_batch(snapshot, catalog, feed_slate)
            support = request_support_mask(
                batch["slate_features"], authority.support_profile,
            )
            feed_requests += len(feed_slate.request_id)
            supported_requests += int(support.sum())
            partition_pairs.append(_partition_hash(
                authority, snapshot, catalog, feed_slate,
                config.environment_seed,
            ))
        neural = authority.respond(
            snapshot, catalog, serving.slate, config.environment_seed,
        )
        factual = formula.respond(
            snapshot, catalog, serving.slate, config.environment_seed,
        )
        neural_hashes.append(_event_hash(neural))
        formula_hashes.append(_event_hash(factual))
        comparisons = _replay_evidence(
            replay_root, logical_time, neural, factual, capture,
        )
        for name, comparison in comparisons.items():
            replay_comparisons[name].append(comparison)
        for name, value in _event_counts(neural).items():
            neural_counts[name] = neural_counts.get(name, 0) + value
        for name, value in _event_counts(factual).items():
            formula_counts[name] = formula_counts.get(name, 0) + value
        world.commit(factual)
        platform.ingest(factual)
    report = {
        "trace_hash": sha256("".join(trace_hashes).encode()).hexdigest(),
        "neural_event_hash": sha256("".join(neural_hashes).encode()).hexdigest(),
        "formula_event_hash": sha256("".join(formula_hashes).encode()).hexdigest(),
        "feed_requests": feed_requests,
        "supported_feed_requests": supported_requests,
        "support_rate": supported_requests / max(feed_requests, 1),
        "partition_pairs": partition_pairs,
        "neural_event_counts": neural_counts,
        "formula_event_counts": formula_counts,
    }
    return report, replay_comparisons


def run_authority_shadow(artifact_dir: Path, config: AuthorityShadowConfig):
    authority, manifest = _load_authority(
        artifact_dir, config.device, config.member_index,
        config.inference_batch_size,
    )
    with TemporaryDirectory(prefix="fid-shadow-replay-") as replay_dir:
        replay_root = Path(replay_dir)
        first, _ = _run_once(config, authority, replay_root, capture=True)
        second, comparisons = _run_once(
            config, authority, replay_root, capture=False,
        )
    partition_invariant = all(left == right for left, right in first["partition_pairs"])
    neural_replay = _semantic_replay(comparisons["neural"])
    factual_replay = _semantic_replay(comparisons["formula"])
    gates = {
        "factual_slate_replay": first["trace_hash"] == second["trace_hash"],
        "factual_response_replay": factual_replay["pass"],
        "neural_response_replay": neural_replay["pass"],
        "batch_partition_invariance": partition_invariant,
        "support_fallback_bounded": first["support_rate"] >= 0.97,
        "reference_cascade_exercised": first["feed_requests"] > 0,
        "artifact_support_bound": manifest.get("support_profile") is not None,
    }
    return {
        "schema": "neural-scm-v4-authority-shadow-v1",
        "config": asdict(config),
        "artifact_manifest_sha256": authority.artifact_sha256,
        "weights_sha256": manifest["weights_sha256"],
        "feature_contract_sha256": manifest["feature_contract_sha256"],
        "authority_version": authority.version,
        "first_run": first,
        "second_run": second,
        "semantic_replay": {
            "factual": factual_replay,
            "neural": neural_replay,
        },
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "hold",
        "evidence_boundary": (
            "Formula responses alone mutate the factual world. Neural responses "
            "are shadow-only simulator evidence, not production traffic."
        ),
    }
