"""Materialize canonical NeuralSCM rows from held-out v4 stress worlds."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch

from .....feed_loop.world_model.contracts import (
    STRUCTURAL_INTERVENTION_NAMES,
    WORLD_LABEL_COUNT,
)
from ...catalog import build_public_catalog
from ...contracts import EventType, Surface
from ...platform import (
    CascadePolicy,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
)
from ..behavior import sample_response_tensors
from ..neural_features import (
    V4_FEATURE_CONTRACT,
    V4_FEATURE_COVERAGE,
    build_neural_scm_batch,
)
from ..runtime import UserEcosystemWorld
from ..state import UserWorldConfig
from .contracts import (
    STRUCTURAL_BRIDGE_SCHEMA,
    StructuralBridgeConfig,
)
from .interventions import paired_interventions, select_responses
from .storage import (
    import_compatible_parts,
    load_family_part,
    load_or_create_build_state,
    write_family_part,
    write_final_manifest,
)


def _selected(values, choice):
    rows = torch.arange(len(choice), device=choice.device)
    return values[rows, choice]


def _labels(response, catalog, slate, choice):
    item = _selected(slate.item_ids, choice)
    stay = _selected(response.dwell_ms, choice).float() / 1_000.0
    duration = catalog.duration_seconds[item].clamp_min(1.0)
    labels = torch.zeros(len(choice), WORLD_LABEL_COUNT, device=choice.device)
    masks = torch.zeros_like(labels)
    action = response.action
    mapping = {
        0: EventType.PLAY,
        1: EventType.PLAY_3S,
        4: EventType.COMPLETE,
        5: EventType.LONG_VIEW,
        7: EventType.LIKE,
        8: EventType.NEGATIVE,
        16: EventType.COMMENT,
        17: EventType.SHARE,
        18: EventType.FOLLOW,
    }
    for column, event_type in mapping.items():
        labels[:, column] = _selected(action[event_type], choice)
    labels[:, 2] = stay
    labels[:, 3] = (stay / duration).clamp(0.0, 1.0)
    labels[:, 6] = labels[:, 0] * (
        stay >= torch.minimum(duration, torch.full_like(duration, 30.0))
    )
    masks[:, tuple((*mapping, 2, 3, 6))] = 1.0
    return labels, masks


def _example_payload(snapshot, catalog, slate, response, batch, choice,
                     family_id, served_scores):
    count = len(choice)
    labels, masks = _labels(response, catalog, slate, choice)
    return {
        "exposed_index": choice,
        "exposure_propensity": torch.full(
            (count,), 1.0 / slate.item_ids.shape[1], device=choice.device,
        ),
        "candidate_features": batch["slate_features"].to(torch.float16),
        "behavior_sequence": batch["sequence"].to(torch.float16),
        "labels": labels,
        "label_masks": masks.to(torch.uint8),
        "lifecycle_bucket": batch["lifecycle"].to(torch.uint8),
        "region_bucket": batch["region"].to(torch.uint8),
        "user_id": slate.user_id,
        "request_step": snapshot.users.session_depth[slate.user_id],
        "session_id": snapshot.users.session_count[slate.user_id],
        "event_day": torch.div(
            slate.event_time, snapshot.ticks_per_day, rounding_mode="floor",
        ),
        "candidate_fine_scores": served_scores,
        "candidate_audit_utility": response.utility,
        "structural_family_id": torch.full_like(choice, family_id),
    }


def _append_payload(storage, payload, limit):
    rows = min(len(payload["exposed_index"]), limit)
    for name, value in payload.items():
        storage.setdefault(name, []).append(value[:rows].detach().cpu())
    return rows


def _capture(storage, snapshot, catalog, slate, response, family_id, limit,
             split, seed, served_scores):
    base_count = min(len(slate.request_id), limit)
    selector = torch.arange(base_count, device=slate.request_id.device)
    slate = slate.select(selector)
    served_scores = served_scores[selector]
    response = select_responses(response, selector)
    batch = build_neural_scm_batch(snapshot, catalog, slate)
    choice = torch.remainder(slate.request_id, slate.item_ids.shape[1]).long()
    base = _example_payload(
        snapshot, catalog, slate, response, batch, choice, family_id,
        served_scores,
    )
    if split == "test":
        intervention_payload = paired_interventions(
            snapshot, catalog, slate, response, choice, seed,
        )
        base.update(intervention_payload)
        return _append_payload(storage, base, limit), base_count
    return _append_payload(storage, base, limit), base_count


def _build_family(config, family_id):
    platform_seed = config.platform_seed + family_id * 100_003
    environment_seed = config.environment_seed + family_id * 100_019
    catalog = build_public_catalog(
        items=config.items, creators=max(config.items // 20, 1),
        merchants=max(config.items // 100, 1), topics=config.topics,
        countries=config.countries, regions_per_country=config.regions_per_country,
        embedding_dim=config.embedding_dim, platform_seed=platform_seed,
        device=config.device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users, topics=config.topics,
        embedding_dim=config.embedding_dim, countries=config.countries,
        regions_per_country=config.regions_per_country,
        environment_seed=environment_seed, future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=config.users), catalog,
        ranking_config=RankingConfig(
            coarse_k=max(64, config.slate_width),
            fine_k=max(24, config.slate_width),
            expose_k=config.slate_width,
        ),
    )
    policy = CascadePolicy(
        name="structural-bridge-reference", coarse_version_id=1,
        fine_version_id=1, mix_version_id=1,
    )
    return world, platform, policy, catalog, platform_seed, environment_seed


def _served_fine_scores(serving, feed):
    exposed = serving.slate.item_ids[feed]
    fine_item = serving.candidate_trace.fine_item_id[feed]
    fine_score = serving.candidate_trace.fine_selected_score[feed]
    match = exposed[:, :, None] == fine_item[:, None, :]
    return torch.where(
        match, fine_score[:, None, :], torch.full_like(
            fine_score[:, None, :], -torch.inf,
        ),
    ).max(dim=2).values


def _base_capture_schedule(rows, augmentation, ticks):
    base_requests = (rows + augmentation - 1) // augmentation
    catchup_ticks = max(8, ticks // 16)
    capture_horizon = max(ticks - catchup_ticks, 1)
    capture_ticks = torch.div(
        torch.arange(base_requests) * capture_horizon,
        base_requests,
        rounding_mode="floor",
    )
    return torch.bincount(capture_ticks, minlength=ticks).tolist()


def _materialize_family(config, family_id, rows, split):
    world, platform, policy, catalog, platform_seed, environment_seed = _build_family(
        config, family_id,
    )
    storage = {}
    captured = 0
    base_requests = 0
    schedule = _base_capture_schedule(rows, 1, config.ticks)
    pending_bases = 0
    capture_ticks = []
    maximum_ticks = config.ticks + config.max_extension_ticks
    for logical_time in range(maximum_ticks):
        if logical_time < config.ticks:
            pending_bases += schedule[logical_time]
        entry = world.schedule(logical_time)
        world.commit(entry)
        platform.ingest(entry)
        requests = platform.open_requests(entry)
        if not len(requests.user_id):
            if logical_time + 1 >= config.ticks and captured >= rows:
                break
            continue
        serving = platform.render(
            platform.snapshot(), requests, policy, 0,
            torch.ones_like(requests.user_id, dtype=torch.float),
        )
        feed = serving.slate.surface == int(Surface.FEED)
        snapshot = world.snapshot()
        if feed.any() and pending_bases and captured < rows:
            slate = serving.slate.select(feed)
            served_scores = _served_fine_scores(serving, feed)
            response = sample_response_tensors(
                snapshot, catalog, slate, environment_seed,
            )
            row_limit = min(
                rows - captured, pending_bases,
            )
            added, bases = _capture(
                storage, snapshot, catalog, slate, response, family_id,
                row_limit, split, environment_seed, served_scores,
            )
            captured += added
            base_requests += bases
            pending_bases -= bases
            capture_ticks.append(logical_time)
        factual = world.respond(snapshot, serving.slate)
        world.commit(factual)
        platform.ingest(factual)
        if logical_time + 1 >= config.ticks and captured >= rows:
            break
    if captured < rows:
        raise ValueError(
            f"structural family {family_id} produced {captured} rows; "
            f"requested {rows}"
        )
    tensors = {
        name: torch.cat(parts) for name, parts in storage.items()
    }
    return tensors, None, {
        "family_id": family_id,
        "platform_seed": platform_seed,
        "environment_seed": environment_seed,
        "rows": captured,
        "base_requests": base_requests,
        "capture_tick_min": min(capture_ticks),
        "capture_tick_max": max(capture_ticks),
        "capture_tick_count": len(capture_ticks),
        "planned_ticks": config.ticks,
        "simulated_ticks": logical_time + 1,
        "extension_ticks": max(logical_time + 1 - config.ticks, 0),
        "row_semantics": (
            "factual_control_with_validation_probes" if split == "test"
            else "factual_control_only"
        ),
    }


def _write_split(output_dir, split, tensors):
    payload = {
        "tensors": {
            **tensors,
            "candidate_utility_source": "v4_formula_structural_world",
        }
    }
    path = output_dir / f"{split}.pt"
    torch.save(payload, path)
    return {
        "rows": len(tensors["user_id"]),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _merge_tensors(parts):
    return {
        name: torch.cat(tuple(part[name] for part in parts))
        for name in parts[0]
    }


def build_structural_bridge(
    output_dir: Path, config: StructuralBridgeConfig, reuse_build: Path | None = None,
):
    split_rows = config.split_rows()
    first_train = split_rows["train"] // 2
    family_plan = (
        (1, "train", first_train),
        (2, "train", split_rows["train"] - first_train),
        (3, "validation", split_rows["validation"]),
        (config.test_family_id, "test", split_rows["test"]),
    )
    state = load_or_create_build_state(output_dir, config, family_plan)
    if reuse_build is not None:
        import_compatible_parts(output_dir, state, reuse_build)
    tensors_by_split = {name: [] for name in ("train", "validation", "test")}
    families = {name: [] for name in ("train", "validation", "test")}
    for family_id, split, rows in family_plan:
        payload = load_family_part(
            output_dir, state, family_id, split,
        )
        if payload is None:
            tensors, paired, family = _materialize_family(
                config, family_id, rows, split,
            )
            write_family_part(
                output_dir, state, family_id, split,
                tensors, paired, family,
            )
        else:
            tensors = payload["tensors"]
            paired = payload["paired"]
            family = payload["family"]
        tensors_by_split[split].append(tensors)
        families[split].append(family)
    split_records = {
        split: _write_split(output_dir, split, _merge_tensors(parts))
        for split, parts in tensors_by_split.items()
    }
    manifest = {
        "schema": STRUCTURAL_BRIDGE_SCHEMA,
        "source": "synthetic_v4_formula_stress_worlds",
        "config": asdict(config),
        "feature_contract": V4_FEATURE_CONTRACT,
        "feature_contract_sha256": V4_FEATURE_CONTRACT["sha256"],
        "feature_coverage": {
            key: "native_v4" if value == "native_v4" else "unused"
            for key, value in V4_FEATURE_COVERAGE.items()
        },
        "split_authority": "multi_train_disjoint_holdout_world_families",
        "families": families,
        "interventions": list(STRUCTURAL_INTERVENTION_NAMES),
        "splits": split_records,
        "family_parts": dict(state["completed"]),
        "evidence_boundary": (
            "Synthetic factual families provide stress coverage. Counterfactual "
            "interventions exist only on the untouched test family as validation "
            "probes; they never train or mutate the factual world."
        ),
    }
    write_final_manifest(output_dir, manifest)
    return manifest
