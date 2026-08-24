"""Powered creator A/B for Feed Posting over bounded request partitions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from ...launches.statistics import CreatorClusterAccumulator
from ...value import DEFAULT_LT_CONFIG
from ..contracts import FeedPostingConfig
from ..models import load_bundle
from ..simulation.features import candidate_features, rule_score
from ..simulation.response import simulate_response
from ..simulation.retrieval import retrieve
from ..simulation.world import build_world_partition


AB_KEYS = (
    "publish_rate", "quality_supply_per_request",
    "feed_stay_seconds_per_request", "feed_active_day_per_request",
    "negative_per_request", "selected_content_risk",
    "platform_lt_per_request",
)


def _outcomes(response):
    published = response["published"].float()
    stay_rate = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_rate = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    return {
        "publish_rate": published,
        "quality_supply_per_request": published * response["quality_potential"],
        "feed_stay_seconds_per_request": response["feed_stay_seconds"],
        "feed_active_day_per_request": response["feed_active_day"],
        "negative_per_request": response["negative"].float(),
        "selected_content_risk": response["content_risk"],
        "platform_lt_per_request": (
            response["feed_stay_seconds"] / 60.0 * stay_rate
            + response["feed_active_day"] * active_rate
        ),
    }


def _partition_outcomes(
    config, start, count, control_bundle, treatment_bundle,
    control_blend, treatment_blend,
):
    world = build_world_partition(config, start, count)
    candidates = retrieve(world, ("trending", "i2i"))
    features = candidate_features(world, candidates)
    semantic = world.catalog.semantic[candidates.prompt_ids]
    history = world.requests.feed_sequence
    baseline = rule_score(features)
    control_scores = _blended_score(
        control_bundle, features, semantic, history, baseline, control_blend
    )
    treatment_scores = _blended_score(
        treatment_bundle, features, semantic, history, baseline, treatment_blend
    )
    return (
        world.requests.creator_id,
        _outcomes(simulate_response(world, candidates, control_scores)),
        _outcomes(simulate_response(world, candidates, treatment_scores)),
    )


def _blended_score(bundle, features, semantic, history, baseline, blend):
    if bundle is None:
        return baseline
    learned = bundle.score(features, semantic, history)
    return baseline + blend * (learned - baseline)


def run_partitioned_feed_posting_ab(
    config: FeedPostingConfig,
    model_path: Path,
    partition_requests: int = 50_000,
    control_model_path: Path | None = None,
    treatment_blend: float = 1.0,
    control_blend: float = 1.0,
):
    if config.world_version != "creator-neural-feed-supply-v4":
        raise ValueError("partitioned Feed posting A/B requires creator V4")
    if not 0.0 <= treatment_blend <= 1.0 or not 0.0 <= control_blend <= 1.0:
        raise ValueError("Feed posting blend must be between zero and one")
    treatment = load_bundle(model_path, config.device)
    control = (
        None if control_model_path is None
        else load_bundle(control_model_path, config.device)
    )
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    accumulator = CreatorClusterAccumulator(AB_KEYS, config.creators, device)
    started = perf_counter()
    for start in range(0, config.requests, partition_requests):
        count = min(partition_requests, config.requests - start)
        creator_ids, control_values, treatment_values = _partition_outcomes(
            config, start, count, control, treatment,
            control_blend, treatment_blend,
        )
        accumulator.add(creator_ids, control_values, treatment_values)
    paired_replay, randomized, observed_creators = accumulator.report()
    gates = {
        "publish_positive": randomized["publish_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_nonnegative": randomized["platform_lt_per_request"][
            "confidence_interval"
        ][0] >= 0,
        "quality_supply_nonnegative": randomized["quality_supply_per_request"][
            "confidence_interval"
        ][0] >= -0.0002,
        "content_risk_guardrail": randomized["selected_content_risk"][
            "confidence_interval"
        ][1] <= 0.0002,
    }
    elapsed = perf_counter() - started
    return {
        "schema": "partitioned-feed-posting-v4-ab-v2",
        "decision_estimator": "creator_cluster_randomized_ab",
        "control": (
            "trending_i2i_plus_rule" if control is None
            else f"trending_i2i_plus_{control.name}"
        ),
        "treatment": f"trending_i2i_plus_{treatment.name}",
        "control_blend": 0.0 if control is None else control_blend,
        "treatment_blend": treatment_blend,
        "requests": config.requests,
        "creators": observed_creators,
        "partition_requests": partition_requests,
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "control_model_sha256": (
            None if control_model_path is None
            else sha256(control_model_path.read_bytes()).hexdigest()
        ),
        "metrics": randomized,
        "paired_counterfactual_replay": paired_replay,
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "hold_or_reject",
        "performance": {
            "seconds": elapsed,
            "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
        },
        "evidence_boundary": (
            "Large-scale paired and creator-randomized A/B in synthetic Feed "
            "Posting V4. It is simulator evidence, not production lift."
        ),
    }
