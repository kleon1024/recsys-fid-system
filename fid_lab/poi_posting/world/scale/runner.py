"""Bounded-memory Supply V4 replay over exact request partitions."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from ....launches.statistics import CreatorClusterAccumulator
from ....value import DEFAULT_LT_CONFIG
from ..contracts import PostingWorldConfig
from ..generator import (
    build_world_partition,
    candidate_features,
    retrieve,
    rule_score,
    simulate_response,
)
from ..models import load_posting_bundle


TOTAL_KEYS = (
    "selected", "published", "relevant_supply", "quality_supply",
    "feed_stay_seconds", "feed_active_day", "negative", "oracle_recalled",
)
AB_KEYS = (
    "publish_rate", "relevant_supply_per_request",
    "quality_supply_per_request", "feed_stay_seconds_per_request",
    "feed_active_day_per_request", "negative_per_request",
    "selected_content_negative_risk", "platform_lt_per_request",
)


def _partition_signature(config, start, count, model_hash):
    payload = {
        "world_version": config.world_version,
        "requests": config.requests,
        "creators": config.creators,
        "seed": config.seed,
        "catalog_seed": config.catalog_seed,
        "start": start,
        "count": count,
        "model_sha256": model_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _partition_totals(config, start, count, bundle):
    world = build_world_partition(config, start, count)
    candidates = retrieve(world, ("popular", "geo"))
    features = candidate_features(world, candidates)
    scores = rule_score(features) if bundle is None else bundle.score(features)
    response = simulate_response(world, candidates, scores)
    published = response["published"].float()
    return {
        "selected": float(response["selected"].sum()),
        "published": float(published.sum()),
        "relevant_supply": float(
            (published * response["selected_relevance"]).double().sum()
        ),
        "quality_supply": float(
            (published * response["supply_quality"]).double().sum()
        ),
        "feed_stay_seconds": float(response["feed_stay_seconds"].double().sum()),
        "feed_active_day": float(response["feed_active_day"].double().sum()),
        "negative": float(response["negative"].double().sum()),
        "oracle_recalled": float(candidates.audit_oracle_recalled.sum()),
    }


def _outcomes(response):
    published = response["published"].float()
    stay_rate = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_rate = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    return {
        "publish_rate": published,
        "relevant_supply_per_request": published * response["selected_relevance"],
        "quality_supply_per_request": published * response["supply_quality"],
        "feed_stay_seconds_per_request": response["feed_stay_seconds"],
        "feed_active_day_per_request": response["feed_active_day"],
        "negative_per_request": response["negative"],
        "selected_content_negative_risk": response[
            "selected_content_negative_risk"
        ],
        "platform_lt_per_request": (
            response["feed_stay_seconds"] / 60.0 * stay_rate
            + response["feed_active_day"] * active_rate
        ),
    }


def _policy_outcomes(config, start, count, control_bundle, treatment_bundle):
    world = build_world_partition(config, start, count)
    candidates = retrieve(world, ("popular", "geo"))
    features = candidate_features(world, candidates)
    control_scores = (
        rule_score(features)
        if control_bundle is None else control_bundle.score(features)
    )
    control = simulate_response(world, candidates, control_scores)
    treatment = simulate_response(
        world, candidates, treatment_bundle.score(features)
    )
    return world.requests.creator_id, _outcomes(control), _outcomes(treatment)


def run_partitioned_supply_ab(
    config: PostingWorldConfig,
    model_path: Path,
    partition_requests: int | None = None,
    control_model_path: Path | None = None,
):
    if config.world_version != "creator-neural-supply-v4":
        raise ValueError("partitioned A/B requires Supply V4")
    partition_requests = partition_requests or config.batch_requests
    bundle = load_posting_bundle(model_path, config.device)
    control_bundle = (
        None if control_model_path is None
        else load_posting_bundle(control_model_path, config.device)
    )
    device = torch.device(config.device)
    accumulator = CreatorClusterAccumulator(AB_KEYS, config.creators, device)
    started = perf_counter()
    for start in range(0, config.requests, partition_requests):
        count = min(partition_requests, config.requests - start)
        creator_ids, control, treatment = _policy_outcomes(
            config, start, count, control_bundle, bundle
        )
        accumulator.add(creator_ids, control, treatment)
    metrics, online, observed_creators = accumulator.report()
    gates = {
        "publish_positive": metrics["publish_rate"]["confidence_interval"][0] > 0,
        "platform_lt_nonnegative": metrics["platform_lt_per_request"][
            "confidence_interval"
        ][0] >= 0,
        "relevant_supply_nonnegative": metrics["relevant_supply_per_request"][
            "confidence_interval"
        ][0] >= -0.0002,
        "negative_guardrail": metrics["selected_content_negative_risk"][
            "confidence_interval"
        ][1] <= 0.0002,
    }
    elapsed = perf_counter() - started
    return {
        "schema": "partitioned-supply-v4-ab-v1",
        "control": (
            "popular_geo_plus_rule" if control_bundle is None
            else f"popular_geo_plus_{control_bundle.name}"
        ),
        "treatment": f"popular_geo_plus_{bundle.name}",
        "requests": config.requests,
        "creators": observed_creators,
        "partition_requests": partition_requests,
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "control_model_sha256": (
            None if control_model_path is None
            else sha256(control_model_path.read_bytes()).hexdigest()
        ),
        "metrics": metrics,
        "creator_randomized_ab": online,
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
            "Large-scale paired and creator-randomized A/B in synthetic Supply V4. "
            "It provides simulator launch evidence, not production lift."
        ),
    }


def _partition_asset(path, signature, config, start, count, bundle):
    if path is not None and path.exists():
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") == "supply-v4-request-partition-v1"
            and payload.get("signature") == signature
            and payload.get("start") == start
            and payload.get("count") == count
            and set(payload.get("totals", {})) == set(TOTAL_KEYS)
        ):
            return payload["totals"], "reused"
    totals = _partition_totals(config, start, count, bundle)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "supply-v4-request-partition-v1",
            "signature": signature,
            "start": start,
            "count": count,
            "totals": totals,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(path)
    return totals, "materialized"


def run_partitioned_supply_replay(
    config: PostingWorldConfig,
    model_path: Path | None = None,
    partition_requests: int | None = None,
    partition_dir: Path | None = None,
):
    if config.world_version != "creator-neural-supply-v4":
        raise ValueError("partitioned replay requires Supply V4")
    partition_requests = partition_requests or config.batch_requests
    if partition_requests < 1:
        raise ValueError("partition request count must be positive")
    bundle = None if model_path is None else load_posting_bundle(
        model_path, config.device
    )
    model_hash = (
        "rule" if model_path is None else sha256(model_path.read_bytes()).hexdigest()
    )
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    totals = {key: 0.0 for key in TOTAL_KEYS}
    partitions = []
    materialized = 0
    reused = 0
    for start in range(0, config.requests, partition_requests):
        count = min(partition_requests, config.requests - start)
        signature = _partition_signature(config, start, count, model_hash)
        path = (
            None if partition_dir is None
            else partition_dir / f"part-{start:012d}.json"
        )
        values, status = _partition_asset(
            path, signature, config, start, count, bundle
        )
        materialized += status == "materialized"
        reused += status == "reused"
        for key, value in values.items():
            totals[key] += value
        partitions.append({
            "logical_key": f"supply.request_partition.{start:012d}",
            "start": start,
            "count": count,
            "signature": signature,
            "status": status,
        })
    elapsed = perf_counter() - started
    return {
        "schema": "partitioned-supply-v4-replay-v1",
        "config": {
            "requests": config.requests,
            "creators": config.creators,
            "batch_requests": partition_requests,
            "seed": config.seed,
            "catalog_seed": config.catalog_seed,
            "world_version": config.world_version,
        },
        "policy": "popular_geo_plus_rule" if bundle is None else (
            f"popular_geo_plus_{bundle.name}"
        ),
        "model_sha256": model_hash,
        "partitions": partitions,
        "resume": {
            "materialized_partitions": materialized,
            "reused_partitions": reused,
            "atomic_partition_writes": partition_dir is not None,
        },
        "metrics": {
            key + "_per_request": value / config.requests
            for key, value in totals.items()
        },
        "performance": {
            "seconds": elapsed,
            "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
        },
        "evidence_boundary": (
            "Bounded-memory deterministic replay of the synthetic Supply V4 world. "
            "It validates scale and partition invariance, not production lift."
        ),
    }
