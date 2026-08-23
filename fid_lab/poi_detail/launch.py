"""Adjacent model and page-level POI Detail Launch Reviews."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch

from ..launches.statistics import aggregate_launch_rows, paired_metric
from ..value import DEFAULT_LT_CONFIG
from .contracts import PoiDetailConfig
from .models.training import load_bundle, save_bundle, train_families
from .simulation.candidates import build_candidates
from .simulation.features import (
    FEATURE_NAMES, candidate_features, candidate_semantic, rule_score,
)
from .simulation.response import simulate_response
from .simulation.world import build_world


COMPARISONS = (
    ("rule", "linear"),
    ("linear", "wide_deep"),
    ("wide_deep", "specialized"),
)


def _outcomes(response):
    stay_value = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_value = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    return {
        "page_action_rate": response["clicked"].float(),
        "deep_action_rate": response["deep"].float(),
        "transaction_rate": response["transaction"].float(),
        "negative_rate": response["negative"].float(),
        "related_selection_rate": (response["selected_kind"] == 0).float(),
        "product_selection_rate": (response["selected_kind"] == 1).float(),
        "review_selection_rate": (response["selected_kind"] == 2).float(),
        "stay_seconds_per_request": response["stay_seconds"],
        "active_day_per_request": response["active_day"],
        "selected_content_risk": response["selected_risk"],
        "platform_lt_per_request": (
            response["stay_seconds"] / 60.0 * stay_value
            + response["active_day"] * active_value
        ),
    }


def _slice(values, start):
    return {name: value[start:] for name, value in values.items()}


def _evaluate(control, treatment):
    metrics = {
        name: paired_metric(control[name], treatment[name]) for name in control
    }
    gates = {
        "deep_action_positive": (
            metrics["deep_action_rate"]["confidence_interval"][0] > 0
        ),
        "platform_lt_nonnegative": (
            metrics["platform_lt_per_request"]["confidence_interval"][0] >= 0
        ),
        "transaction_nonnegative": (
            metrics["transaction_rate"]["confidence_interval"][0] >= -0.0002
        ),
        "negative_guardrail": (
            metrics["negative_rate"]["confidence_interval"][1] <= 0.0002
        ),
        "risk_guardrail": (
            metrics["selected_content_risk"]["confidence_interval"][1]
            <= 0.0002
        ),
    }
    return metrics, gates, "pass" if all(gates.values()) else "hold_or_reject"


def _launch_rows(worlds, start):
    rows = []
    for control, treatment in COMPARISONS:
        metrics, gates, decision = _evaluate(
            _slice(worlds[control], start), _slice(worlds[treatment], start)
        )
        rows.append({
            "stage": "fine", "control": control, "treatment": treatment,
            "metrics": metrics, "gates": gates, "decision": decision,
            "promoted": decision == "pass",
        })
    return rows


def _model_evidence(
    config, bundles, features, semantic, history, kinds, artifact_dir
):
    report = {}
    for name, bundle in bundles.items():
        evidence = dict(bundle.offline)
        if artifact_dir is not None:
            path = artifact_dir / f"seed-{config.seed}-{name}.pt"
            evidence["artifact"] = save_bundle(bundle, path, config)
            loaded = load_bundle(path, config.device)
            before = bundle.score(
                features[:64], semantic[:64], history[:64], kinds[:64]
            )
            after = loaded.score(
                features[:64], semantic[:64], history[:64], kinds[:64]
            )
            evidence["serialized_replay_max_abs_delta"] = float(
                (before - after).abs().max()
            )
        report[name] = evidence
    return report


def _sample_profile(response, config):
    labels, masks = response["labels"], response["label_masks"]
    task_rates = {}
    for index, task in enumerate(("click", "deep_action", "transaction", "negative")):
        observable = masks[:, :, index].sum().clamp_min(1.0)
        task_rates[task] = float((labels[:, :, index] * masks[:, :, index]).sum()
                                 / observable)
    return {
        "requests": config.requests,
        "candidate_rows": int(config.requests * config.candidates),
        "exposed_rows": int(config.requests * config.exposed),
        "label_positive_rates": task_rates,
        "module_candidate_rows": {
            "related_poi": config.requests * config.candidates_per_module,
            "product": config.requests * config.candidates_per_module,
            "review": config.requests * config.candidates_per_module,
        },
        "exposure_quota": {
            "related_poi": config.exposed_related,
            "product": config.exposed_product,
            "review": config.exposed_review,
        },
    }


def _end_to_end_candidates(worlds, start):
    rows = []
    control = _slice(worlds["rule"], start)
    for name, values in worlds.items():
        if name == "rule":
            metrics = {
                metric: paired_metric(values, values)
                for metric, values in control.items()
            }
            gates, decision = {"baseline": True}, "baseline"
        else:
            metrics, gates, decision = _evaluate(control, _slice(values, start))
        rows.append({
            "stage": "end_to_end", "control": "quota_mix_plus_rule",
            "treatment": f"quota_mix_plus_{name}",
            "metrics": metrics, "gates": gates, "decision": decision,
        })
    return rows


def run_poi_detail_seed(config=PoiDetailConfig(), artifact_dir=None):
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA POI Detail launch requested but unavailable")
    started = perf_counter()
    world = build_world(config)
    candidates = build_candidates(world)
    features = candidate_features(world, candidates)
    semantic = candidate_semantic(world, candidates)
    rule_response = simulate_response(world, candidates, rule_score(features))
    bundles = train_families(
        config, world, candidates, features, semantic, rule_response
    )
    scores = {"rule": rule_score(features)}
    scores.update({
        name: bundle.score(
            features, semantic, world.requests.history_sequence,
            candidates.module_kind,
        ) for name, bundle in bundles.items()
    })
    worlds = {
        name: _outcomes(simulate_response(world, candidates, score))
        for name, score in scores.items()
    }
    test_start = int(config.requests * 0.85)
    launches = _launch_rows(worlds, test_start)
    elapsed = perf_counter() - started
    return {
        "schema": "poi-detail-request-launch-ladder-v1",
        "config": asdict(config), "feature_names": FEATURE_NAMES,
        "logging_policy": "quota_mix_plus_rule",
        "logging_contract": {
            "one_page_request_identity": True,
            "module_models_have_separate_weights": True,
            "labels_only_on_exposed_candidates": True,
            "review_transaction_masked": True,
            "fixed_module_exposure_quota": True,
            "time_split": [0.70, 0.15, 0.15],
        },
        "sample_profile": _sample_profile(rule_response, config),
        "models": _model_evidence(
            config, bundles, features, semantic,
            world.requests.history_sequence, candidates.module_kind, artifact_dir,
        ),
        "launches": launches,
        "end_to_end_candidates": _end_to_end_candidates(worlds, test_start),
        "performance": {
            "seconds": elapsed, "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if config.device.startswith("cuda") else 0
            ),
        },
        "evidence_boundary": (
            "Teacher-hidden synthetic POI Detail page with shared exposure and "
            "separate related-POI, product, and review weights. External page, "
            "transaction, and review logs remain required for production authority."
        ),
    }


def _accepted_family(rows):
    active = "rule"
    for row in rows:
        if row["control"] == active and row["decision"] == "pass_all_seeds":
            active = row["treatment"]
    return active


def run_repeated_poi_detail(
    config=PoiDetailConfig(), seeds=(20260824, 20260825, 20260826),
    artifact_dir: Path | None = None,
):
    reports = [
        run_poi_detail_seed(replace(config, seed=seed), artifact_dir)
        for seed in seeds
    ]
    fine_rows = [
        aggregate_launch_rows(
            [report["launches"][index] for report in reports],
            "deep_action_rate", "platform_lt_per_request",
        ) for index in range(len(COMPARISONS))
    ]
    active = _accepted_family(fine_rows)
    end_name = f"quota_mix_plus_{active}"
    end_rows = [
        next(row for row in report["end_to_end_candidates"]
             if row["treatment"] == end_name)
        for report in reports
    ]
    end_to_end = aggregate_launch_rows(
        end_rows, "deep_action_rate", "platform_lt_per_request"
    )
    return {
        "schema": "poi-detail-request-launch-review-v1",
        "seeds": list(seeds), "config": asdict(config),
        "launches": [*fine_rows, end_to_end],
        "release_state": {
            "fine": active,
            "end_to_end": (
                end_name if end_to_end["decision"] == "pass_all_seeds"
                else "quota_mix_plus_rule"
            ),
        },
        "seed_reports": reports,
        "evidence_boundary": reports[0]["evidence_boundary"],
    }
