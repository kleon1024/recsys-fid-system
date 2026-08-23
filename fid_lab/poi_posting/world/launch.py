"""Candidate, fine-rank, and supply-to-Feed POI posting Launch Reviews."""

from __future__ import annotations

from dataclasses import asdict, replace
from math import erfc, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...value import DEFAULT_LT_CONFIG
from .contracts import PostingWorldConfig
from .generator import (
    FEATURE_NAMES,
    build_world,
    candidate_features,
    retrieve,
    rule_score,
    simulate_response,
)
from .models import (
    load_posting_bundle,
    save_posting_bundle,
    train_posting_models,
)


BASE_ROUTES = ("popular", "geo")
SEMANTIC_ROUTES = (*BASE_ROUTES, "semantic")
FULL_ROUTES = (*SEMANTIC_ROUTES, "history")


def _outcome_values(response):
    stay_rate = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_rate = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    published = response["published"].float()
    return {
        "selection_rate": response["selected"].float(),
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


def _paired_metric(control, treatment):
    delta = (treatment - control).double().cpu().numpy()
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / sqrt(len(delta)))
    control_mean = float(control.double().mean())
    treatment_mean = float(treatment.double().mean())
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "absolute_effect": mean,
        "relative_effect": (
            None if abs(control_mean) < 1e-12 else mean / abs(control_mean)
        ),
        "standard_error": standard_error,
        "confidence_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "p_value": erfc(abs(mean / max(standard_error, 1e-12)) / sqrt(2.0)),
    }


def _compare(control, treatment, extra=None):
    metrics = {
        name: _paired_metric(control[name], treatment[name])
        for name in control
    }
    if extra:
        metrics.update(extra)
    gates = {
        "publish_positive": metrics["publish_rate"]["confidence_interval"][0] > 0,
        "platform_lt_nonnegative": (
            metrics["platform_lt_per_request"]["confidence_interval"][0] >= 0
        ),
        "relevant_supply_nonnegative": (
            metrics["relevant_supply_per_request"]["confidence_interval"][0]
            >= -0.0002
        ),
        "negative_guardrail": (
            metrics["selected_content_negative_risk"]["confidence_interval"][1]
            <= 0.0002
        ),
    }
    return metrics, gates, "pass" if all(gates.values()) else "hold_or_reject"


def _request_slice(values, start):
    return {name: value[start:] for name, value in values.items()}


def _response_world(world, candidates, score):
    response = simulate_response(world, candidates, score)
    return response, _outcome_values(response)


def _launch_campaign(
    stage, variants, worlds, start, recall=None, fixed_control=False
):
    active = variants[0]
    launches = []
    for treatment in variants[1:]:
        control = variants[0] if fixed_control else active
        extra = None
        if recall is not None:
            extra = {
                "audit_oracle_recall": _paired_metric(
                    recall[control][start:].float(),
                    recall[treatment][start:].float(),
                )
            }
        metrics, gates, decision = _compare(
            _request_slice(worlds[control], start),
            _request_slice(worlds[treatment], start),
            extra,
        )
        if recall is not None:
            gates["recall_nonnegative"] = (
                metrics["audit_oracle_recall"]["confidence_interval"][0] >= 0
            )
            decision = "pass" if all(gates.values()) else "hold_or_reject"
        promoted = decision == "pass"
        launches.append({
            "stage": stage,
            "control": control,
            "treatment": treatment,
            "metrics": metrics,
            "gates": gates,
            "decision": decision,
            "promoted": promoted,
        })
        if promoted:
            active = treatment
    return launches, active


def _model_evidence(bundles, features, config, artifact_dir):
    report = {}
    for name, bundle in bundles.items():
        evidence = dict(bundle.offline)
        if artifact_dir is not None:
            path = artifact_dir / f"seed-{config.seed}-{name}.pt"
            evidence["artifact"] = save_posting_bundle(bundle, path, config)
            loaded = load_posting_bundle(path, config.device)
            before = bundle.score(features["popular_geo"][:256])
            after = loaded.score(features["popular_geo"][:256])
            evidence["serialized_replay_max_abs_delta"] = float(
                (before - after).abs().max()
            )
        report[name] = evidence
    return report


def _end_to_end_reviews(control, fine_worlds, start, candidate, active_model):
    treatment = _request_slice(fine_worlds[active_model], start)
    metrics, gates, decision = _compare(control, treatment)
    selected = {
        "stage": "end_to_end",
        "control": "popular_geo_plus_rule",
        "treatment": f"{candidate}_plus_{active_model}",
        "metrics": metrics,
        "gates": gates,
        "decision": decision,
        "promoted": decision == "pass",
    }
    candidates = []
    for name, values in fine_worlds.items():
        metrics, gates, decision = _compare(
            control, _request_slice(values, start)
        )
        candidates.append({
            "stage": "end_to_end",
            "control": "popular_geo_plus_rule",
            "treatment": f"{candidate}_plus_{name}",
            "metrics": metrics,
            "gates": gates,
            "decision": decision,
        })
    return selected, candidates


def run_posting_launch_ladder(
    config=PostingWorldConfig(), artifact_dir: Path | None = None
):
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA posting ladder requested but unavailable")
    started = perf_counter()
    world = build_world(config)
    candidate_sets = {
        "popular_geo": retrieve(world, BASE_ROUTES),
        "semantic_recall": retrieve(world, SEMANTIC_ROUTES),
        "history_recall": retrieve(world, FULL_ROUTES),
    }
    features = {
        name: candidate_features(world, candidates)
        for name, candidates in candidate_sets.items()
    }
    logging_candidates = candidate_sets["popular_geo"]
    logging_score = rule_score(features["popular_geo"])
    logging_response, _ = _response_world(
        world, logging_candidates, logging_score
    )
    bundles = train_posting_models(
        config,
        features["popular_geo"],
        logging_response["top_indices"],
        logging_response["labels"],
    )
    model_evidence = _model_evidence(
        bundles, features, config, artifact_dir
    )
    candidate_worlds = {}
    for name in candidate_sets:
        score = bundles["linear"].score(features[name])
        _, candidate_worlds[name] = _response_world(
            world, candidate_sets[name], score
        )
    test_start = int(config.requests * 0.85)
    candidate_launches, candidate_active = _launch_campaign(
        "candidate",
        ("popular_geo", "semantic_recall", "history_recall"),
        candidate_worlds,
        test_start,
        {name: value.audit_oracle_recalled for name, value in candidate_sets.items()},
        fixed_control=True,
    )
    active_candidates = candidate_sets[candidate_active]
    active_features = features[candidate_active]
    fine_scores = {"rule": rule_score(active_features)}
    fine_scores.update({
        name: bundle.score(active_features) for name, bundle in bundles.items()
    })
    fine_worlds = {
        name: _response_world(world, active_candidates, score)[1]
        for name, score in fine_scores.items()
    }
    fine_launches, fine_active = _launch_campaign(
        "fine",
        ("rule", "linear", "wide_deep", "mmoe"),
        fine_worlds,
        test_start,
    )
    control = _request_slice(
        _outcome_values(logging_response), test_start
    )
    end_to_end, end_to_end_candidates = _end_to_end_reviews(
        control, fine_worlds, test_start, candidate_active, fine_active
    )
    elapsed = perf_counter() - started
    return {
        "schema": "poi-posting-request-launch-ladder-v1",
        "config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "logging_policy": "popular_geo_plus_rule",
        "logging_contract": {
            "oracle_forced_into_candidates": False,
            "only_exposed_candidates_are_behavioral_training_rows": True,
            "publish_is_entire_space_selected_and_published": True,
            "teacher_uses_hidden_latent_draft": True,
            "models_use_noisy_observed_draft": True,
            "time_split": [0.70, 0.15, 0.15],
        },
        "models": model_evidence,
        "launches": [*candidate_launches, *fine_launches, end_to_end],
        "end_to_end_candidates": end_to_end_candidates,
        "release_state": {
            "candidate": candidate_active,
            "fine": fine_active,
            "end_to_end": end_to_end["treatment"] if end_to_end["promoted"] else (
                "popular_geo_plus_rule"
            ),
        },
        "performance": {
            "seconds": elapsed,
            "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if config.device.startswith("cuda") else 0
            ),
        },
        "evidence_boundary": (
            "Teacher-hidden multi-agent synthetic posting world. It validates "
            "request closure, model training, stage isolation, and effect recovery; "
            "supply remains outside external V4 authority until creator logs and "
            "randomized supply interventions are available."
        ),
    }


def _aggregate_rows(rows):
    controls = {row["control"] for row in rows}
    treatments = {row["treatment"] for row in rows}
    if len(controls) != 1 or len(treatments) != 1:
        return {
            "stage": rows[0]["stage"],
            "control": sorted(controls),
            "treatment": sorted(treatments),
            "decision": "hold_control_divergence",
            "seed_decisions": [row["decision"] for row in rows],
        }
    metric_summary = {}
    for name in rows[0]["metrics"]:
        effects = np.asarray([
            row["metrics"][name]["absolute_effect"] for row in rows
        ])
        metric_summary[name] = {
            "mean_effect": float(effects.mean()),
            "seed_std": float(effects.std(ddof=1 if len(effects) > 1 else 0)),
            "per_seed": effects.tolist(),
        }
    passed = [row["decision"] == "pass" for row in rows]
    primary = metric_summary["publish_rate"]["mean_effect"]
    platform_lt = metric_summary["platform_lt_per_request"]["mean_effect"]
    if all(passed):
        decision = "pass_all_seeds"
    elif primary < 0 or platform_lt < 0:
        decision = "reject_mean_regression"
    else:
        decision = "hold_seed_instability"
    return {
        "stage": rows[0]["stage"],
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "seed_passes": sum(passed),
        "seed_decisions": [row["decision"] for row in rows],
        "metrics": metric_summary,
    }


def run_repeated_posting_launch_ladder(
    config=PostingWorldConfig(),
    seeds=(20260824, 20260825, 20260826),
    artifact_dir: Path | None = None,
):
    reports = [
        run_posting_launch_ladder(
            replace(config, seed=seed), artifact_dir
        ) for seed in seeds
    ]
    stage_rows = []
    for index in range(len(reports[0]["launches"]) - 1):
        stage_rows.append(_aggregate_rows([
            report["launches"][index] for report in reports
        ]))
    candidate_active = "popular_geo"
    for row in stage_rows:
        if row["stage"] == "candidate" and row["decision"] == "pass_all_seeds":
            candidate_active = row["treatment"]
    fine_active = "rule"
    for row in stage_rows:
        if row["stage"] != "fine":
            continue
        if row["control"] != fine_active:
            row["decision"] = "hold_global_control_mismatch"
        elif row["decision"] == "pass_all_seeds":
            fine_active = row["treatment"]
    end_name = f"{candidate_active}_plus_{fine_active}"
    end_rows = []
    for report in reports:
        end_rows.append(next(
            row for row in report["end_to_end_candidates"]
            if row["treatment"] == end_name
        ))
    end_to_end = _aggregate_rows(end_rows)
    launches = [*stage_rows, end_to_end]
    return {
        "schema": "poi-posting-request-launch-review-v2",
        "seeds": list(seeds),
        "config": asdict(config),
        "launches": launches,
        "release_state": {
            "candidate": candidate_active,
            "fine": fine_active,
            "end_to_end": (
                end_name
                if end_to_end["decision"] == "pass_all_seeds"
                else "popular_geo_plus_rule"
            ),
        },
        "seed_reports": reports,
        "evidence_boundary": reports[0]["evidence_boundary"],
    }
