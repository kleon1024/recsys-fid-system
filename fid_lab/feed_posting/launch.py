"""Stage-isolated Feed-posting candidate, rank, and end-to-end reviews."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch

from ..launches.statistics import aggregate_launch_rows, paired_metric
from ..value import DEFAULT_LT_CONFIG
from .contracts import FeedPostingConfig
from .simulation.features import FEATURE_NAMES, candidate_features, rule_score
from .simulation.response import simulate_response
from .simulation.retrieval import retrieve
from .simulation.world import build_world
from .models import load_bundle, save_bundle, train_models


CANDIDATE_VARIANTS = {
    "trending_i2i": ("trending", "i2i"),
    "creator_history": ("trending", "i2i", "creator_history"),
    "semantic_full": ("trending", "i2i", "creator_history", "semantic"),
}
MODEL_ORDER = ("rule", "linear", "wide_deep", "din", "transformer_mmoe")


def _outcomes(response):
    published = response["published"].float()
    stay_value = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_value = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    return {
        "prompt_click_rate": response["clicked"].float(),
        "create_start_rate": response["created"].float(),
        "publish_rate": published,
        "quality_supply_per_request": published * response["quality_potential"],
        "negative_per_request": published * response["content_risk"],
        "selected_content_risk": response["content_risk"],
        "feed_stay_seconds_per_request": response["feed_stay_seconds"],
        "feed_active_day_per_request": response["feed_active_day"],
        "platform_lt_per_request": (
            response["feed_stay_seconds"] / 60.0 * stay_value
            + response["feed_active_day"] * active_value
        ),
    }


def _slice(values, start):
    return {name: value[start:] for name, value in values.items()}


def _evaluate(control, treatment, recall=None):
    metrics = {
        name: paired_metric(control[name], treatment[name]) for name in control
    }
    gates = {
        "publish_positive": metrics["publish_rate"]["confidence_interval"][0] > 0,
        "platform_lt_nonnegative": (
            metrics["platform_lt_per_request"]["confidence_interval"][0] >= 0
        ),
        "quality_supply_nonnegative": (
            metrics["quality_supply_per_request"]["confidence_interval"][0]
            >= -0.0002
        ),
        "content_risk_guardrail": (
            metrics["selected_content_risk"]["confidence_interval"][1]
            <= 0.0002
        ),
    }
    if recall is not None:
        metrics["audit_oracle_recall"] = paired_metric(*recall)
        gates["recall_nonnegative"] = (
            metrics["audit_oracle_recall"]["confidence_interval"][0] >= 0
        )
    decision = "pass" if all(gates.values()) else "hold_or_reject"
    return metrics, gates, decision


def _campaign(stage, order, worlds, start, recalls=None, fixed_control=False):
    active, rows = order[0], []
    for treatment in order[1:]:
        control = order[0] if fixed_control else active
        recall = None if recalls is None else (
            recalls[control][start:].float(), recalls[treatment][start:].float()
        )
        metrics, gates, decision = _evaluate(
            _slice(worlds[control], start), _slice(worlds[treatment], start), recall
        )
        promoted = decision == "pass"
        rows.append({
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
    return rows, active


def _model_evidence(bundles, features, semantic, history, config, artifact_dir):
    report = {}
    for name, bundle in bundles.items():
        evidence = dict(bundle.offline)
        if artifact_dir is not None:
            path = artifact_dir / f"seed-{config.seed}-{name}.pt"
            evidence["artifact"] = save_bundle(bundle, path, config)
            loaded = load_bundle(path, config.device)
            before = bundle.score(features[:128], semantic[:128], history[:128])
            after = loaded.score(features[:128], semantic[:128], history[:128])
            evidence["serialized_replay_max_abs_delta"] = float(
                (before - after).abs().max()
            )
        report[name] = evidence
    return report


def _rank_worlds(world, candidates, features, bundles):
    semantic = world.catalog.semantic[candidates.prompt_ids]
    scores = {"rule": rule_score(features)}
    scores.update({
        name: bundle.score(features, semantic, world.requests.feed_sequence)
        for name, bundle in bundles.items()
    })
    return {
        name: _outcomes(simulate_response(world, candidates, score))
        for name, score in scores.items()
    }


def _train_for_policy(config, world, candidates, features, response, model_names=None):
    semantic = world.catalog.semantic[candidates.prompt_ids]
    arguments = (
        config, features, semantic, world.requests.feed_sequence,
        response["top_indices"], response["labels"],
    )
    return train_models(*arguments) if model_names is None else train_models(
        *arguments, model_names=model_names
    )


def _end_to_end_rows(control, candidate_name, fine_worlds, start, selected):
    rows = []
    for model_name, values in fine_worlds.items():
        metrics, gates, decision = _evaluate(control, _slice(values, start))
        rows.append({
            "stage": "end_to_end",
            "control": "trending_i2i_plus_rule",
            "treatment": f"{candidate_name}_plus_{model_name}",
            "metrics": metrics,
            "gates": gates,
            "decision": decision,
            "promoted": decision == "pass" and (
                f"{candidate_name}_plus_{model_name}" == selected
            ),
        })
    return rows


def _candidate_phase_report(config, rows, active, started):
    elapsed = perf_counter() - started
    return {
        "schema": "feed-posting-candidate-phase-v1",
        "config": asdict(config),
        "launches": rows,
        "release_state": {"candidate": active},
        "performance": {
            "seconds": elapsed,
            "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if config.device.startswith("cuda") else 0
            ),
        },
    }


def run_feed_posting_launch(
    config=FeedPostingConfig(), artifact_dir=None, forced_candidate=None,
    candidate_only=False,
):
    """Materialize one seed once, then isolate candidate and ranking changes."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA Feed-posting ladder requested but unavailable")
    started = perf_counter()
    world = build_world(config)
    candidates = {
        name: retrieve(world, routes) for name, routes in CANDIDATE_VARIANTS.items()
    }
    features = {
        name: candidate_features(world, value) for name, value in candidates.items()
    }
    base_candidates = candidates["trending_i2i"]
    base_features = features["trending_i2i"]
    logging = simulate_response(world, base_candidates, rule_score(base_features))
    candidate_bundle = _train_for_policy(
        config, world, base_candidates, base_features, logging, ("linear",)
    )["linear"]
    candidate_worlds = {}
    for name, candidate_set in candidates.items():
        score = candidate_bundle.score(
            features[name], world.catalog.semantic[candidate_set.prompt_ids],
            world.requests.feed_sequence,
        )
        candidate_worlds[name] = _outcomes(
            simulate_response(world, candidate_set, score)
        )
    test_start = int(config.requests * 0.85)
    candidate_rows, candidate_active = _campaign(
        "candidate", tuple(CANDIDATE_VARIANTS), candidate_worlds, test_start,
        {name: value.audit_oracle_recalled for name, value in candidates.items()},
        fixed_control=True,
    )
    if forced_candidate is not None:
        if forced_candidate not in CANDIDATE_VARIANTS:
            raise ValueError(f"unknown forced candidate policy: {forced_candidate}")
        candidate_active = forced_candidate
    if candidate_only:
        return _candidate_phase_report(
            config, candidate_rows, candidate_active, started
        )
    rank_candidates = candidates[candidate_active]
    rank_features = features[candidate_active]
    rank_logging = simulate_response(
        world, rank_candidates, rule_score(rank_features)
    )
    bundles = _train_for_policy(
        config, world, rank_candidates, rank_features, rank_logging
    )
    rank_semantic = world.catalog.semantic[rank_candidates.prompt_ids]
    evidence = _model_evidence(
        bundles, rank_features, rank_semantic, world.requests.feed_sequence,
        config, artifact_dir,
    )
    fine_worlds = _rank_worlds(
        world, candidates[candidate_active], features[candidate_active], bundles
    )
    fine_rows, fine_active = _campaign(
        "fine", MODEL_ORDER, fine_worlds, test_start, fixed_control=True
    )
    control = _slice(_outcomes(logging), test_start)
    selected = f"{candidate_active}_plus_{fine_active}"
    end_rows = []
    for candidate_name in CANDIDATE_VARIANTS:
        candidate_fine_worlds = (
            fine_worlds if candidate_name == candidate_active else _rank_worlds(
                world, candidates[candidate_name], features[candidate_name], bundles
            )
        )
        end_rows.extend(_end_to_end_rows(
            control, candidate_name, candidate_fine_worlds, test_start, selected
        ))
    elapsed = perf_counter() - started
    return {
        "schema": "feed-posting-request-launch-ladder-v1",
        "config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "logging_policy": "trending_i2i_plus_rule",
        "fine_training_policy": f"{candidate_active}_plus_rule",
        "logging_contract": {
            "oracle_forced_into_candidates": False,
            "training_rows_are_exposed_candidates_only": True,
            "labels_follow_click_create_publish_cascade": True,
            "teacher_uses_hidden_creator_intent": True,
            "models_use_noisy_observed_state": True,
            "time_split": [0.70, 0.15, 0.15],
        },
        "models": evidence,
        "launches": [*candidate_rows, *fine_rows],
        "end_to_end_candidates": end_rows,
        "release_state": {"candidate": candidate_active, "fine": fine_active},
        "performance": {
            "seconds": elapsed,
            "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if config.device.startswith("cuda") else 0
            ),
        },
        "evidence_boundary": (
            "Teacher-hidden multi-agent synthetic Feed-to-creation world. It "
            "validates request closure, sequence-aware training, stage-isolated "
            "A/B effects, and supply-to-Feed outcomes; production readiness still "
            "requires creator logs and randomized supply interventions."
        ),
    }


def run_repeated_feed_posting_ladder(
    config=FeedPostingConfig(), seeds=(20260824, 20260825, 20260826),
    artifact_dir: Path | None = None,
):
    candidate_reports = [
        run_feed_posting_launch(replace(config, seed=seed), candidate_only=True)
        for seed in seeds
    ]
    candidate_count = len(CANDIDATE_VARIANTS) - 1
    candidate_rows = []
    for index in range(candidate_count):
        candidate_rows.append(aggregate_launch_rows(
            [report["launches"][index] for report in candidate_reports],
            "publish_rate", "platform_lt_per_request",
        ))
    candidate_options = [
        row for row in candidate_rows if row["decision"] == "pass_all_seeds"
    ]
    candidate_active = max(
        candidate_options,
        key=lambda row: row["metrics"]["platform_lt_per_request"]["mean_effect"],
        default={"treatment": "trending_i2i"},
    )["treatment"]
    reports = [
        run_feed_posting_launch(
            replace(config, seed=seed), artifact_dir, candidate_active
        ) for seed in seeds
    ]
    fine_rows = []
    for index in range(candidate_count, len(reports[0]["launches"])):
        fine_rows.append(aggregate_launch_rows(
            [report["launches"][index] for report in reports],
            "publish_rate", "platform_lt_per_request",
        ))
    fine_options = [
        row for row in fine_rows if row["decision"] == "pass_all_seeds"
    ]
    fine_active = max(
        fine_options,
        key=lambda row: row["metrics"]["platform_lt_per_request"]["mean_effect"],
        default={"treatment": "rule"},
    )["treatment"]
    end_name = f"{candidate_active}_plus_{fine_active}"
    end_rows = [
        next(row for row in report["end_to_end_candidates"]
             if row["treatment"] == end_name)
        for report in reports
    ]
    end_to_end = aggregate_launch_rows(
        end_rows, "publish_rate", "platform_lt_per_request"
    )
    return {
        "schema": "feed-posting-request-launch-review-v1",
        "seeds": list(seeds),
        "config": asdict(config),
        "launches": [*candidate_rows, *fine_rows, end_to_end],
        "release_state": {
            "candidate": candidate_active,
            "fine": fine_active,
            "end_to_end": (
                end_name if end_to_end["decision"] == "pass_all_seeds"
                else "trending_i2i_plus_rule"
            ),
        },
        "seed_reports": reports,
        "candidate_seed_reports": candidate_reports,
        "evidence_boundary": reports[0]["evidence_boundary"],
    }
