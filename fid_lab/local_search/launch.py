"""Two-phase Local Search retrieval, ranking, and end-to-end Launch Reviews."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from ..launches.statistics import aggregate_launch_rows, paired_metric
from ..value import DEFAULT_LT_CONFIG
from .contracts import LocalSearchConfig
from .models.ranking import load_ranker, save_ranker, train_rankers
from .models.retrieval import load_retriever, save_retriever, train_retriever
from .simulation.features import FEATURE_NAMES, candidate_features, rule_score
from .simulation.response import simulate_response
from .simulation.retrieval import retrieve
from .simulation.world import build_world


RETRIEVAL_VARIANTS = {
    "lexical_geo": ("lexical", "geo"),
    "semantic_tower": ("lexical", "geo", "semantic_tower"),
    "journey_history": ("lexical", "geo", "semantic_tower", "history"),
    "retarget_full": (
        "lexical", "geo", "semantic_tower", "history", "retarget"
    ),
}
FINE_COMPARISONS = (
    ("rule", "linear"),
    ("rule", "xgboost_pairwise"),
    ("linear", "wide_deep"),
    ("wide_deep", "din"),
    ("din", "transformer_mmoe"),
)


def _outcomes(response):
    stay_value = DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
    active_value = DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    success = response["detail"] | response["saved"] | response["ordered"]
    return {
        "click_rate": response["clicked"].float(),
        "query_success_rate": success.float(),
        "detail_rate": response["detail"].float(),
        "save_rate": response["saved"].float(),
        "order_rate": response["ordered"].float(),
        "closed_loop_order_rate": response["closed_loop_order"].float(),
        "open_loop_order_rate": response["open_loop_order"].float(),
        "pixel_observability": response["pixel_observable"].float(),
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


def _sample_profile(response):
    examples = response["examples"]
    labels, masks = examples.labels, examples.label_masks
    audit_rows = min(len(examples.request_id), 2_048)
    digest = sha256()
    for tensor in (
        examples.request_id[:audit_rows], examples.poi_ids[:audit_rows],
        examples.route_bits[:audit_rows], examples.exposed_indices[:audit_rows],
        examples.position_propensity[:audit_rows], labels[:audit_rows],
        masks[:audit_rows], examples.served_scores[:audit_rows],
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    task_rates = {}
    observable_rates = {}
    for index, task in enumerate(("click", "detail", "save", "order")):
        observable = masks[:, :, index].sum().clamp_min(1.0)
        task_rates[task] = float((labels[:, :, index] * masks[:, :, index]).sum()
                                 / observable)
        observable_rates[task] = float(masks[:, :, index].mean())
    return {
        "requests": int(len(examples.request_id)),
        "candidate_rows": int(examples.poi_ids.numel()),
        "exposed_rows": int(examples.exposed_indices.numel()),
        "label_positive_rates": task_rates,
        "label_observable_rates": observable_rates,
        "position_propensity": examples.position_propensity[0].tolist(),
        "open_loop_order_share": float(
            response["open_loop_order"].sum()
            / response["ordered"].sum().clamp_min(1)
        ),
        "audit_rows": audit_rows,
        "audit_sha256": digest.hexdigest(),
    }


def _evaluate(control, treatment, recall=None):
    metrics = {
        name: paired_metric(control[name], treatment[name]) for name in control
    }
    gates = {
        "query_success_positive": (
            metrics["query_success_rate"]["confidence_interval"][0] > 0
        ),
        "platform_lt_nonnegative": (
            metrics["platform_lt_per_request"]["confidence_interval"][0] >= 0
        ),
        "order_nonnegative": metrics["order_rate"]["confidence_interval"][0]
        >= -0.0002,
        "risk_guardrail": (
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


def _campaign(stage, order, worlds, start, recalls=None):
    rows = []
    control = order[0]
    for treatment in order[1:]:
        recall = None if recalls is None else (
            recalls[control][start:].float(), recalls[treatment][start:].float()
        )
        metrics, gates, decision = _evaluate(
            _slice(worlds[control], start), _slice(worlds[treatment], start), recall
        )
        rows.append({
            "stage": stage, "control": control, "treatment": treatment,
            "metrics": metrics, "gates": gates, "decision": decision,
            "promoted": decision == "pass",
        })
    return rows


def _fine_campaign(worlds, start):
    rows = []
    for control, treatment in FINE_COMPARISONS:
        metrics, gates, decision = _evaluate(
            _slice(worlds[control], start), _slice(worlds[treatment], start)
        )
        rows.append({
            "stage": "fine", "control": control, "treatment": treatment,
            "metrics": metrics, "gates": gates, "decision": decision,
            "promoted": decision == "pass",
        })
    return rows


def _rank_worlds(world, candidates, features, bundles):
    semantic = world.catalog.semantic[candidates.poi_ids]
    scores = {"rule": rule_score(features)}
    scores.update({
        name: bundle.score(features, semantic, world.requests.history_sequence)
        for name, bundle in bundles.items()
    })
    return {
        name: _outcomes(simulate_response(world, candidates, score))
        for name, score in scores.items()
    }


def _artifact_evidence(
    config, retrieval_bundle, rankers, candidates, features, world, artifact_dir
):
    report = {"retrieval": dict(retrieval_bundle.offline), "rankers": {}}
    if artifact_dir is not None:
        retrieval_path = artifact_dir / f"seed-{config.seed}-two-tower.pt"
        report["retrieval"]["artifact"] = save_retriever(
            retrieval_bundle, retrieval_path
        )
        loaded = load_retriever(retrieval_path, config.device)
        before = retrieval_bundle.model.encode_query(
            world.requests.observed_query[:64].new_zeros(
                64, retrieval_bundle.model.query_width
            )
        )
        after = loaded.model.encode_query(
            world.requests.observed_query[:64].new_zeros(
                64, loaded.model.query_width
            )
        )
        report["retrieval"]["serialized_replay_max_abs_delta"] = float(
            (before - after).abs().max()
        )
    semantic = world.catalog.semantic[candidates.poi_ids]
    for name, bundle in rankers.items():
        evidence = dict(bundle.offline)
        if artifact_dir is not None:
            suffix = ".json" if name == "xgboost_pairwise" else ".pt"
            path = artifact_dir / f"seed-{config.seed}-{name}{suffix}"
            evidence["artifact"] = save_ranker(bundle, path, config)
            loaded = load_ranker(path, config.device)
            before = bundle.score(
                features[:64], semantic[:64], world.requests.history_sequence[:64]
            )
            after = loaded.score(
                features[:64], semantic[:64], world.requests.history_sequence[:64]
            )
            evidence["serialized_replay_max_abs_delta"] = float(
                (before - after).abs().max()
            )
        report["rankers"][name] = evidence
    return report


def _end_to_end(control, candidate_name, worlds, start):
    rows = []
    for ranker, values in worlds.items():
        metrics, gates, decision = _evaluate(control, _slice(values, start))
        rows.append({
            "stage": "end_to_end",
            "control": "lexical_geo_plus_rule",
            "treatment": f"{candidate_name}_plus_{ranker}",
            "metrics": metrics, "gates": gates, "decision": decision,
        })
    return rows


def run_local_search_seed(
    config=LocalSearchConfig(), artifact_dir=None, forced_retrieval=None,
    candidate_only=False,
):
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA Local Search launch requested but unavailable")
    started = perf_counter()
    world = build_world(config)
    baseline = retrieve(world, RETRIEVAL_VARIANTS["lexical_geo"])
    baseline_features = candidate_features(world, baseline)
    baseline_response = simulate_response(
        world, baseline, rule_score(baseline_features)
    )
    retrieval_bundle = train_retriever(config, world, baseline_response)
    candidate_sets = {
        name: retrieve(world, routes, retrieval_bundle.model)
        for name, routes in RETRIEVAL_VARIANTS.items()
    }
    features = {
        name: candidate_features(world, candidates)
        for name, candidates in candidate_sets.items()
    }
    candidate_worlds = {
        name: _outcomes(simulate_response(
            world, candidate_sets[name], rule_score(features[name])
        )) for name in RETRIEVAL_VARIANTS
    }
    test_start = int(config.requests * 0.85)
    candidate_rows = _campaign(
        "retrieval", tuple(RETRIEVAL_VARIANTS), candidate_worlds, test_start,
        {name: value.audit_oracle_recalled for name, value in candidate_sets.items()},
    )
    if candidate_only:
        return {
            "schema": "local-search-retrieval-phase-v1",
            "config": asdict(config), "launches": candidate_rows,
            "performance": {"seconds": perf_counter() - started},
        }
    retrieval_active = forced_retrieval or "lexical_geo"
    active_candidates = candidate_sets[retrieval_active]
    active_features = features[retrieval_active]
    logging = simulate_response(
        world, active_candidates, rule_score(active_features)
    )
    rankers = train_rankers(
        config, world, active_candidates, active_features, logging
    )
    fine_worlds = _rank_worlds(
        world, active_candidates, active_features, rankers
    )
    fine_rows = _fine_campaign(fine_worlds, test_start)
    control = _slice(_outcomes(baseline_response), test_start)
    end_rows = []
    for candidate_name, candidates in candidate_sets.items():
        worlds = fine_worlds if candidate_name == retrieval_active else _rank_worlds(
            world, candidates, features[candidate_name], rankers
        )
        end_rows.extend(_end_to_end(control, candidate_name, worlds, test_start))
    evidence = _artifact_evidence(
        config, retrieval_bundle, rankers, active_candidates, active_features,
        world, artifact_dir,
    )
    elapsed = perf_counter() - started
    return {
        "schema": "local-search-request-launch-ladder-v1",
        "config": asdict(config), "feature_names": FEATURE_NAMES,
        "logging_policy": "lexical_geo_plus_rule",
        "fine_training_policy": f"{retrieval_active}_plus_rule",
        "logging_contract": {
            "oracle_forced_into_candidates": False,
            "labels_only_on_actual_exposure": True,
            "position_propensity_logged": True,
            "open_loop_unobservable_order_masked": True,
            "retrieval_promotion_precedes_fresh_logging": True,
            "transaction_not_directly_exchanged_to_lt": True,
            "time_split": [0.70, 0.15, 0.15],
        },
        "models": evidence, "launches": [*candidate_rows, *fine_rows],
        "sample_profile": _sample_profile(logging),
        "end_to_end_candidates": end_rows,
        "performance": {
            "seconds": elapsed, "requests_per_second": config.requests / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if config.device.startswith("cuda") else 0
            ),
        },
        "evidence_boundary": (
            "Teacher-hidden synthetic Local Search journeys with position-biased "
            "exposure, closed/open-loop order observability, and platform LT "
            "components. External query, POI, and randomized search logs are still "
            "required for production authority."
        ),
    }


def _best_pass(rows, stage, default):
    choices = [
        row for row in rows
        if row["stage"] == stage and row["decision"] == "pass_all_seeds"
    ]
    return max(
        choices,
        key=lambda row: row["metrics"]["platform_lt_per_request"]["mean_effect"],
        default={"treatment": default},
    )["treatment"]


def _accepted_fine_ranker(rows):
    active = "rule"
    for row in rows:
        if row["control"] != active:
            continue
        if row["decision"] == "pass_all_seeds":
            active = row["treatment"]
    return active


def run_repeated_local_search(
    config=LocalSearchConfig(), seeds=(20260824, 20260825, 20260826),
    artifact_dir: Path | None = None,
):
    retrieval_reports = [
        run_local_search_seed(replace(config, seed=seed), candidate_only=True)
        for seed in seeds
    ]
    retrieval_count = len(RETRIEVAL_VARIANTS) - 1
    retrieval_rows = [
        aggregate_launch_rows(
            [report["launches"][index] for report in retrieval_reports],
            "query_success_rate", "platform_lt_per_request",
        ) for index in range(retrieval_count)
    ]
    retrieval_active = _best_pass(retrieval_rows, "retrieval", "lexical_geo")
    reports = [
        run_local_search_seed(
            replace(config, seed=seed), artifact_dir, retrieval_active
        ) for seed in seeds
    ]
    fine_rows = [
        aggregate_launch_rows(
            [report["launches"][index] for report in reports],
            "query_success_rate", "platform_lt_per_request",
        ) for index in range(retrieval_count, len(reports[0]["launches"]))
    ]
    fine_active = _accepted_fine_ranker(fine_rows)
    end_name = f"{retrieval_active}_plus_{fine_active}"
    end_rows = [
        next(row for row in report["end_to_end_candidates"]
             if row["treatment"] == end_name)
        for report in reports
    ]
    end_to_end = aggregate_launch_rows(
        end_rows, "query_success_rate", "platform_lt_per_request"
    )
    return {
        "schema": "local-search-request-launch-review-v1",
        "seeds": list(seeds), "config": asdict(config),
        "launches": [*retrieval_rows, *fine_rows, end_to_end],
        "release_state": {
            "retrieval": retrieval_active, "fine": fine_active,
            "end_to_end": (
                end_name if end_to_end["decision"] == "pass_all_seeds"
                else "lexical_geo_plus_rule"
            ),
        },
        "seed_reports": reports,
        "retrieval_seed_reports": retrieval_reports,
        "evidence_boundary": reports[0]["evidence_boundary"],
    }
