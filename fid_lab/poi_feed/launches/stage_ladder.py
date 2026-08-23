"""GPU POI distribution retrieval, coarse, fine, and mix experiments."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np

from ...feed_loop.cascade.contracts import BASE_RECALL_ROUTES
from ...feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)
from ...feed_loop.tensor_policies import (
    LOCAL_EXPANSION,
    LOCAL_INTENT_RANKER,
    LOCAL_RETARGET,
    LOCAL_SEARCH,
    LOCAL_STATIC,
)


def _policies():
    return {
        "retrieval": (
            replace(
                LOCAL_INTENT_RANKER, name="poi_retrieval_base_6route",
                enabled_routes=BASE_RECALL_ROUTES,
            ),
            replace(
                LOCAL_INTENT_RANKER, name="poi_retrieval_post_search_7route",
                enabled_routes=(*BASE_RECALL_ROUTES, "post_search"),
            ),
            replace(
                LOCAL_INTENT_RANKER, name="poi_retrieval_retarget_8route",
                enabled_routes=(*BASE_RECALL_ROUTES, "post_search", "retarget"),
            ),
        ),
        "coarse": (
            replace(LOCAL_INTENT_RANKER, name="poi_coarse_quality", coarse_model="quality_only"),
            replace(LOCAL_INTENT_RANKER, name="poi_coarse_lr", coarse_model="lr_v1"),
            replace(
                LOCAL_INTENT_RANKER, name="poi_coarse_dcnv2_distilled",
                coarse_model="dcnv2_distilled",
            ),
        ),
        "fine": (
            replace(LOCAL_STATIC, name="poi_fine_static"),
            replace(LOCAL_SEARCH, name="poi_fine_post_search"),
            replace(LOCAL_RETARGET, name="poi_fine_retarget"),
            replace(LOCAL_INTENT_RANKER, name="poi_fine_intent_ranker"),
        ),
        "mix": (
            replace(
                LOCAL_INTENT_RANKER, name="poi_mix_feed_guarded",
                mix_local_weight=0.0,
            ),
            replace(
                LOCAL_INTENT_RANKER, name="poi_mix_local_expansion",
                mix_local_weight=LOCAL_EXPANSION.mix_local_weight,
            ),
        ),
    }


def _gate(metrics, primary):
    primary_interval = metrics[primary]["confidence_interval"]
    lt_interval = metrics["lt_value_per_user"]["confidence_interval"]
    stay_interval = metrics["stay_per_exposure"]["confidence_interval"]
    negative_interval = metrics["negative_rate"]["confidence_interval"]
    gates = {
        "local_primary_positive": primary_interval[0] > 0,
        "platform_lt_nonnegative": lt_interval[0] >= 0,
        "stay_guardrail": stay_interval[0] >= -0.02,
        "negative_guardrail": negative_interval[1] <= 0.0005,
    }
    return gates, "pass" if all(gates.values()) else "hold_or_reject"


def _semantic_key(policy):
    payload = asdict(policy)
    payload.pop("name")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _run_world(config, policy, cache):
    key = _semantic_key(policy)
    cache_hit = key in cache
    if not cache_hit:
        cache[key] = run_tensor_feed(config, policy)
    return cache[key], cache_hit


def _request_trace(report):
    requests = {}
    for row in report["request_candidate_trace"]["rows"]:
        request = requests.setdefault(
            row["request_id"], {"recalled": set(), "coarse": set(), "exposed": None}
        )
        request["recalled"].add(row["candidate_id"])
        if row["coarse_pass"]:
            request["coarse"].add(row["candidate_id"])
        if row["exposed"]:
            request["exposed"] = row["candidate_id"]
    return requests


def _stage_delta(stage, control, treatment):
    before = _request_trace(control)
    after = _request_trace(treatment)
    common = sorted(set(before) & set(after))
    field = "recalled" if stage == "retrieval" else (
        "coarse" if stage == "coarse" else "exposed"
    )
    changed = sum(before[key][field] != after[key][field] for key in common)
    return {
        "audit_requests": len(common),
        "changed_requests": changed,
        "changed_rate": changed / max(len(common), 1),
        "compared_stage_output": field,
    }


def _seed_run(users, steps, seed, device):
    config = TensorFeedConfig(
        users=users, steps=steps, batch_users=min(users, 200_000),
        seed=seed, device=device, signal_version="kuairand-calibrated-v3",
        trace_users=16, trace_requests_per_user=steps,
    )
    report = {"config": asdict(config), "stages": {}}
    cache = {}
    primary = {
        "retrieval": "anchor_click_rate",
        "coarse": "anchor_click_rate",
        "fine": "local_value_tree_score_per_exposure",
        "mix": "local_value_tree_score_per_exposure",
    }
    for stage, policies in _policies().items():
        worlds = {}
        cache_hits = {}
        for policy in policies:
            worlds[policy.name], cache_hits[policy.name] = _run_world(
                config, policy, cache
            )
        launches = []
        for control, treatment in zip(policies, policies[1:]):
            metrics = combine_tensor_ab(
                worlds[control.name], worlds[treatment.name]
            )
            gates, decision = _gate(metrics, primary[stage])
            stage_delta = _stage_delta(
                stage, worlds[control.name], worlds[treatment.name]
            )
            primary_effect = (
                metrics[primary[stage]]["treatment_mean"]
                - metrics[primary[stage]]["control_mean"]
            )
            lt_effect = (
                metrics["lt_value_per_user"]["treatment_mean"]
                - metrics["lt_value_per_user"]["control_mean"]
            )
            gates["audit_stage_output_changed"] = (
                stage_delta["changed_requests"] > 0
            )
            gates["measurable_output_effect"] = (
                abs(primary_effect) > 1e-15 or abs(lt_effect) > 1e-15
            )
            if not gates["measurable_output_effect"]:
                decision = "reject_no_effect"
            launches.append({
                "control": control.name,
                "treatment": treatment.name,
                "primary_metric": primary[stage],
                "metrics": metrics,
                "gates": gates,
                "decision": decision,
                "stage_delta": stage_delta,
                "candidate_graph": worlds[treatment.name]["candidate_graph"],
                "performance": {
                    **worlds[treatment.name]["performance"],
                    "semantic_cache_hit": cache_hits[treatment.name],
                },
            })
        report["stages"][stage] = launches
    report["execution"] = {
        "declared_policy_arms": sum(len(value) for value in _policies().values()),
        "unique_semantic_worlds": len(cache),
    }
    return report


def _aggregate_launch(stage, treatment, seeds):
    rows = [
        next(
            row for row in report["stages"][stage]
            if row["treatment"] == treatment
        )
        for report in seeds
    ]
    primary = rows[0]["primary_metric"]
    effects = np.asarray([
        row["metrics"][primary]["treatment_mean"]
        - row["metrics"][primary]["control_mean"]
        for row in rows
    ])
    lt_effects = np.asarray([
        row["metrics"]["lt_value_per_user"]["treatment_mean"]
        - row["metrics"]["lt_value_per_user"]["control_mean"]
        for row in rows
    ])
    seed_passes = sum(row["decision"] == "pass" for row in rows)
    changed_seeds = sum(
        row["gates"]["audit_stage_output_changed"] for row in rows
    )
    effective_seeds = sum(
        row["gates"]["measurable_output_effect"] for row in rows
    )
    if effective_seeds == 0:
        decision = "reject_no_effect"
    elif seed_passes == len(rows):
        decision = "pass_all_seeds"
    elif effects.mean() < 0 or lt_effects.mean() < 0:
        decision = "reject_mean_regression"
    else:
        decision = "hold_seed_instability"
    return {
        "stage": stage,
        "control": rows[0]["control"],
        "treatment": treatment,
        "primary_metric": primary,
        "decision": decision,
        "seed_passes": seed_passes,
        "changed_seeds": changed_seeds,
        "effective_seeds": effective_seeds,
        "primary_effect": {
            "mean": float(effects.mean()),
            "std": float(effects.std(ddof=1 if len(effects) > 1 else 0)),
            "per_seed": effects.tolist(),
        },
        "platform_lt_effect_per_user": {
            "mean": float(lt_effects.mean()),
            "std": float(lt_effects.std(ddof=1 if len(lt_effects) > 1 else 0)),
            "per_seed": lt_effects.tolist(),
        },
        "seed_reports": rows,
    }


def run_poi_stage_ladder(users=1_000_000, steps=24,
                         seeds=(20260824, 20260825, 20260826),
                         device="cuda:0"):
    seed_reports = [
        _seed_run(users, steps, seed, device) for seed in seeds
    ]
    policies = _policies()
    launches = [
        _aggregate_launch(stage, policy.name, seed_reports)
        for stage, variants in policies.items()
        for policy in variants[1:]
    ]
    return {
        "schema": "poi-distribution-stage-launch-review-v2",
        "seeds": list(seeds),
        "users_per_seed": users,
        "steps": steps,
        "launches": launches,
        "stage_decisions": {
            stage: [
                row["decision"] for row in launches if row["stage"] == stage
            ]
            for stage in policies
        },
        "execution": {
            "comparison_order": "adjacent_evolution_steps",
            "declared_policy_arms_per_seed": seed_reports[0]["execution"][
                "declared_policy_arms"
            ],
            "unique_semantic_worlds_per_seed": seed_reports[0]["execution"][
                "unique_semantic_worlds"
            ],
            "semantic_world_reuse": True,
        },
        "evidence_boundary": (
            "GPU common-random V3 Local behavior simulation; validates stage "
            "mechanics and synthetic effect recovery, not live business lift."
        ),
    }
