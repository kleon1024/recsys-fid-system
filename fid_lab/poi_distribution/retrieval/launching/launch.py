"""GPU shadow/replay and multi-seed A/B for POI ANN retrieval."""

from __future__ import annotations

from dataclasses import asdict

from ....feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    combine_tensor_ab,
    combine_tensor_counterfactual_ab,
    run_tensor_feed,
)
from ....feed_loop.tensor_policies import PERSONALIZED
from ..policy import TensorPoiRetrievalPolicy
from .analysis import aggregate


def _seed_run(bundles, users, steps, seed, device):
    config = TensorFeedConfig(
        users=users,
        steps=steps,
        batch_users=min(users, 200_000),
        seed=seed,
        catalog_seed=20260823,
        retain_paired_user_metrics=True,
        device=device,
        signal_version="kuairand-local-neural-v4",
        trace_users=16,
        trace_requests_per_user=steps,
    )
    policies = (
        PERSONALIZED,
        TensorPoiRetrievalPolicy(bundles["two_tower"]),
        TensorPoiRetrievalPolicy(bundles["multi_interest"]),
    )
    worlds = {policy.name: run_tensor_feed(config, policy) for policy in policies}
    comparisons = []
    for control, treatment in zip(policies, policies[1:]):
        paired = combine_tensor_counterfactual_ab(
            worlds[control.name], worlds[treatment.name]
        )
        online = combine_tensor_ab(worlds[control.name], worlds[treatment.name])
        comparisons.append({
            "control": control.name,
            "treatment": treatment.name,
            "metrics": paired,
            "online_disjoint_metrics": online,
            "candidate_graph": worlds[treatment.name]["candidate_graph"],
            "performance": worlds[treatment.name]["performance"],
        })
    return {"config": asdict(config), "comparisons": comparisons}


def run_retrieval_launch(
    bundles,
    training_report,
    users=500_000,
    steps=24,
    seeds=(20260824, 20260825, 20260826),
    device="cuda:0",
):
    seed_reports = [
        _seed_run(bundles, users, steps, seed, device) for seed in seeds
    ]
    offline = {
        "poi_ann_rule": training_report["semantic_baseline"],
        **{
            name: training_report["models"][name]["offline"]
            for name in ("two_tower", "multi_interest")
        },
    }
    launches = [
        aggregate(0, seed_reports, offline["poi_ann_rule"], offline["two_tower"]),
        aggregate(1, seed_reports, offline["two_tower"], offline["multi_interest"]),
    ]
    return {
        "schema": "poi-retrieval-v4-launch-review-v1",
        "users_per_seed": users,
        "steps": steps,
        "seeds": list(seeds),
        "fixed_budget": {
            "catalog": training_report["corpus"],
            "ann_pool_per_request": 24,
            "ann_route_top_k": 8,
            "final_merged_candidates": 48,
        },
        "launches": launches,
        "model_artifacts": {
            name: training_report["models"][name]["artifact"]
            for name in ("two_tower", "multi_interest")
        },
        "seed_configs": [report["config"] for report in seed_reports],
        "evaluation_protocol": {
            "shadow_replay": "same users and common-random event streams",
            "online_ab": "orthogonal disjoint hash cells",
            "changed_owner": "ANN route scorer only",
        },
        "evidence_boundary": "Synthetic Neural Local V4 simulator evidence only.",
    }
