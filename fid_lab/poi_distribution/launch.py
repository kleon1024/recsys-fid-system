"""Multi-seed online A/B ladder for trained POI distribution models."""

from __future__ import annotations

from dataclasses import asdict, replace

from ..feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    combine_tensor_ab,
    combine_tensor_counterfactual_ab,
    run_tensor_feed,
)
from ..feed_loop.tensor_policies import PERSONALIZED
from .launch_analysis import PRIMARY_BY_STAGE, aggregate, gate
from .policy import TensorPoiModelPolicy


def _arms(bundles):
    coarse = (
        replace(
            PERSONALIZED, name="poi_coarse_quality_control",
            coarse_model="quality_only",
        ),
        TensorPoiModelPolicy(
            bundles["linear"], "coarse", 0.025,
            deployment_name="poi_coarse_linear",
        ),
        TensorPoiModelPolicy(
            bundles["dcnv2"], "coarse", 0.025,
            deployment_name="poi_coarse_dcnv2",
        ),
    )
    fine = (
        replace(PERSONALIZED, name="poi_fine_rule_control"),
        *(
            TensorPoiModelPolicy(
                bundles[name], "fine", 0.025,
                deployment_name=f"poi_fine_{name}",
            )
            for name in ("linear", "wide_deep", "dcnv2", "mmoe")
        ),
    )
    mix = (
        fine[2],
        TensorPoiModelPolicy(
            bundles["wide_deep"], "mix", 0.003,
            deployment_name="poi_mix_wide_deep_a003",
        ),
        TensorPoiModelPolicy(
            bundles["wide_deep"], "mix", 0.006,
            deployment_name="poi_mix_wide_deep_a006",
        ),
    )
    end_to_end = (
        coarse[0],
        TensorPoiModelPolicy(
            bundles["linear"], "end_to_end", 0.025,
            fine_strength=0.025,
            deployment_name="poi_e2e_linear_coarse_fine",
        ),
    )
    return {
        "coarse": coarse, "fine": fine, "mix": mix,
        "end_to_end": end_to_end,
    }


COMPARISONS = {
    "coarse": ((0, 1), (1, 2)),
    "fine": ((0, 1), (1, 2), (2, 3), (2, 4)),
    "mix": ((0, 1), (1, 2)),
    "end_to_end": ((0, 1),),
}
def _seed_run(users, steps, seed, device, bundles, selected_stages):
    config = TensorFeedConfig(
        users=users, steps=steps, batch_users=min(users, 200_000),
        seed=seed, device=device, signal_version="kuairand-local-neural-v4",
        trace_users=16, trace_requests_per_user=steps,
    )
    stages = {}
    for stage, policies in _arms(bundles).items():
        if stage not in selected_stages:
            continue
        worlds = {policy.name: run_tensor_feed(config, policy) for policy in policies}
        stages[stage] = []
        for control_index, treatment_index in COMPARISONS[stage]:
            control = policies[control_index]
            treatment = policies[treatment_index]
            online_metrics = combine_tensor_ab(
                worlds[control.name], worlds[treatment.name]
            )
            metrics = combine_tensor_counterfactual_ab(
                worlds[control.name], worlds[treatment.name]
            )
            gates, decision = gate(metrics, stage)
            stages[stage].append({
                "stage": stage,
                "control": control.name,
                "treatment": treatment.name,
                "metrics": metrics,
                "online_disjoint_metrics": online_metrics,
                "gates": gates,
                "decision": decision,
                "candidate_graph": worlds[treatment.name]["candidate_graph"],
                "performance": worlds[treatment.name]["performance"],
            })
    return {"config": asdict(config), "stages": stages}


def run_poi_distribution_launch(
    bundles, users=200_000, steps=24,
    seeds=(20260824, 20260825, 20260826), device="cuda:0",
    stages=("coarse", "fine", "mix"),
):
    seed_reports = [
        _seed_run(users, steps, seed, device, bundles, stages) for seed in seeds
    ]
    arms = _arms(bundles)
    launches = [
        aggregate(stage, index, seed_reports)
        for stage in arms if stage in stages
        for index in range(len(COMPARISONS[stage]))
    ]
    return {
        "schema": "poi-distribution-trained-launch-review-v1",
        "seeds": list(seeds), "users_per_seed": users, "steps": steps,
        "launches": launches,
        "seed_reports": seed_reports,
        "stages": list(stages),
        "evaluation_protocol": {
            "shadow_replay": "same hashed users in common-random worlds",
            "online_ab": "disjoint control and treatment hash cells",
            "stage_primary": PRIMARY_BY_STAGE,
        },
        "evidence_boundary": (
            "Common-random GPU A/B on the hidden neural Local V4 teacher; "
            "simulator evidence only."
        ),
    }
