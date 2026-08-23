"""Paired-world Local Service ranking iterations on the main Feed."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ...value import unified_lt_launch_decision
from ..ab import experiment_metrics
from ..contracts import SimulationConfig
from ..environment import build_catalog
from ..policies import HeuristicPolicy, LocalIntentPolicy
from ..population import run_population


def _decision(metrics: dict[str, dict[str, float]]) -> str:
    return unified_lt_launch_decision(metrics["lt_value"])


def _policies():
    feed = HeuristicPolicy()
    return (
        feed,
        LocalIntentPolicy(feed, "local_static_v1", 0.08),
        LocalIntentPolicy(feed, "local_post_search_v2", 0.08, search_weight=0.55),
        LocalIntentPolicy(
            feed,
            "local_search_retarget_v3",
            0.08,
            search_weight=0.55,
            retarget_weight=0.45,
        ),
        LocalIntentPolicy(
            feed,
            "local_intent_quality_rank_v4",
            0.08,
            search_weight=0.55,
            retarget_weight=0.45,
            intent_quality_weight=0.10,
            embedding_correction_weight=1.0,
        ),
        LocalIntentPolicy(
            feed,
            "local_value_expansion_v5",
            0.14,
            search_weight=0.55,
            retarget_weight=0.45,
            intent_quality_weight=0.10,
            embedding_correction_weight=1.0,
        ),
    )


def run_local_service_launch_suite(
    users: int = 4_000,
    items: int = 8_000,
    seed: int = 20260823,
) -> dict[str, object]:
    config = SimulationConfig(users=users, items=items, joiner_users=0, seed=seed)
    catalog = build_catalog(config)
    user_ids = np.arange(users, dtype=np.int64) + 90_000_000
    policies = _policies()
    trajectories = {
        policy.name: run_population(config, catalog, policy, user_ids)
        for policy in policies
    }
    launches = []
    for launch_index, (control, treatment) in enumerate(zip(policies, policies[1:])):
        assigned = np.random.default_rng(seed + 500 + launch_index).random(users) < 0.5
        metrics, _ = experiment_metrics(
            trajectories[control.name], trajectories[treatment.name], assigned
        )
        launches.append(
            {
                "launch_id": f"L-LOCAL-RANK-{launch_index + 1:03d}",
                "control": control.name,
                "treatment": treatment.name,
                "assignment": {
                    "control_users": int((~assigned).sum()),
                    "treatment_users": int(assigned.sum()),
                },
                "metrics": metrics,
                "decision": _decision(metrics),
            }
        )
    return {
        "suite": "local-service-main-feed-lt-v1",
        "config": asdict(config),
        "lt_contract": "lt-platform-metrics-v1",
        "launches": launches,
        "evidence_boundary": (
            "Synthetic paired worlds validate mechanics and estimator behavior; "
            "they are not production effect estimates."
        ),
    }
