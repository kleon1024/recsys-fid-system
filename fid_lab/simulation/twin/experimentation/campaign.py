"""Sequential stage-isolated Launch Reviews with accepted-control continuity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..contracts import BASELINE_POLICY, TwinConfig, TwinPolicy
from ..kernel import DigitalTwinKernel
from .experiment import TwinExperiment, evaluate_from_preperiod


class LaunchStage(str, Enum):
    RETRIEVAL = "retrieval"
    COARSE = "coarse"
    FINE = "fine"
    MIX = "mix"


@dataclass(frozen=True)
class PolicyMutation:
    launch_id: str
    stage: LaunchStage
    hypothesis: str
    changes: dict[str, object]


DEFAULT_CAMPAIGN = (
    PolicyMutation(
        "L-TWIN-RECALL-001", LaunchStage.RETRIEVAL,
        "Personalized ANN candidates improve shared-platform outcomes.",
        {"enabled_routes": (True, False, True, True, False, True)},
    ),
    PolicyMutation(
        "L-TWIN-RECALL-002", LaunchStage.RETRIEVAL,
        "Realtime graph and long-tail routes add nonredundant coverage.",
        {"enabled_routes": (True, True, True, True, True, True)},
    ),
    PolicyMutation(
        "L-TWIN-COARSE-001", LaunchStage.COARSE,
        "A stronger quality term protects valuable candidates at coarse rank.",
        {"quality_weight": 0.45, "coarse_keep": 28},
    ),
    PolicyMutation(
        "L-TWIN-COARSE-002", LaunchStage.COARSE,
        "Geo and commercial intent improve cross-surface coarse ranking.",
        {"geo_weight": 0.28, "commerce_weight": 0.16},
    ),
    PolicyMutation(
        "L-TWIN-FINE-001", LaunchStage.FINE,
        "Realtime short interest improves request-level fine ranking.",
        {"realtime_weight": 0.30},
    ),
    PolicyMutation(
        "L-TWIN-FINE-002", LaunchStage.FINE,
        "Exposure-ledger fatigue reduces repeated item families safely.",
        {
            "author_fatigue_penalty": 0.06,
            "cluster_fatigue_penalty": 0.09,
            "topic_fatigue_penalty": 0.05,
        },
    ),
    PolicyMutation(
        "L-TWIN-MIX-001", LaunchStage.MIX,
        "Calibrated Local and commerce values add value without Feed harm.",
        {"local_value_weight": 0.06, "product_value_weight": 0.06},
    ),
    PolicyMutation(
        "L-TWIN-MIX-002", LaunchStage.MIX,
        "Constrained Ads and Live value improve monetization safely.",
        {"ad_value_weight": 0.055, "live_value_weight": 0.045},
    ),
)


def launch_decision(experiment: TwinExperiment):
    metrics = experiment.report["cuped_ab"]
    primary = metrics["synthetic_lt_measurement"]
    stay = metrics["stay_seconds"]
    negative = metrics["negative"]
    requests = metrics["requests"]
    trace = experiment.report["trace"]["gates"]
    trace_pass = all(
        value for arm in trace.values() for value in arm.values()
    )
    gates = {
        "request_trace_closes": trace_pass,
        "primary_direction_nonnegative": primary["difference"] >= 0.0,
        "primary_noninferior": primary["confidence_interval"][0]
        >= -0.005 * max(abs(primary["control_mean"]), 1e-6),
        "stay_noninferior": stay["confidence_interval"][0]
        >= -0.01 * max(abs(stay["control_mean"]), 1e-6),
        "negative_guardrail": negative["confidence_interval"][1] <= 0.002,
        "request_guardrail": requests["confidence_interval"][0]
        >= -0.01 * max(abs(requests["control_mean"]), 1e-6),
    }
    if all(gates.values()):
        decision = "pass"
    elif trace_pass and primary["confidence_interval"][1] > 0:
        decision = "hold"
    else:
        decision = "reject"
    return decision, gates


def run_launch_campaign(
    config: TwinConfig,
    proposals: tuple[PolicyMutation, ...] = DEFAULT_CAMPAIGN,
    initial_policy: TwinPolicy = BASELINE_POLICY,
    salt: int = 0x1B873593,
) -> dict[str, object]:
    kernel = DigitalTwinKernel(config)
    world = kernel.initialize()
    active = initial_policy
    launches = []
    for index, proposal in enumerate(proposals):
        preperiod = kernel.preperiod_from(world, active)
        candidate = replace(
            active, name=f"{active.name}+{proposal.launch_id}",
            **proposal.changes,
        )
        experiment = evaluate_from_preperiod(
            config, kernel, preperiod, active, candidate,
            salt + index * 104_729,
        )
        decision, gates = launch_decision(experiment)
        before = active.name
        if decision == "pass":
            active = candidate
        world = experiment.mixed.snapshot
        launches.append({
            "launch_id": proposal.launch_id,
            "stage": proposal.stage.value,
            "hypothesis": proposal.hypothesis,
            "isolated_changes": proposal.changes,
            "control": before,
            "candidate": candidate.name,
            "decision": decision,
            "gates": gates,
            "active_after": active.name,
            "primary": experiment.report["cuped_ab"][
                "synthetic_lt_measurement"
            ],
            "stay": experiment.report["cuped_ab"]["stay_seconds"],
            "negative": experiment.report["cuped_ab"]["negative"],
            "trace": experiment.report["trace"],
            "performance": experiment.report["performance"],
        })
    return {
        "schema": "multi-surface-continuous-launch-campaign-v1",
        "config": config.manifest(),
        "control_rule": "every proposal compares with the last accepted control",
        "launches": launches,
        "active_policy": active.manifest(),
        "stage_counts": {
            stage.value: sum(row["stage"] == stage.value for row in launches)
            for stage in LaunchStage
        },
        "evidence_boundary": (
            "Synthetic sequential Launch Reviews; not production lift."
        ),
    }
