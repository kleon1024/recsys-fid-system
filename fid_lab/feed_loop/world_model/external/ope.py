"""Doubly robust policy ordering on KuaiRand uniform random exposures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .kuairand.kernel import KuaiBehaviorKernel, SlateResponse
from .kuairand.data.randomized import calibration_masks, load_randomized_split
from .kuairand.evaluation.policy import (
    STANDARDIZED_FEED_WEIGHTS,
    calibrate_response,
    policy_distribution,
)
from .kuairand.evaluation.statistics import cluster_interval
from .kuairand.launch.contracts import (
    PolicySpec,
    assert_artifact_compatible,
    stream_sha256,
)
from .kuairand.launch.pipeline import LaunchStage, LaunchState
from .replay import (
    GUARD_TOLERANCES,
)


METRIC_NAMES = (
    "click", "long_view", "like", "comment", "forward", "follow", "hate",
    "stay_norm",
)
POLICY_TEMPERATURE = 0.20
UNIFORM_MIXTURE = 0.50
MIN_ESS_FRACTION = 0.05
def _q_values(response: SlateResponse):
    return torch.cat((response.probabilities, response.stay_norm[:, :, None]), dim=2)


def _catalog_slate(catalog, rows):
    sparse = catalog["sparse"][None].expand(rows, -1, -1)
    dense = catalog["dense"][None].expand(rows, -1, -1)
    return sparse, dense


def _logged_indices(catalog_ids, logged_ids):
    indices = torch.searchsorted(catalog_ids, logged_ids)
    if not torch.equal(catalog_ids[indices], logged_ids):
        raise ValueError("randomized action is outside the declared uniform pool")
    return indices


def _dr_batch(split, indices, catalog, control, treatment, world, rules,
              temperature, uniform_mixture, utility_mode, eligible):
    candidate_sparse, candidate_dense = _catalog_slate(catalog, len(indices))
    arguments = (
        split.sparse[indices], split.dense[indices], candidate_sparse,
        candidate_dense, split.history_items[indices],
        split.history_feedback[indices],
    )
    control_response = control.score_slate(*arguments)
    treatment_response = treatment.score_slate(*arguments)
    world_response = world.score_slate(*arguments)
    calibrated = {
        "control": calibrate_response(control_response, rules["wide_deep"]),
        "treatment": calibrate_response(
            treatment_response, rules["treatment"]
        ),
        "world": calibrate_response(world_response, rules["independent_world"]),
    }
    policies = {
        "control": policy_distribution(
            calibrated["control"],
            PolicySpec(utility_mode, temperature, uniform_mixture),
            eligible,
        ),
        "treatment": policy_distribution(
            calibrated["treatment"],
            PolicySpec(utility_mode, temperature, uniform_mixture),
            eligible,
        ),
    }
    q_values = _q_values(calibrated["world"])
    catalog_ids = catalog["raw_video_ids"]
    logged = _logged_indices(catalog_ids, split.raw_video_ids[indices])
    device = q_values.device
    rows = torch.arange(len(indices), device=device)
    logged = logged.to(device)
    observed = split.labels[indices].to(device)
    logged_q = q_values[rows, logged]
    propensity = split.exposure_propensity[indices].to(device)
    outputs = {}
    for name, policy in policies.items():
        direct = torch.einsum("bn,bnt->bt", policy, q_values)
        weight = policy[rows, logged] / propensity
        estimate = direct + weight[:, None] * (observed - logged_q)
        outputs[name] = (estimate.cpu(), weight.cpu())
    return outputs


def policy_value_gates(metrics: dict, diagnostics: dict) -> dict[str, bool]:
    """One primary outcome plus bounded multi-objective guardrails."""
    return {
        "stay_primary_positive": (
            metrics["stay_norm"]["confidence_interval_95"][0] > 0
        ),
        "long_view_guardrail": (
            metrics["long_view"]["confidence_interval_95"][0]
            >= -GUARD_TOLERANCES["long_view"]
        ),
        "hate_guardrail": (
            metrics["hate"]["confidence_interval_95"][1]
            <= GUARD_TOLERANCES["hate"]
        ),
        "click_guardrail": (
            metrics["click"]["confidence_interval_95"][0]
            >= -GUARD_TOLERANCES["click"]
        ),
        "like_guardrail": (
            metrics["like"]["confidence_interval_95"][0]
            >= -GUARD_TOLERANCES["like"]
        ),
        "importance_support": all(
            value["effective_sample_fraction"] >= MIN_ESS_FRACTION
            for value in diagnostics.values()
        ),
    }


def _report(estimates, weights, users, corpus_size, requested_rows,
            temperature, uniform_mixture, utility_mode, eligible_items,
            minimum_standard_exposures):
    metrics = {}
    delta = estimates["treatment"] - estimates["control"]
    for index, name in enumerate(METRIC_NAMES):
        mean, standard_error, interval = cluster_interval(
            delta[:, index], users
        )
        metrics[name] = {
            "control_dr": float(estimates["control"][:, index].mean()),
            "treatment_dr": float(estimates["treatment"][:, index].mean()),
            "absolute_delta": mean,
            "cluster_standard_error": standard_error,
            "confidence_interval_95": interval,
        }
    diagnostics = {}
    for name, value in weights.items():
        ess = float(np.square(value.sum()) / np.square(value).sum())
        diagnostics[name] = {
            "effective_sample_size": ess,
            "effective_sample_fraction": ess / len(value),
            "maximum_importance_weight": float(value.max()),
            "mean_importance_weight": float(value.mean()),
        }
    gates = policy_value_gates(metrics, diagnostics)
    return {
        "schema": "kuairand-1k-randomized-dr-ope-v1",
        "rows": len(users), "requested_rows": requested_rows,
        "users": int(len(np.unique(users))), "candidate_corpus": corpus_size,
        "eligible_learned_items": eligible_items,
        "minimum_standard_exposures": minimum_standard_exposures,
        "metrics": metrics, "importance_sampling": diagnostics,
        "gates": gates,
        "decision": "randomized_ope_pass" if all(gates.values())
        else "randomized_ope_hold",
        "policy": {
            "temperature": temperature,
            "uniform_mixture": uniform_mixture,
            "utility_mode": utility_mode,
            "standardized_feed_weights": STANDARDIZED_FEED_WEIGHTS,
            "treatment_guard_tolerances": GUARD_TOLERANCES,
            "guard_scope": (
                "Not applied inside OPE because the hard feasible set violates "
                "random-log support; the guard is evaluated by stateful shadow."
            ),
            "utility_semantics": "ranking utility only, not LT",
        },
    }


def run_randomized_ope(dataset_dir: Path, control_artifact: Path,
                       treatment_artifact: Path, world_artifact: Path,
                       benchmark_report: Path, device="cuda:0", rows=8_192,
                       batch_size=8, seed=20260824,
                       temperature=POLICY_TEMPERATURE,
                       uniform_mixture=UNIFORM_MIXTURE,
                       treatment_calibration_report: Path | None = None,
                       treatment_calibration_key="sequence_transformer",
                       world_calibration_report: Path | None = None,
                       utility_mode="raw_probability",
                       minimum_standard_exposures=5):
    policy_spec = PolicySpec(
        utility_mode, temperature, uniform_mixture, minimum_standard_exposures
    )
    dataset_manifest = assert_artifact_compatible(
        dataset_dir, (control_artifact, treatment_artifact, world_artifact)
    )
    split = load_randomized_split(dataset_dir, "random_test")
    calibration, evaluation = calibration_masks(split, seed)
    del calibration
    eligible = np.flatnonzero(evaluation)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(
        eligible, min(rows, len(eligible)), replace=False
    ))
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    eligible = catalog["standard_exposure_count"] >= minimum_standard_exposures
    if not eligible.any():
        raise ValueError("no catalog item meets learned-policy exposure support")
    control = KuaiBehaviorKernel.load(control_artifact, device)
    treatment = KuaiBehaviorKernel.load(treatment_artifact, device)
    world = KuaiBehaviorKernel.load(world_artifact, device)
    base_rules = json.loads(benchmark_report.read_text())["randomized_calibration"]
    rules = {
        "wide_deep": base_rules["wide_deep"],
        "treatment": base_rules[treatment_calibration_key],
        "independent_world": base_rules["independent_world"],
    }
    if treatment_calibration_report is not None:
        adapter = json.loads(treatment_calibration_report.read_text())
        rules["treatment"] = adapter["randomized_calibration"][
            "sequence_randomized_adapter"
        ]
    if world_calibration_report is not None:
        world_adapter = json.loads(world_calibration_report.read_text())
        rules["independent_world"] = world_adapter["randomized_calibration"][
            "sequence_randomized_adapter"
        ]
    estimates = {name: [] for name in ("control", "treatment")}
    weights = {name: [] for name in estimates}
    for start in range(0, len(selected), batch_size):
        indices = torch.from_numpy(selected[start:start + batch_size])
        result = _dr_batch(
            split, indices, catalog, control, treatment, world, rules,
            temperature, uniform_mixture, utility_mode, eligible.to(device),
        )
        for name, (estimate, weight) in result.items():
            estimates[name].append(estimate.numpy())
            weights[name].append(weight.numpy())
    joined_estimates = {
        name: np.concatenate(value) for name, value in estimates.items()
    }
    joined_weights = {
        name: np.concatenate(value) for name, value in weights.items()
    }
    report = _report(
        joined_estimates, joined_weights,
        split.user_ids[selected].numpy(), len(catalog["raw_video_ids"]), rows,
        temperature, uniform_mixture,
        utility_mode,
        int(eligible.sum()), minimum_standard_exposures,
    )
    report["artifacts"] = {
        "dataset_manifest": dataset_manifest,
        "control": stream_sha256(control_artifact),
        "treatment": stream_sha256(treatment_artifact),
        "world": stream_sha256(world_artifact),
    }
    report["policy_contract"] = policy_spec.to_dict()
    launch_state = LaunchState().record(LaunchStage.OFFLINE_CAPACITY, True)
    launch_state = launch_state.record(LaunchStage.RANDOMIZED_CALIBRATION, True)
    launch_state = launch_state.record(
        LaunchStage.RANDOMIZED_OPE, report["decision"] == "randomized_ope_pass"
    )
    report["launch_state"] = {
        "active_authority": launch_state.active_authority,
        "completed": [stage.value for stage in launch_state.completed],
        "decision": launch_state.decision,
    }
    report["evidence_boundary"] = (
        "Doubly robust estimates use uniform random-exposure propensities, "
        "user-disjoint calibration/evaluation, and a softened full-support policy."
    )
    return report
