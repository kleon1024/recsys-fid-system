"""Stateful common-random shadow over catalog-aware randomized contexts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.randomized import calibration_masks, load_randomized_split
from ..kernel import KuaiBehaviorKernel
from ..launch.contracts import assert_artifact_compatible, stream_sha256
from .policy import calibrate_response, policy_utility
from .statistics import cluster_interval
from ...replay import (
    GUARD_TOLERANCES,
    selected_metrics,
    treatment_guard_mask,
)


METRICS = ("click", "long_view", "like", "hate", "stay_norm")


def _rules(
    benchmark_report: Path,
    adapter_report: Path,
    world_adapter_report: Path | None,
    world_calibration_key: str,
) -> dict:
    benchmark = json.loads(benchmark_report.read_text())["randomized_calibration"]
    adapter = json.loads(adapter_report.read_text())["randomized_calibration"]
    rules = {
        "control": benchmark["wide_deep"],
        "treatment": adapter["sequence_randomized_adapter"],
        "world": benchmark[world_calibration_key],
    }
    if world_adapter_report is not None:
        world_adapter = json.loads(world_adapter_report.read_text())[
            "randomized_calibration"
        ]
        rules["world"] = world_adapter["sequence_randomized_adapter"]
    return rules


def _gates(metrics: dict) -> dict[str, bool]:
    return {
        "stay_primary_positive": metrics["stay_norm"]["confidence_interval_95"][0] > 0,
        "long_view_guardrail": metrics["long_view"]["confidence_interval_95"][0]
        >= -GUARD_TOLERANCES["long_view"],
        "hate_guardrail": metrics["hate"]["confidence_interval_95"][1]
        <= GUARD_TOLERANCES["hate"],
        "click_guardrail": metrics["click"]["confidence_interval_95"][0]
        >= -GUARD_TOLERANCES["click"],
        "like_guardrail": metrics["like"]["confidence_interval_95"][0]
        >= -GUARD_TOLERANCES["like"],
    }


def _user_means(values: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.stack(
        [
            np.bincount(inverse, weights=values[:, index]) / counts
            for index in range(values.shape[1])
        ],
        axis=1,
    )
    return unique, means


def _simulated_user_ab(
    totals: dict,
    users: np.ndarray,
    steps: int,
    seed: int,
    simulated_users: int,
) -> dict:
    unique, control = _user_means(totals["control"].numpy() / steps, users)
    treatment_users, treatment = _user_means(
        totals["treatment"].numpy() / steps, users
    )
    if not np.array_equal(unique, treatment_users):
        raise ValueError("control and treatment user support differs")
    source_users = len(unique)
    if simulated_users < source_users:
        raise ValueError("simulated A/B cannot discard randomized user support")
    rng = np.random.default_rng(seed + 77)
    if simulated_users > source_users:
        profiles = rng.integers(source_users, size=simulated_users)
        control = control[profiles]
        treatment = treatment[profiles]
    assigned = rng.random(simulated_users) < 0.5
    observed = np.where(assigned[:, None], treatment, control)
    metrics = {}
    for position, name in enumerate(METRICS):
        control_values = observed[~assigned, position]
        treatment_values = observed[assigned, position]
        effect = float(treatment_values.mean() - control_values.mean())
        standard_error = float(
            np.sqrt(
                treatment_values.var(ddof=1) / len(treatment_values)
                + control_values.var(ddof=1) / len(control_values)
            )
        )
        interval = [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ]
        paired_truth = float((treatment[:, position] - control[:, position]).mean())
        metrics[name] = {
            "control_mean": float(control_values.mean()),
            "treatment_mean": float(treatment_values.mean()),
            "absolute_effect": effect,
            "standard_error": standard_error,
            "confidence_interval_95": interval,
            "paired_world_truth": paired_truth,
            "truth_inside_interval": interval[0] <= paired_truth <= interval[1],
        }
    gates = _gates(metrics)
    gates["randomization_recovers_truth"] = all(
        metric["truth_inside_interval"] for metric in metrics.values()
    )
    return {
        "experiment_unit": "user_id",
        "users": simulated_users,
        "source_randomized_profiles": source_users,
        "population_source": (
            "bootstrap over user-level randomized-evaluation profiles; "
            "synthetic power evidence only"
        ),
        "control_users": int((~assigned).sum()),
        "treatment_users": int(assigned.sum()),
        "metrics": metrics,
        "gates": gates,
        "decision": "simulated_ab_pass" if all(gates.values())
        else "simulated_ab_hold",
        "value_semantics": "behavior outcomes only; not unified LT",
    }


def _initialize_shadow(dataset_dir, artifacts, report_paths, settings):
    manifest = assert_artifact_compatible(dataset_dir, artifacts)
    split = load_randomized_split(dataset_dir, "random_test")
    _, evaluation = calibration_masks(split, settings["seed"])
    eligible_rows = np.flatnonzero(evaluation)
    selected_rows = np.sort(np.random.default_rng(settings["seed"]).choice(
        eligible_rows, min(settings["rows"], len(eligible_rows)), replace=False
    ))
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    eligible_items = torch.nonzero(
        catalog["standard_exposure_count"]
        >= settings["minimum_standard_exposures"]
    ).flatten()
    if len(eligible_items) < settings["candidates"]:
        raise ValueError("candidate support is smaller than the requested slate")
    kernels = {
        name: KuaiBehaviorKernel.load(path, settings["device"])
        for name, path in zip(
            ("control", "treatment", "world"), artifacts, strict=True
        )
    }
    rules = _rules(*report_paths)
    return manifest, split, selected_rows, catalog, eligible_items, kernels, rules


def _run_arm(arm, kernels, rules, arguments, candidate_sparse, history, seed):
    response = calibrate_response(kernels[arm].score_slate(*arguments), rules[arm])
    utility = policy_utility(response, "raw_probability")
    if arm == "treatment":
        base = calibrate_response(
            kernels["control"].score_slate(*arguments), rules["control"]
        )
        base_choice = policy_utility(base, "raw_probability").argmax(dim=1)
        utility = utility.masked_fill(
            ~treatment_guard_mask(base, response, base_choice), -torch.inf
        )
    choice = utility.argmax(dim=1)
    world = calibrate_response(
        kernels["world"].score_slate(*arguments), rules["world"]
    )
    actions, _ = kernels["world"].sample_selected(world, choice, seed)
    chosen_items = candidate_sparse[
        torch.arange(len(choice)), choice.cpu(), 1
    ]
    updated = kernels["world"].advance_history(
        *history, chosen_items, actions.cpu()
    )
    return selected_metrics(world, choice).cpu(), updated


def _run_shadow_steps(split, selected_rows, catalog, eligible_items, kernels,
                      rules, settings):
    totals = {
        arm: torch.zeros(len(selected_rows), len(METRICS))
        for arm in ("control", "treatment")
    }
    histories = {
        arm: (
            split.history_items[selected_rows].clone(),
            split.history_feedback[selected_rows].clone(),
        )
        for arm in totals
    }
    for step in range(settings["steps"]):
        for start in range(0, len(selected_rows), settings["batch_size"]):
            stop = min(start + settings["batch_size"], len(selected_rows))
            index = torch.as_tensor(selected_rows[start:stop])
            seed = settings["seed"] + step * 1_000_003 + start
            generator = torch.Generator().manual_seed(seed)
            sampled = eligible_items[torch.randint(
                len(eligible_items), (len(index), settings["candidates"]),
                generator=generator,
            )]
            candidate_sparse = catalog["sparse"][sampled]
            candidate_dense = catalog["dense"][sampled]
            for arm in totals:
                history = tuple(value[start:stop] for value in histories[arm])
                arguments = (
                    split.sparse[index], split.dense[index], candidate_sparse,
                    candidate_dense, *history,
                )
                metrics, updated = _run_arm(
                    arm, kernels, rules, arguments, candidate_sparse, history, seed
                )
                totals[arm][start:stop] += metrics
                histories[arm][0][start:stop], histories[arm][1][start:stop] = updated
    return totals


def _shadow_metrics(totals, users, steps):
    delta = (totals["treatment"] - totals["control"]) / steps
    output = {}
    for position, name in enumerate(METRICS):
        mean, standard_error, interval = cluster_interval(
            delta[:, position].numpy(), users
        )
        output[name] = {
            "absolute_delta": mean,
            "cluster_standard_error": standard_error,
            "confidence_interval_95": interval,
            "control_mean": float((totals["control"][:, position] / steps).mean()),
        }
    return output


@torch.inference_mode()
def run_randomized_shadow(
    dataset_dir: Path,
    control_artifact: Path,
    treatment_artifact: Path,
    world_artifact: Path,
    benchmark_report: Path,
    adapter_report: Path,
    world_adapter_report: Path | None = None,
    world_calibration_key: str = "independent_world",
    *,
    device: str = "cuda:0",
    rows: int = 20_000,
    candidates: int = 50,
    steps: int = 8,
    batch_size: int = 128,
    seed: int = 20260824,
    minimum_standard_exposures: int = 5,
    simulated_ab_users: int = 100_000,
) -> dict:
    settings = {
        "device": device, "rows": rows, "candidates": candidates,
        "steps": steps, "batch_size": batch_size, "seed": seed,
        "minimum_standard_exposures": minimum_standard_exposures,
    }
    artifacts = (control_artifact, treatment_artifact, world_artifact)
    report_paths = (
        benchmark_report, adapter_report, world_adapter_report,
        world_calibration_key,
    )
    manifest, split, selected_rows, catalog, eligible_items, kernels, rules = (
        _initialize_shadow(dataset_dir, artifacts, report_paths, settings)
    )
    totals = _run_shadow_steps(
        split, selected_rows, catalog, eligible_items, kernels, rules, settings
    )
    users = split.user_ids[selected_rows].numpy()
    metrics = _shadow_metrics(totals, users, steps)
    gates = _gates(metrics)
    simulated_ab = _simulated_user_ab(
        totals, users, steps, seed, simulated_ab_users
    )
    return {
        "schema": "kuairand-randomized-stateful-shadow-v1",
        "rows": len(selected_rows),
        "users": int(len(np.unique(users))),
        "steps": steps,
        "candidates": candidates,
        "eligible_items": len(eligible_items),
        "metrics": metrics,
        "gates": gates,
        "decision": "stateful_shadow_pass" if all(gates.values())
        else "stateful_shadow_hold",
        "simulated_ab": simulated_ab,
        "artifacts": {
            "dataset_manifest": manifest,
            "control": stream_sha256(control_artifact),
            "treatment": stream_sha256(treatment_artifact),
            "world": stream_sha256(world_artifact),
        },
        "evidence_boundary": (
            "Common-random stateful replay in an independently trained world model; "
            "ranking utility is not LT and the result is not a live A/B test."
        ),
    }
