"""Distribution, rollout, intervention, and anti-exploitation gates."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..contracts import ACCEPTANCE_THRESHOLDS, BINARY_ACTIONS
from ..data import WorldModelSplit
from ..ensemble import StructuralNoise, WorldModelEnsemble
from .boundary import boundary_invariance_report
from .policy_evidence import verify_policy_evidence
from .support import anti_exploitation_report, support_report
from .synthetic import synthetic_causal_validation


def _ece(labels, probabilities, bins=20):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if mask.any():
            total += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(total)


def _safe_corr(values):
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = np.divide(
        centered, scale, out=np.zeros_like(centered), where=scale > 1e-8
    )
    return standardized.T @ standardized / max(len(values) - 1, 1)


def _masked_correlation_mae(observed, generated, masks):
    errors = []
    for left in range(observed.shape[1]):
        for right in range(left + 1, observed.shape[1]):
            common = masks[:, left] & masks[:, right]
            if common.sum() < 3:
                continue
            actual = observed[common][:, (left, right)]
            simulated = generated[common][:, :, (left, right)].reshape(-1, 2)
            if actual[:, 0].std() < 1e-8 or actual[:, 1].std() < 1e-8:
                continue
            errors.append(abs(
                _safe_corr(actual)[0, 1] - _safe_corr(simulated)[0, 1]
            ))
    return (float(np.mean(errors)) if errors else math.inf, len(errors))


def _lag1(values):
    output = []
    for column in range(values.shape[2]):
        left = values[:, :-1, column].reshape(-1)
        right = values[:, 1:, column].reshape(-1)
        if left.std() < 1e-8 or right.std() < 1e-8:
            output.append(0.0)
        else:
            output.append(float(np.corrcoef(left, right)[0, 1]))
    return np.asarray(output)


def _collect_predictions(ensemble, split, device, limit):
    count = min(len(split), limit)
    probabilities = []
    deviations = []
    generated_actions = []
    generated_stay = []
    utilities = []
    for start in range(0, count, ensemble.config.batch_size):
        index = torch.arange(start, min(start + ensemble.config.batch_size, count))
        batch = split.batch(index, device)
        predicted = ensemble.predict(batch)
        probabilities.append(predicted["probability_mean"].cpu())
        deviations.append(predicted["probability_std"].cpu())
        utilities.append(predicted["utility_mean"].cpu())
        noise = StructuralNoise.generate(
            len(index), ensemble.config, device, ensemble.config.seed + start
        )
        sampled_members = ensemble.sample_members(batch, noise)
        generated_actions.append(torch.stack([
            torch.stack([
                sampled["actions"][action.name].float()
                for action in BINARY_ACTIONS
            ], dim=1)
            for sampled in sampled_members
        ], dim=1).cpu())
        generated_stay.append(torch.stack([
            sampled["stay_seconds"] for sampled in sampled_members
        ], dim=1).cpu())
    return (
        torch.cat(probabilities).numpy(), torch.cat(deviations).numpy(),
        torch.cat(generated_actions).numpy(), torch.cat(generated_stay).numpy(),
        torch.cat(utilities).numpy(),
    )


def _distribution_report(ensemble, split, device, limit):
    count = min(len(split), limit)
    probabilities, deviations, generated, generated_stay, utility = _collect_predictions(
        ensemble, split, device, count
    )
    labels = split.labels[:count].numpy()
    label_masks = split.label_masks[:count].numpy() > 0.5
    binary_labels = np.stack(
        [labels[:, action.label_index] for action in BINARY_ACTIONS], axis=1
    )
    binary_masks = np.stack(
        [label_masks[:, action.label_index] for action in BINARY_ACTIONS], axis=1
    )
    ece = {
        action.name: (
            _ece(binary_labels[binary_masks[:, index], index],
                 probabilities[binary_masks[:, index], index])
            if binary_masks[:, index].any() else None
        )
        for index, action in enumerate(BINARY_ACTIONS)
    }
    supported_ece = [value for value in ece.values() if value is not None]
    correlation_mae, correlation_pairs = _masked_correlation_mae(
        binary_labels, generated, binary_masks,
    )
    stay_observable = label_masks[:, 2]
    observed_stay = labels[stay_observable, 2]
    simulated_stay = generated_stay[stay_observable].reshape(-1)
    if not len(observed_stay):
        raise ValueError("distribution evaluation requires observable stay labels")
    observed_quantiles = np.quantile(observed_stay, (0.5, 0.9))
    simulated_quantiles = np.quantile(simulated_stay, (0.5, 0.9))
    relative = np.abs(simulated_quantiles - observed_quantiles) / np.maximum(
        observed_quantiles, 1.0
    )
    observable_deviations = deviations[binary_masks]
    observed_utility = (
        0.55 * np.log1p(labels[:, 2]) / math.log(181.0)
        + 0.30 * labels[:, 5]
        + 0.10 * labels[:, 7]
        - 0.05 * labels[:, 8]
    )
    utility_correlation = (
        float(np.corrcoef(observed_utility, utility)[0, 1])
        if observed_utility.std() > 1e-8 and utility.std() > 1e-8 else 0.0
    )
    return {
        "rows": count,
        "binary_ece": ece,
        "binary_label_coverage": {
            action.name: float(binary_masks[:, index].mean())
            for index, action in enumerate(BINARY_ACTIONS)
        },
        "stay_label_coverage": float(stay_observable.mean()),
        "mean_binary_ece": (
            float(np.mean(supported_ece)) if supported_ece else math.inf
        ),
        "joint_correlation_mae": correlation_mae,
        "joint_correlation_pairs": correlation_pairs,
        "stay_quantiles": {
            "observed": {"p50": float(observed_quantiles[0]), "p90": float(observed_quantiles[1])},
            "simulated": {"p50": float(simulated_quantiles[0]), "p90": float(simulated_quantiles[1])},
            "relative_error": {"p50": float(relative[0]), "p90": float(relative[1])},
        },
        "ensemble_probability_std": {
            "mean": float(observable_deviations.mean()),
            "p99": float(np.quantile(observable_deviations, 0.99)),
            "maximum": float(observable_deviations.max()),
        },
        "utility_head": {
            "mae": float(np.abs(observed_utility - utility).mean()),
            "correlation": utility_correlation,
            "predicted_std": float(utility.std()),
            "observed_std": float(observed_utility.std()),
        },
    }


def _trajectory_rows(split, steps, limit):
    users = split.user_ids.numpy()
    order = np.argsort(users, kind="stable")
    ordered_users = users[order]
    possible = np.arange(max(len(order) - steps + 1, 0))
    starts = possible[
        ordered_users[possible] == ordered_users[possible + steps - 1]
    ]
    if not len(starts):
        raise ValueError("sequence evaluation has no complete user trajectories")
    if len(starts) > limit:
        starts = starts[np.arange(limit) * len(starts) // limit]
    return torch.from_numpy(order[starts[:, None] + np.arange(steps)[None]])


def _observed_events(split, trajectories):
    labels = split.labels[trajectories]
    features = split.selected_features[trajectories]
    log_181 = math.log(181.0)
    return torch.stack((
        features[:, :, 17],
        torch.log1p(labels[:, :, 2]) / log_181,
        labels[:, :, 5],
        labels[:, :, 6],
        labels[:, :, 7],
        labels[:, :, 8],
        labels[:, :, 9],
        labels[:, :, 12],
    ), dim=2)


def _free_running_events(ensemble, split, trajectories, device, event_mask):
    sequence = split.sequence[trajectories[:, 0]].to(device)
    generated = []
    for step in range(trajectories.shape[1]):
        batch = split.batch(trajectories[:, step], device)
        batch["sequence"] = sequence
        noise = StructuralNoise.generate(
            len(sequence), ensemble.config, device,
            ensemble.config.seed + 90_000 + step,
        )
        samples = ensemble.sample_members(batch, noise)
        event = torch.stack([
            sample["event"] for sample in samples
        ]).float().mean(dim=0)
        event *= event_mask[None]
        generated.append(event.cpu())
        sequence = torch.roll(sequence, shifts=-1, dims=1)
        sequence[:, -1] = event
    return torch.stack(generated, dim=1)


def _sequence_report(ensemble, split, device, limit, steps=8):
    trajectories = _trajectory_rows(split, steps, limit)
    observed = _observed_events(split, trajectories).numpy()
    history_observable = np.abs(split.sequence.numpy()).sum(axis=(0, 1)) > 0
    event_mask = torch.from_numpy(history_observable).to(device)
    simulated = _free_running_events(
        ensemble, split, trajectories, device, event_mask,
    ).numpy()
    evaluated = np.flatnonzero(history_observable[2:8]) + 2
    if not len(evaluated):
        raise ValueError("sequence evaluation has no observed response channels")
    observed_lag = _lag1(observed[:, :, evaluated])
    simulated_lag = _lag1(simulated[:, :, evaluated])
    return {
        "rows": len(trajectories),
        "steps": steps,
        "candidate_context": "next_factual_request_without_teacher_forced_response",
        "evaluated_event_channels": evaluated.tolist(),
        "observed_lag1": observed_lag.tolist(),
        "simulated_lag1": simulated_lag.tolist(),
        "sequence_lag1_mae": float(np.abs(observed_lag - simulated_lag).mean()),
    }


def evaluate_world_model(ensemble: WorldModelEnsemble, split: WorldModelSplit,
                         device_name: str, manifest_sha256: str,
                         policy_evidence: Path | None = None,
                         structural_split: WorldModelSplit | None = None,
                         support_profile: dict | None = None,
                         distribution_rows: int = 100_000,
                         rollout_rows: int = 10_000):
    device = torch.device(device_name)
    distribution = _distribution_report(
        ensemble, split, device, distribution_rows
    )
    sequence = _sequence_report(ensemble, split, device, rollout_rows)
    external_policy = verify_policy_evidence(policy_evidence, manifest_sha256)
    synthetic = synthetic_causal_validation(
        ensemble,
        split if structural_split is None else structural_split,
        device_name,
    )
    boundary = boundary_invariance_report(ensemble, split, device_name)
    if support_profile is None:
        support = {
            "available": False,
            "external": {"pass": False},
            "structural": {"pass": False},
        }
        exploitation = {"available": False, "pass": False}
    else:
        support = {
            "available": True,
            "external": support_report(split, support_profile),
            "structural": (
                support_report(structural_split, support_profile)
                if structural_split is not None else {"pass": False}
            ),
        }
        exploitation = {
            "available": True,
            **anti_exploitation_report(split, support_profile),
        }
    threshold = ACCEPTANCE_THRESHOLDS
    gates = {
        "distribution": (
            distribution["mean_binary_ece"] <= threshold["mean_binary_ece"]
            and distribution["joint_correlation_mae"] <= threshold["joint_correlation_mae"]
            and distribution["stay_quantiles"]["relative_error"]["p50"]
            <= threshold["stay_median_relative_error"]
            and distribution["stay_quantiles"]["relative_error"]["p90"]
            <= threshold["stay_p90_relative_error"]
        ),
        "free_running_sequence": sequence["sequence_lag1_mae"] <= threshold["sequence_lag1_mae"],
        "uncertainty": distribution["ensemble_probability_std"]["p99"] <= threshold["maximum_ensemble_probability_std"],
        "structural_intervention_recovery": synthetic["intervention_recovery"]["pass"],
        "external_policy_order": external_policy["policy_order_pass"],
        "boundary_invariance": boundary["pass"],
        "support_distance": (
            support["external"]["pass"]
            and support["structural"]["pass"]
        ),
        "anti_exploitation": exploitation["pass"],
    }
    return {
        "schema": "neural-scm-world-model-evaluation-v1",
        "distribution": distribution,
        "sequence": sequence,
        "external_policy": external_policy,
        "structural_robustness": synthetic,
        "boundary_invariance": boundary,
        "support_distance": support,
        "anti_exploitation": exploitation,
        "thresholds": threshold,
        "gates": gates,
        "promotion_eligible": all(gates.values()),
        "decision": "promote_world_model_authority" if all(gates.values()) else "hold_research_challenger",
    }
