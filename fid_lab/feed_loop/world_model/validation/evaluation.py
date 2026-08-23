"""Distribution, rollout, intervention, and anti-exploitation gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..contracts import ACCEPTANCE_THRESHOLDS, BINARY_ACTIONS
from ..data import WorldModelSplit
from ..ensemble import StructuralNoise, WorldModelEnsemble
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
    for start in range(0, count, ensemble.config.batch_size):
        index = torch.arange(start, min(start + ensemble.config.batch_size, count))
        batch = split.batch(index, device)
        predicted = ensemble.predict(batch)
        probabilities.append(predicted["probability_mean"].cpu())
        deviations.append(predicted["probability_std"].cpu())
        noise = StructuralNoise.generate(
            len(index), ensemble.config, device, ensemble.config.seed + start
        )
        sampled_members = ensemble.sample_members(batch, noise)
        generated_actions.append(torch.cat([
            torch.stack([
                sampled["actions"][action.name].float()
                for action in BINARY_ACTIONS
            ], dim=1)
            for sampled in sampled_members
        ]).cpu())
        generated_stay.append(torch.cat([
            sampled["stay_seconds"] for sampled in sampled_members
        ]).cpu())
    return (
        torch.cat(probabilities).numpy(), torch.cat(deviations).numpy(),
        torch.cat(generated_actions).numpy(), torch.cat(generated_stay).numpy(),
    )


def _distribution_report(ensemble, split, device, limit):
    count = min(len(split), limit)
    probabilities, deviations, generated, generated_stay = _collect_predictions(
        ensemble, split, device, count
    )
    labels = split.labels[:count].numpy()
    binary_labels = np.stack(
        [labels[:, action.label_index] for action in BINARY_ACTIONS], axis=1
    )
    repeated_binary_labels = np.tile(
        binary_labels, (ensemble.config.ensemble_members, 1)
    )
    ece = {
        action.name: _ece(binary_labels[:, index], probabilities[:, index])
        for index, action in enumerate(BINARY_ACTIONS)
    }
    correlation_mae = float(np.abs(
        _safe_corr(repeated_binary_labels) - _safe_corr(generated)
    ).mean())
    observed_stay = labels[:, 2]
    observed_quantiles = np.quantile(observed_stay, (0.5, 0.9))
    simulated_quantiles = np.quantile(generated_stay, (0.5, 0.9))
    relative = np.abs(simulated_quantiles - observed_quantiles) / np.maximum(
        observed_quantiles, 1.0
    )
    return {
        "rows": count,
        "binary_ece": ece,
        "mean_binary_ece": float(np.mean(tuple(ece.values()))),
        "joint_correlation_mae": correlation_mae,
        "stay_quantiles": {
            "observed": {"p50": float(observed_quantiles[0]), "p90": float(observed_quantiles[1])},
            "simulated": {"p50": float(simulated_quantiles[0]), "p90": float(simulated_quantiles[1])},
            "relative_error": {"p50": float(relative[0]), "p90": float(relative[1])},
        },
        "ensemble_probability_std": {
            "mean": float(deviations.mean()),
            "p99": float(np.quantile(deviations, 0.99)),
            "maximum": float(deviations.max()),
        },
    }


def _sequence_report(ensemble, split, device, limit, steps=8):
    count = min(len(split), limit)
    index = torch.arange(count)
    batch = split.batch(index, device)
    simulated = ensemble.rollout(
        batch, steps, ensemble.config.seed + 90_000
    ).cpu().numpy()
    observed = split.sequence[:count].numpy()
    observed_lag = _lag1(observed[:, :, 2:8])
    simulated_lag = _lag1(simulated[:, :, 2:8])
    return {
        "rows": count,
        "steps": steps,
        "observed_lag1": observed_lag.tolist(),
        "simulated_lag1": simulated_lag.tolist(),
        "sequence_lag1_mae": float(np.abs(observed_lag - simulated_lag).mean()),
    }


def _kendall_tau(observed, predicted):
    concordant = 0
    discordant = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            product = (observed[left] - observed[right]) * (
                predicted[left] - predicted[right]
            )
            concordant += product > 0
            discordant += product < 0
    pairs = concordant + discordant
    return float((concordant - discordant) / pairs) if pairs else 0.0


def _causal_report(path: Path | None, manifest_sha256: str):
    unavailable = {
        "available": False,
        "reason": "held-out randomized interventions and frozen policy outcomes not supplied",
        "intervention_recovery_pass": False,
        "policy_order_pass": False,
    }
    if path is None or not path.exists():
        return unavailable
    evidence = json.loads(path.read_text())
    if evidence.get("world_model_manifest_sha256") != manifest_sha256:
        return {**unavailable, "reason": "causal evidence is not bound to this artifact"}
    interventions = evidence.get("interventions", [])
    policies = evidence.get("policies", [])
    observed = np.asarray([row["observed_effect"] for row in interventions])
    predicted = np.asarray([row["predicted_effect"] for row in interventions])
    sign_accuracy = float((np.sign(observed) == np.sign(predicted)).mean()) if len(observed) else 0.0
    normalized_mae = float(np.abs(observed - predicted).mean() / max(np.abs(observed).mean(), 1e-8)) if len(observed) else math.inf
    observed_policy = np.asarray([row["observed_value"] for row in policies])
    predicted_policy = np.asarray([row["predicted_value"] for row in policies])
    tau = _kendall_tau(observed_policy, predicted_policy)
    return {
        "available": True,
        "interventions": len(interventions),
        "sign_accuracy": sign_accuracy,
        "normalized_mae": normalized_mae,
        "policies": len(policies),
        "policy_kendall_tau": tau,
        "intervention_recovery_pass": (
            len(interventions) >= 3
            and sign_accuracy >= ACCEPTANCE_THRESHOLDS["intervention_sign_accuracy"]
            and normalized_mae <= ACCEPTANCE_THRESHOLDS["intervention_normalized_mae"]
        ),
        "policy_order_pass": (
            len(policies) >= 3
            and tau >= ACCEPTANCE_THRESHOLDS["policy_kendall_tau"]
        ),
    }


def evaluate_world_model(ensemble: WorldModelEnsemble, split: WorldModelSplit,
                         device_name: str, manifest_sha256: str,
                         causal_evidence: Path | None = None,
                         distribution_rows: int = 100_000,
                         rollout_rows: int = 10_000):
    device = torch.device(device_name)
    distribution = _distribution_report(
        ensemble, split, device, distribution_rows
    )
    sequence = _sequence_report(ensemble, split, device, rollout_rows)
    causal = _causal_report(causal_evidence, manifest_sha256)
    synthetic = synthetic_causal_validation(ensemble, split, device_name)
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
        "synthetic_intervention_recovery": synthetic["intervention_recovery"]["pass"],
        "synthetic_policy_order": synthetic["policy_order"]["pass"],
        "intervention_recovery": causal["intervention_recovery_pass"],
        "policy_order": causal["policy_order_pass"],
    }
    return {
        "schema": "neural-scm-world-model-evaluation-v1",
        "distribution": distribution,
        "sequence": sequence,
        "causal": causal,
        "synthetic_causal": synthetic,
        "thresholds": threshold,
        "gates": gates,
        "promotion_eligible": all(gates.values()),
        "decision": "promote_world_model_authority" if all(gates.values()) else "hold_research_challenger",
    }
