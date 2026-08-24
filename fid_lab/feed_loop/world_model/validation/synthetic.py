"""Independent paired checks on held-out v4 structural world families."""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import (
    ACCEPTANCE_THRESHOLDS,
    BINARY_ACTIONS,
    STRUCTURAL_INTERVENTION_NAMES,
)
from ..data import WorldModelSplit
from ..ensemble import WorldModelEnsemble


def _unavailable_interventions(reason):
    return {
        "world": "held_out_v4_structural_families",
        "available": False,
        "reason": reason,
        "interventions": [],
        "sign_accuracy": None,
        "normalized_mae": None,
        "pass": False,
        "production_evidence": False,
    }


def _treated_batch(split, batch, index, rows, device):
    return {
        **batch,
        "selected_features": split.structural_intervention_features[
            :rows, index
        ].to(device),
        "slate_features": split.structural_intervention_slates[
            :rows, index
        ].to(device),
        "sequence": split.structural_intervention_sequences[
            :rows, index
        ].to(device),
    }


def intervention_recovery(ensemble: WorldModelEnsemble, split: WorldModelSplit,
                          device: torch.device, limit=50_000):
    required = (
        split.structural_intervention_features,
        split.structural_intervention_slates,
        split.structural_intervention_sequences,
        split.structural_intervention_effects,
        split.structural_family_ids,
    )
    if any(value is None for value in required):
        return _unavailable_interventions(
            "held-out structural potential outcomes are absent",
        )
    count = min(len(split), limit)
    index = torch.arange(count)
    batch = split.batch(index, device)
    baseline = ensemble.predict(batch)["utility_mean"]
    observed = []
    predicted = []
    details = []
    for offset, name in enumerate(STRUCTURAL_INTERVENTION_NAMES):
        treated = _treated_batch(split, batch, offset, count, device)
        observed_effect = float(
            split.structural_intervention_effects[:count, offset].mean()
        )
        predicted_effect = float(
            (ensemble.predict(treated)["utility_mean"] - baseline).mean()
        )
        observed.append(observed_effect)
        predicted.append(predicted_effect)
        details.append({
            "name": name, "observed_effect": observed_effect,
            "predicted_effect": predicted_effect,
        })
    observed_array = np.asarray(observed)
    predicted_array = np.asarray(predicted)
    sign_accuracy = float((np.sign(observed_array) == np.sign(predicted_array)).mean())
    normalized_mae = float(
        np.abs(observed_array - predicted_array).mean()
        / max(np.abs(observed_array).mean(), 1e-8)
    )
    return {
        "world": "held_out_v4_structural_families",
        "available": True,
        "family_ids": torch.unique(
            split.structural_family_ids[:count]
        ).tolist(),
        "interventions": details,
        "sign_accuracy": sign_accuracy,
        "normalized_mae": normalized_mae,
        "pass": (
            sign_accuracy >= ACCEPTANCE_THRESHOLDS["intervention_sign_accuracy"]
            and normalized_mae <= ACCEPTANCE_THRESHOLDS["intervention_normalized_mae"]
        ),
        "production_evidence": False,
    }


def _candidate_model_scores(ensemble, split, device, limit, request_batch=128):
    count = min(len(split), limit)
    candidate_count = split.slate_features.shape[1]
    output = []
    for start in range(0, count, request_batch):
        stop = min(start + request_batch, count)
        rows = torch.arange(start, stop)
        slate = split.slate_features[rows].to(device)
        sequence = split.sequence[rows].to(device)
        lifecycle = split.lifecycle[rows].to(device)
        region = split.region[rows].to(device)
        repeated = {
            "selected_features": slate.reshape(-1, slate.shape[-1]),
            "slate_features": slate[:, None].expand(
                -1, candidate_count, -1, -1
            ).reshape(-1, candidate_count, slate.shape[-1]),
            "sequence": sequence[:, None].expand(
                -1, candidate_count, -1, -1
            ).reshape(-1, sequence.shape[1], sequence.shape[2]),
            "lifecycle": lifecycle[:, None].expand(-1, candidate_count).reshape(-1),
            "region": region[:, None].expand(-1, candidate_count).reshape(-1),
            "labels": torch.zeros(
                (stop - start) * candidate_count, 16, device=device
            ),
        }
        predicted = ensemble.predict(repeated)
        probabilities = predicted["probability_mean"]
        names = {action.name: index for index, action in enumerate(BINARY_ACTIONS)}
        score = (
            predicted["stay_mean"]
            + 0.7 * probabilities[:, names["long_view"]]
            + 0.2 * probabilities[:, names["like"]]
            - 0.2 * probabilities[:, names["negative_feedback"]]
        )
        output.append(score.reshape(stop - start, candidate_count).cpu())
    return torch.cat(output)


def policy_order_agreement(ensemble: WorldModelEnsemble, split: WorldModelSplit,
                           device: torch.device, limit=10_000):
    if split.candidate_utility_source != "synthetic_oracle" or not torch.isfinite(
        split.candidate_audit_utility
    ).all():
        return {
            "world": split.candidate_utility_source,
            "available": False,
            "reason": (
                "candidate-level counterfactual utility is not observed in "
                "the external randomized bridge"
            ),
            "policies": [],
            "kendall_tau": None,
            "pass": False,
            "production_evidence": False,
        }
    count = min(len(split), limit)
    features = split.slate_features[:count]
    model_scores = _candidate_model_scores(ensemble, split, device, count)
    choices = {
        "popular": features[:, :, 3].argmax(dim=1),
        "quality": features[:, :, 1].argmax(dim=1),
        "served_rule": split.candidate_fine_scores[:count].argmax(dim=1),
        "world_model": model_scores.argmax(dim=1),
    }
    rows = torch.arange(count)
    observed = []
    predicted = []
    details = []
    for name, choice in choices.items():
        observed_value = float(split.candidate_audit_utility[:count][rows, choice].mean())
        predicted_value = float(model_scores[rows, choice].mean())
        observed.append(observed_value)
        predicted.append(predicted_value)
        details.append({
            "name": name, "observed_value": observed_value,
            "predicted_value": predicted_value,
        })
    tau = _kendall_tau(np.asarray(observed), np.asarray(predicted))
    return {
        "world": "frozen_v3_candidate_audit_utility",
        "policies": details,
        "kendall_tau": tau,
        "pass": tau >= ACCEPTANCE_THRESHOLDS["policy_kendall_tau"],
        "production_evidence": False,
    }


def _kendall_tau(observed, predicted):
    products = []
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            products.append(
                np.sign(observed[left] - observed[right])
                * np.sign(predicted[left] - predicted[right])
            )
    return float(np.mean(products)) if products else 0.0


def synthetic_causal_validation(ensemble, split, device_name):
    device = torch.device(device_name)
    return {
        "intervention_recovery": intervention_recovery(
            ensemble, split, device
        ),
        "policy_order": policy_order_agreement(ensemble, split, device),
    }
