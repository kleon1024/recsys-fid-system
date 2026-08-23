"""Independent paired-world checks against the frozen V3 synthetic oracle."""

from __future__ import annotations

import math

import numpy as np
import torch

from ...scale.calibration.nonlinear import nonlinear_stay_adjustment
from ...scale.graph.random import normal, uniform
from ..contracts import ACCEPTANCE_THRESHOLDS, BINARY_ACTIONS
from ..data import WorldModelSplit
from ..ensemble import StructuralNoise, WorldModelEnsemble


INTERVENTIONS = {
    "interest_affinity_up": (0, 0.08),
    "item_quality_up": (1, 0.08),
    "realtime_fatigue_up": (7, 0.08),
}


def _treated_batch(batch, exposed_index, feature_index, delta):
    selected = batch["selected_features"].clone()
    selected[:, feature_index] = (selected[:, feature_index] + delta).clamp(0.0, 1.0)
    slate = batch["slate_features"].clone()
    rows = torch.arange(len(selected), device=selected.device)
    slate[rows, exposed_index, feature_index] = selected[:, feature_index]
    return {**batch, "selected_features": selected, "slate_features": slate}


def _v3_stay(features, user_ids, steps, seed):
    affinity = features[:, 0]
    quality = features[:, 1]
    fatigue = features[:, 7]
    satisfaction = features[:, 6]
    duration = torch.expm1(
        features[:, 12] * math.log(181.0)
    ).clamp(1.0, 180.0)
    played = uniform(user_ids, steps, 31, seed) < torch.sigmoid(
        1.7 + 0.4 * affinity - 0.6 * fatigue
    )
    log_mean = (
        0.8 + 1.7 * affinity + 0.55 * quality
        + 0.20 * satisfaction - fatigue + nonlinear_stay_adjustment(features)
    )
    return torch.minimum(
        duration,
        torch.exp(log_mean + 2.8 * normal(user_ids, steps, 32, seed)),
    ) * played


def intervention_recovery(ensemble: WorldModelEnsemble, split: WorldModelSplit,
                          device: torch.device, limit=50_000):
    count = min(len(split), limit)
    index = torch.arange(count)
    batch = split.batch(index, device)
    exposed = split.exposed_index[:count].to(device)
    user_ids = split.user_ids[:count].to(device)
    steps = split.request_steps[:count].to(device)
    observed = []
    predicted = []
    details = []
    for offset, (name, (feature_index, delta)) in enumerate(INTERVENTIONS.items()):
        treated = _treated_batch(batch, exposed, feature_index, delta)
        noise = StructuralNoise.generate(
            count, ensemble.config, device, ensemble.config.seed + 50_000 + offset
        )
        control_stay = torch.stack([
            sample["stay_seconds"] for sample in ensemble.sample_members(batch, noise)
        ]).mean(dim=0)
        treated_stay = torch.stack([
            sample["stay_seconds"] for sample in ensemble.sample_members(treated, noise)
        ]).mean(dim=0)
        observed_effect = float((
            _v3_stay(treated["selected_features"], user_ids, steps, ensemble.config.seed)
            - _v3_stay(batch["selected_features"], user_ids, steps, ensemble.config.seed)
        ).mean())
        predicted_effect = float((treated_stay - control_stay).mean())
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
        "world": "frozen_v3_synthetic_oracle",
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
