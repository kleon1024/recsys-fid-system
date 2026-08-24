"""Randomized Feed policy ordering bound to the core NeuralSCM artifact."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from ....contracts import ACCEPTANCE_THRESHOLDS
from ....data import WorldModelSplit, load_world_split
from ....training import load_world_ensemble
from ....validation.policy_evidence import kendall_tau
from ..data.core_bridge import catalog_action_features
from ..data.randomized import calibration_masks, load_randomized_split, subset_split
from ..launch.contracts import load_dataset_manifest, stream_sha256


UNIFORM_MIXTURE = 0.50
MIN_ESS_FRACTION = 0.05
MAX_IMPORTANCE_WEIGHT = 20.0
MAX_POLICY_VALUE_NORMALIZED_MAE = 0.25
MIN_MONTE_CARLO_ACTIONS = 64
MIN_IDENTIFIED_PAIRS = 2
POLICY_NAMES = ("uniform", "popular", "tail", "long_duration")


def _standardized(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp_min(1e-6)


def _policy_probabilities(catalog) -> torch.Tensor:
    popularity = torch.log1p(catalog["standard_exposure_count"].float())
    duration = catalog["dense"][:, 0].float()
    scores = torch.stack((
        torch.zeros_like(popularity),
        _standardized(popularity),
        -_standardized(popularity),
        _standardized(duration),
    ))
    learned = torch.softmax(scores / 1.25, dim=1)
    uniform = torch.full_like(learned, 1.0 / learned.shape[1])
    probabilities = UNIFORM_MIXTURE * uniform + (1.0 - UNIFORM_MIXTURE) * learned
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(len(probabilities)),
        rtol=1e-5, atol=1e-6,
    )
    return probabilities


def _predicted_utility(prediction) -> torch.Tensor:
    return prediction["utility_mean"]


def _observed_utility(split: WorldModelSplit) -> torch.Tensor:
    log_181 = math.log(181.0)
    return (
        0.55 * torch.log1p(split.labels[:, 2]) / log_181
        + 0.30 * split.labels[:, 5]
        + 0.10 * split.labels[:, 7]
        - 0.05 * split.labels[:, 8]
    )


def _prediction_batch(split, rows, selected, device):
    requests, actions = selected.shape[:2]
    slate = split.slate_features[rows].to(device)
    slate = slate[:, None].expand(-1, actions, -1, -1).clone()
    slate[:, :, 0] = selected.to(device)
    sequence = split.sequence[rows].to(device)
    return {
        "selected_features": selected.to(device).reshape(-1, selected.shape[-1]),
        "slate_features": slate.reshape(-1, *slate.shape[2:]),
        "sequence": sequence[:, None].expand(
            -1, actions, -1, -1
        ).reshape(-1, *sequence.shape[1:]),
        "lifecycle": split.lifecycle[rows].to(device)[:, None].expand(
            -1, actions
        ).reshape(-1),
        "region": split.region[rows].to(device)[:, None].expand(
            -1, actions
        ).reshape(-1),
        "labels": torch.zeros(requests * actions, 1, device=device),
    }


def _score_actions(ensemble, bridge, source, catalog, action_indices, device,
                   request_batch):
    scores = []
    for start in range(0, len(bridge), request_batch):
        stop = min(start + request_batch, len(bridge))
        rows = torch.arange(start, stop)
        actions = action_indices[start:stop]
        features = catalog_action_features(
            subset_split(source, rows), catalog, actions,
        ).float()
        prediction = ensemble.predict(
            _prediction_batch(bridge, rows, features, device)
        )
        scores.append(_predicted_utility(prediction).reshape(
            stop - start, actions.shape[1]
        ).cpu())
    return torch.cat(scores)


def _logged_catalog_indices(catalog, raw_video_ids):
    catalog_ids = catalog["raw_video_ids"].long()
    positions = torch.searchsorted(catalog_ids, raw_video_ids.long())
    safe = positions.clamp_max(len(catalog_ids) - 1)
    if not torch.equal(catalog_ids[safe], raw_video_ids.long()):
        raise ValueError("randomized logged action is outside the catalog")
    return positions


def _bootstrap_values(weights, rewards, user_ids, seed, repetitions=400):
    users, inverse = np.unique(user_ids, return_inverse=True)
    policy_count = weights.shape[0]
    numerator = np.zeros((len(users), policy_count))
    denominator = np.zeros((len(users), policy_count))
    for policy in range(policy_count):
        np.add.at(numerator[:, policy], inverse, weights[policy] * rewards)
        np.add.at(denominator[:, policy], inverse, weights[policy])
    rng = np.random.default_rng(seed)
    output = np.empty((repetitions, policy_count))
    for index in range(repetitions):
        sampled = rng.integers(0, len(users), len(users))
        output[index] = numerator[sampled].sum(axis=0) / denominator[
            sampled
        ].sum(axis=0).clip(min=1e-12)
    return output


def _artifact_closure(dataset_dir, bridge_dir, artifact_dir):
    dataset = load_dataset_manifest(dataset_dir)
    bridge_path = bridge_dir / "manifest.json"
    bridge = json.loads(bridge_path.read_text())
    artifact_path = artifact_dir / "manifest.json"
    artifact = json.loads(artifact_path.read_text())
    bridge_sha = sha256(bridge_path.read_bytes()).hexdigest()
    artifact_sha = sha256(artifact_path.read_bytes()).hexdigest()
    if (
        artifact.get("dataset_manifest_sha256") != bridge_sha
        and bridge_sha not in artifact.get("dataset_source_manifest_sha256s", [])
    ):
        raise ValueError("world artifact is not bound to the bridge manifest")
    if not bridge.get("feature_contract_sha256"):
        raise ValueError("bridge manifest lacks a feature contract")
    if artifact.get("feature_contract_sha256") != bridge["feature_contract_sha256"]:
        raise ValueError("world artifact and bridge feature contracts disagree")
    if bridge.get("source_catalog_sha256") != dataset.get("catalog_sha256"):
        raise ValueError("bridge and randomized catalog are incompatible")
    weights_sha = stream_sha256(artifact_dir / "world_model.pt")
    if weights_sha != artifact.get("weights_sha256"):
        raise ValueError("world-model weights do not match the artifact manifest")
    return {
        "world_model_manifest_sha256": artifact_sha,
        "world_model_weights_sha256": weights_sha,
        "bridge_manifest_sha256": bridge_sha,
        "feature_contract_sha256": bridge["feature_contract_sha256"],
        "source_dataset_manifest_sha256": sha256(
            (dataset_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_catalog_sha256": dataset["catalog_sha256"],
    }


def _selected_evaluation_rows(dataset_dir, bridge_dir, rows, seed):
    randomized = load_randomized_split(dataset_dir, "random_test")
    _, evaluation = calibration_masks(randomized, seed)
    source = subset_split(
        randomized, torch.from_numpy(evaluation).nonzero().flatten()
    )
    total = len(source)
    count = min(rows, total)
    selection = torch.arange(count) * total // count
    selected_source = subset_split(source, selection)
    bridge = load_world_split(bridge_dir, "test", count, "uniform")
    if not torch.equal(selected_source.user_ids.long(), bridge.user_ids):
        raise ValueError("randomized source and bridge row order disagree")
    expected_topic = selected_source.sparse[:, 3].float() / 8_192.0
    torch.testing.assert_close(
        bridge.selected_features[:, 17], expected_topic, rtol=1e-3, atol=1e-3,
    )
    return selected_source, bridge


def _pairwise_diagnostics(bootstrap):
    rows = []
    identified = 0
    for left in range(len(POLICY_NAMES)):
        for right in range(left + 1, len(POLICY_NAMES)):
            delta = bootstrap[:, left] - bootstrap[:, right]
            interval = np.quantile(delta, (0.025, 0.975))
            is_identified = bool(interval[0] > 0 or interval[1] < 0)
            identified += is_identified
            rows.append({
                "left": POLICY_NAMES[left], "right": POLICY_NAMES[right],
                "delta_ci95": interval.tolist(), "identified": is_identified,
            })
    return rows, identified


def run_neural_policy_evidence(dataset_dir: Path, bridge_dir: Path,
                               artifact_dir: Path, device_name="cuda:0",
                               rows=8_192, action_samples=64,
                               request_batch=16, seed=20260824):
    if action_samples < MIN_MONTE_CARLO_ACTIONS:
        raise ValueError("randomized policy evidence requires at least 64 actions")
    closure = _artifact_closure(dataset_dir, bridge_dir, artifact_dir)
    source, bridge = _selected_evaluation_rows(
        dataset_dir, bridge_dir, rows, seed
    )
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    policies = _policy_probabilities(catalog)
    logged = _logged_catalog_indices(catalog, source.raw_video_ids)
    generator = torch.Generator().manual_seed(seed + 701)
    sampled = torch.randint(
        len(catalog["raw_video_ids"]),
        (len(bridge), action_samples), generator=generator,
    )
    ensemble = load_world_ensemble(artifact_dir, device_name)
    device = torch.device(device_name)
    sampled_q = _score_actions(
        ensemble, bridge, source, catalog, sampled, device, request_batch,
    )
    logged_q = _score_actions(
        ensemble, bridge, source, catalog, logged[:, None], device,
        request_batch,
    )[:, 0]
    reward = _observed_utility(bridge)
    catalog_size = policies.shape[1]
    logged_weights = policies[:, logged] * catalog_size
    sampled_weights = policies[:, sampled] * catalog_size
    direct = (sampled_weights * sampled_q[None]).mean(dim=2)
    snips = (
        (logged_weights * reward[None]).sum(dim=1)
        / logged_weights.sum(dim=1)
    )
    predicted = direct.mean(dim=1)
    dr = (
        direct + logged_weights * (reward - logged_q)[None]
    ).mean(dim=1)
    weights_np = logged_weights.numpy()
    bootstrap = _bootstrap_values(
        weights_np, reward.numpy(), bridge.user_ids.numpy(), seed,
    )
    intervals = np.quantile(bootstrap, (0.025, 0.975), axis=0).T
    pairwise, identified = _pairwise_diagnostics(bootstrap)
    tau = kendall_tau(snips.numpy(), predicted.numpy())
    value_normalized_mae = float(
        (snips - predicted).abs().mean() / snips.abs().mean().clamp_min(1e-8)
    )
    policy_rows = []
    ess_fractions = []
    maximum_weights = []
    for index, name in enumerate(POLICY_NAMES):
        weights = logged_weights[index]
        ess = float(weights.sum().square() / weights.square().sum())
        ess_fraction = ess / len(weights)
        maximum = float(weights.max())
        ess_fractions.append(ess_fraction)
        maximum_weights.append(maximum)
        policy_rows.append({
            "name": name,
            "observed_value": float(snips[index]),
            "observed_ci95": intervals[index].tolist(),
            "predicted_value": float(predicted[index]),
            "doubly_robust_value": float(dr[index]),
            "effective_sample_size": ess,
            "effective_sample_fraction": ess_fraction,
            "maximum_importance_weight": maximum,
            "mean_importance_weight": float(weights.mean()),
        })
    gates = {
        "artifact_closure": True,
        "full_support": float(policies.min()) > 0.0,
        "monte_carlo_actions": action_samples >= MIN_MONTE_CARLO_ACTIONS,
        "importance_support": min(ess_fractions) >= MIN_ESS_FRACTION,
        "bounded_importance_weights": max(maximum_weights) <= MAX_IMPORTANCE_WEIGHT,
        "identified_policy_pairs": identified >= MIN_IDENTIFIED_PAIRS,
        "policy_order": tau >= ACCEPTANCE_THRESHOLDS["policy_kendall_tau"],
        "policy_value_calibration": (
            value_normalized_mae <= MAX_POLICY_VALUE_NORMALIZED_MAE
        ),
    }
    return {
        "schema": "neural-scm-kuairand-policy-evidence-v1",
        **closure,
        "rows": len(bridge),
        "users": int(bridge.user_ids.unique().numel()),
        "catalog_items": catalog_size,
        "monte_carlo_actions_per_request": action_samples,
        "reward_contract": (
            "0.55*normalized_stay + 0.30*long_view + 0.10*like "
            "- 0.05*negative_feedback; Feed policy utility, not LT"
        ),
        "policy_contract": {
            "names": POLICY_NAMES,
            "uniform_mixture": UNIFORM_MIXTURE,
            "logging_propensity": 1.0 / catalog_size,
            "context": "fixed seven-item reference slate with action in slot zero",
        },
        "policies": policy_rows,
        "pairwise_observed_differences": pairwise,
        "identified_policy_pairs": identified,
        "policy_kendall_tau": tau,
        "policy_value_normalized_mae": value_normalized_mae,
        "gates": gates,
        "policy_order_pass": all(gates.values()),
        "decision": "pass" if all(gates.values()) else "hold",
        "evidence_boundary": (
            "User-disjoint KuaiRand random exposures identify Feed action-policy "
            "value within the declared 7,388-item pool. They do not identify "
            "retention, creator supply, Local, Ads, Commerce, or LT."
        ),
    }
