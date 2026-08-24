"""Multi-domain training support, abstention, and exploitation probes."""

from __future__ import annotations

import torch

from ..data import WorldModelSplit
from ..feature_contract import SUPPORT_BOUNDED_FEATURES, V4_REQUIRED_FEATURES


SUPPORT_PROFILE_SCHEMA = "neural-scm-support-profile-v4"
MAX_SUPPORT_ROWS = 50_000
MAX_REQUEST_OOD_RATE = 0.03
MIN_ADVERSARIAL_REJECTION_RATE = 0.99


def _row_indices(rows, limit=MAX_SUPPORT_ROWS):
    count = min(rows, limit)
    return torch.arange(count) * rows // count


def _sources(value):
    return value if isinstance(value, tuple) else (value,)


def _parameters(component, device):
    return (
        torch.tensor(
            component["distance_feature_indices"], device=device,
            dtype=torch.long,
        ),
        torch.tensor(component["center"], device=device),
        torch.tensor(component["scale"], device=device),
        float(component["request_distance_threshold"]),
    )


def _request_distance(features, indices, center, scale):
    selected = features[:, :, indices]
    candidate_distance = torch.sqrt(
        ((selected - center) / scale).square().mean(dim=2)
    )
    return candidate_distance.max(dim=1).values


def _fit_component(train_features, validation_features, name):
    indices = torch.tensor(sorted(V4_REQUIRED_FEATURES))
    distance_indices = torch.tensor(sorted(
        V4_REQUIRED_FEATURES - SUPPORT_BOUNDED_FEATURES
    ))
    bounded_indices = torch.tensor(sorted(
        V4_REQUIRED_FEATURES & SUPPORT_BOUNDED_FEATURES
    ))
    train_rows = _row_indices(len(train_features))
    train_features = train_features[train_rows].float()
    train_values = train_features[:, :, distance_indices].reshape(
        -1, len(distance_indices),
    )
    center = torch.quantile(train_values, 0.50, dim=0)
    lower = torch.quantile(train_values, 0.005, dim=0)
    upper = torch.quantile(train_values, 0.995, dim=0)
    scale = ((upper - lower) / 5.15).clamp_min(0.05)
    validation_rows = _row_indices(len(validation_features))
    distance = _request_distance(
        validation_features[validation_rows].float(),
        distance_indices, center, scale,
    )
    threshold = torch.quantile(distance, 0.995).clamp_min(1.0)
    return {
        "name": name,
        "feature_indices": indices.tolist(),
        "distance_feature_indices": distance_indices.tolist(),
        "bounded_feature_indices": bounded_indices.tolist(),
        "bounded_lower": train_features[:, :, bounded_indices].amin(
            dim=(0, 1)
        ).tolist(),
        "bounded_upper": train_features[:, :, bounded_indices].amax(
            dim=(0, 1)
        ).tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "lower_005": lower.tolist(),
        "upper_995": upper.tolist(),
        "request_distance_threshold": float(threshold),
        "fit_rows": len(train_rows),
        "calibration_rows": len(validation_rows),
        "calibration_request_ood_rate": float(
            (distance > threshold).float().mean()
        ),
    }


def fit_support_profile(train: WorldModelSplit, validation: WorldModelSplit):
    train_sources = _sources(train)
    validation_sources = _sources(validation)
    if len(train_sources) != len(validation_sources):
        raise ValueError("support train and validation sources must align")
    components = []
    for index, (left, right) in enumerate(zip(
        train_sources, validation_sources, strict=True,
    )):
        family_ids = left.structural_family_ids
        families = (
            torch.unique(family_ids).tolist()
            if family_ids is not None else []
        )
        if len(families) > 1:
            for family_id in families:
                components.append(_fit_component(
                    left.slate_features[family_ids == family_id],
                    right.slate_features,
                    f"source_{index}_family_{family_id}",
                ))
        else:
            components.append(_fit_component(
                left.slate_features, right.slate_features, f"source_{index}",
            ))
    return {
        "schema": SUPPORT_PROFILE_SCHEMA,
        "combination": "union_of_source_components",
        "feature_indices": components[0]["feature_indices"],
        "components": components,
        "fit_rows": sum(row["fit_rows"] for row in components),
        "fit_rows_by_source": [row["fit_rows"] for row in components],
        "calibration_rows": sum(row["calibration_rows"] for row in components),
        "calibration_rows_by_source": [
            row["calibration_rows"] for row in components
        ],
    }


def _normalized_component_distances(features, profile):
    distances = []
    for component in profile["components"]:
        indices, center, scale, threshold = _parameters(
            component, features.device,
        )
        normalized = (
            _request_distance(features.float(), indices, center, scale)
            / threshold
        )
        bounded_indices = torch.tensor(
            component["bounded_feature_indices"], device=features.device,
            dtype=torch.long,
        )
        lower = torch.tensor(component["bounded_lower"], device=features.device)
        upper = torch.tensor(component["bounded_upper"], device=features.device)
        bounded = features[:, :, bounded_indices]
        valid_bounds = (
            (bounded >= lower - 1e-6) & (bounded <= upper + 1e-6)
        ).all(dim=(1, 2))
        distances.append(normalized.masked_fill(~valid_bounds, torch.inf))
    return torch.stack(distances, dim=1)


def request_support_mask(features: torch.Tensor, profile: dict):
    if profile.get("schema") != SUPPORT_PROFILE_SCHEMA:
        raise ValueError("world-model artifact lacks a valid support profile")
    return _normalized_component_distances(features, profile).amin(dim=1) <= 1.0


def support_report(split: WorldModelSplit, profile: dict, limit=50_000):
    rows = _row_indices(len(split), limit)
    features = split.slate_features[rows]
    normalized = _normalized_component_distances(features, profile)
    minimum = normalized.amin(dim=1)
    ood_rate = float((minimum > 1.0).float().mean())
    return {
        "rows": len(rows),
        "request_ood_rate": ood_rate,
        "request_distance": {
            "metric": "minimum_component_normalized_distance",
            "p50": float(torch.quantile(minimum, 0.50)),
            "p99": float(torch.quantile(minimum, 0.99)),
            "maximum": float(minimum.max()),
            "threshold": 1.0,
        },
        "component_request_ood_rates": [
            float((normalized[:, index] > 1.0).float().mean())
            for index in range(normalized.shape[1])
        ],
        "pass": ood_rate <= MAX_REQUEST_OOD_RATE,
    }


def anti_exploitation_report(split: WorldModelSplit, profile: dict, limit=2_048):
    rows = _row_indices(len(split), limit)
    source = split.slate_features[rows].float()
    rejection_rates = []
    for component in profile["components"]:
        features = source.clone()
        indices, center, scale, _ = _parameters(component, features.device)
        direction = torch.ones_like(center)
        if 7 in component["distance_feature_indices"]:
            direction[component["distance_feature_indices"].index(7)] = -1.0
        features[:, :, indices] = center + 8.0 * scale * direction
        bounded_indices = torch.tensor(
            component["bounded_feature_indices"], device=features.device,
            dtype=torch.long,
        )
        upper = torch.tensor(component["bounded_upper"], device=features.device)
        features[:, :, bounded_indices] = upper + 1.0
        rejection_rates.append(float(
            (~request_support_mask(features, profile)).float().mean()
        ))
    rejection_rate = min(rejection_rates)
    return {
        "rows": len(rows),
        "attack": "each_component_all_required_features_eight_robust_scales",
        "component_rejection_rates": rejection_rates,
        "rejection_rate": rejection_rate,
        "minimum_rejection_rate": MIN_ADVERSARIAL_REJECTION_RATE,
        "pass": rejection_rate >= MIN_ADVERSARIAL_REJECTION_RATE,
    }
