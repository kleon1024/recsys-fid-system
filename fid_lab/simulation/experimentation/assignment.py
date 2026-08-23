"""Stable mutually-exclusive-within-layer, orthogonal-across-layer assignment."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import blake2b

import numpy as np
import torch

from .contracts import ExperimentLayer, FeedParameters


def _uniform(subject_id: int, salt: str) -> float:
    digest = blake2b(
        f"{salt}:{subject_id}".encode(), digest_size=8, person=b"feed-ab"
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def _salt_uint64(salt: str) -> np.uint64:
    digest = blake2b(salt.encode(), digest_size=8, person=b"feed-ab").digest()
    return np.uint64(int.from_bytes(digest, "big"))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def assign_binary_torch(subject_ids: torch.Tensor, salt: int = 0x1B873593):
    """Avalanche-hashed 50/50 assignment for device-resident experiments."""
    mask = 0x7FFFFFFF
    values = torch.bitwise_xor(subject_ids.long(), salt) & mask
    values = torch.bitwise_xor(values, values >> 16)
    values = (values * 0x045D9F3B) & mask
    values = torch.bitwise_xor(values, values >> 16)
    values = (values * 0x045D9F3B) & mask
    values = torch.bitwise_xor(values, values >> 16)
    return values < 2**30


def assign_layer_numpy(
    subject_ids: np.ndarray, layer: ExperimentLayer
) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    """Vectorized assignment; -1 is layer default and nonnegative values index cells."""
    identifiers = np.asarray(subject_ids, dtype=np.uint64)
    hashed = _splitmix64(identifiers ^ _salt_uint64(layer.salt))
    bucket = hashed.astype(np.float64) / float(2**64)
    cells = tuple(
        (experiment.name, variant.name)
        for experiment in layer.experiments
        for variant in experiment.variants
    )
    allocations = np.asarray(
        [
            variant.allocation
            for experiment in layer.experiments
            for variant in experiment.variants
        ]
    )
    thresholds = np.cumsum(allocations)
    assignment = np.searchsorted(thresholds, bucket, side="right").astype(np.int16)
    assignment[assignment == len(cells)] = -1
    return assignment, cells


def _assignment_for_layer(subject_id: int, layer: ExperimentLayer):
    bucket = _uniform(subject_id, layer.salt)
    cursor = 0.0
    for experiment in layer.experiments:
        for variant in experiment.variants:
            cursor += variant.allocation
            if bucket < cursor:
                return experiment, variant
    return None


def validate_layer_ownership(layers: tuple[ExperimentLayer, ...]) -> None:
    owners = {}
    for layer in layers:
        parameters = {
            parameter
            for experiment in layer.experiments
            for variant in experiment.variants
            for parameter in variant.parameters
        }
        for parameter in parameters:
            if parameter in owners:
                raise ValueError(
                    f"parameter {parameter} is owned by both {owners[parameter]} and {layer.name}"
                )
            owners[parameter] = layer.name


def assign_layers(
    subject_id: int,
    layers: tuple[ExperimentLayer, ...],
    defaults: FeedParameters,
) -> dict[str, object]:
    validate_layer_ownership(layers)
    parameters = defaults
    assignments = {}
    for layer in layers:
        selected = _assignment_for_layer(subject_id, layer)
        if selected is None:
            assignments[layer.name] = {
                "experiment": "layer_default",
                "variant": "control",
            }
            continue
        experiment, variant = selected
        parameters = parameters.overlay(variant.parameters)
        assignments[layer.name] = {
            "experiment": experiment.name,
            "variant": variant.name,
        }
    return {
        "subject_id": subject_id,
        "assignments": assignments,
        "parameters": asdict(parameters),
    }
