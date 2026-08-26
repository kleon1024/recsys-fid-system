"""Orthogonal layer assignment compiled into one factual serving policy."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from ...randomness.counter import uniform
from ..contracts import PlatformRequestBatch
from ..engine import ExperimentAssignment
from ..platform.ranking import CascadePolicy


@dataclass(frozen=True)
class PolicyLayer:
    name: str
    salt: int
    changes: dict[str, object]
    control_fraction: float
    treatment_fraction: float
    eligible_surfaces: tuple[int, ...] | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("policy layer requires a name")
        if (
            self.control_fraction <= 0.0
            or self.treatment_fraction <= 0.0
            or self.control_fraction + self.treatment_fraction > 1.0
        ):
            raise ValueError("layer fractions must be positive and sum to <= 1")
        valid = {field.name for field in fields(CascadePolicy)}
        unknown = set(self.changes) - valid
        if unknown:
            raise ValueError(f"unknown policy parameters: {sorted(unknown)}")
        if not self.changes:
            raise ValueError("policy layer must change at least one parameter")


@dataclass(frozen=True)
class LayeredExperimentPlan:
    base_policy: CascadePolicy
    layers: tuple[PolicyLayer, ...]
    assignment_unit: str = "user"

    def __post_init__(self):
        if not self.layers:
            raise ValueError("layered experiment plan requires at least one layer")
        if self.assignment_unit not in {"user", "request", "creator"}:
            raise ValueError("assignment unit must be user, request or creator")
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("policy layer names must be unique")
        owners: dict[str, str] = {}
        for layer in self.layers:
            for parameter in layer.changes:
                if parameter in owners:
                    raise ValueError(
                        f"parameter {parameter} is owned by both "
                        f"{owners[parameter]} and {layer.name}"
                    )
                owners[parameter] = layer.name

    def assign(self, requests: PlatformRequestBatch) -> ExperimentAssignment:
        entity = requests.request_id
        if self.assignment_unit == "user":
            entity = requests.user_id
        elif self.assignment_unit == "creator":
            entity = torch.where(
                requests.user_creator_id >= 0,
                requests.user_creator_id,
                requests.user_id,
            )
        cells = []
        probabilities = []
        for layer in self.layers:
            eligible = torch.ones_like(entity, dtype=torch.bool)
            if layer.eligible_surfaces is not None:
                eligible.zero_()
                for surface in layer.eligible_surfaces:
                    eligible |= requests.surface == surface
            draw = uniform(entity, 0, 1_901 + layer.salt, layer.salt)
            cell = torch.full_like(entity, -1)
            control = eligible & (draw < layer.control_fraction)
            treatment = eligible & (draw >= layer.control_fraction) & (
                draw < layer.control_fraction + layer.treatment_fraction
            )
            cell[control] = 0
            cell[treatment] = 1
            probability = torch.ones_like(draw)
            probability[eligible] = 1.0 - (
                layer.control_fraction + layer.treatment_fraction
            )
            probability[control] = layer.control_fraction
            probability[treatment] = layer.treatment_fraction
            cells.append(cell)
            probabilities.append(probability)
        layer_cells = torch.stack(cells, dim=1)
        layer_probability = torch.stack(probabilities, dim=1)
        signature = torch.zeros_like(entity)
        radix = 1
        for layer_cell in cells:
            signature += (layer_cell + 1) * radix
            radix *= 3
        policies = {0: self.base_policy}
        for value in torch.unique(signature).tolist():
            changes = {}
            remainder = value
            for layer in self.layers:
                layer_cell = remainder % 3 - 1
                remainder //= 3
                if layer_cell == 1:
                    changes.update(layer.changes)
            policies[value] = replace(
                self.base_policy,
                name=f"{self.base_policy.name}-cell-{value}",
                **changes,
            )
        active_cells = tuple(
            int(value) for value in torch.unique(signature).tolist() if value != 0
        )
        return ExperimentAssignment(
            cell_by_request=signature,
            probability_by_request=layer_probability.prod(dim=1),
            policies=policies,
            analysis_cells=active_cells,
            default_cell=0,
            layer_names=tuple(layer.name for layer in self.layers),
            layer_cell_by_request=layer_cells,
            layer_probability_by_request=layer_probability,
        )
