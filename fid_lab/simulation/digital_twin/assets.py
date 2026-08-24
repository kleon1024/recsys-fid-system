"""Declared asset lineage; execution engines resolve this graph by key."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter


@dataclass(frozen=True)
class AssetSpec:
    key: str
    inputs: tuple[str, ...]
    owner: str


class AssetGraph:
    def __init__(self, assets: tuple[AssetSpec, ...]):
        self._assets = {asset.key: asset for asset in assets}
        if len(self._assets) != len(assets):
            raise ValueError("digital-twin asset keys must be unique")
        unknown = {
            dependency
            for asset in assets
            for dependency in asset.inputs
            if dependency not in self._assets
        }
        if unknown:
            raise ValueError(f"asset graph has unknown inputs: {sorted(unknown)}")
        try:
            self._order = tuple(TopologicalSorter({
                asset.key: set(asset.inputs) for asset in assets
            }).static_order())
        except CycleError as error:
            raise ValueError("digital-twin asset graph contains a cycle") from error

    def order(self) -> tuple[str, ...]:
        return self._order

    def spec(self, key: str) -> AssetSpec:
        return self._assets[key]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "digital-twin-assets-v1",
            "order": list(self._order),
            "assets": {
                key: {
                    "inputs": list(self._assets[key].inputs),
                    "owner": self._assets[key].owner,
                }
                for key in self._order
            },
        }


DIGITAL_TWIN_ASSETS = AssetGraph((
    AssetSpec("world.exogenous", (), "world"),
    AssetSpec("events.pending_delivery", ("world.exogenous",), "events"),
    AssetSpec("world.sessions", ("world.exogenous",), "world"),
    AssetSpec("platform.requests", ("world.sessions",), "platform"),
    AssetSpec(
        "platform.rendered_slates", ("platform.requests",), "platform"
    ),
    AssetSpec(
        "events.observable",
        (
            "events.pending_delivery",
            "world.sessions",
            "platform.rendered_slates",
        ),
        "events",
    ),
    AssetSpec(
        "projection.online_state", ("events.observable",), "platform"
    ),
    AssetSpec("samples.recall", ("events.observable",), "samples"),
    AssetSpec("samples.coarse", ("events.observable",), "samples"),
    AssetSpec("samples.fine", ("events.observable",), "samples"),
    AssetSpec(
        "models.candidate",
        ("samples.recall", "samples.coarse", "samples.fine"),
        "learning",
    ),
    AssetSpec(
        "evaluation.shadow",
        ("models.candidate", "platform.rendered_slates"),
        "experiments",
    ),
    AssetSpec(
        "experiment.mixed_ab",
        ("evaluation.shadow", "projection.online_state"),
        "experiments",
    ),
    AssetSpec(
        "release.decision", ("experiment.mixed_ab",), "release"
    ),
    AssetSpec(
        "world.factual_successor",
        ("events.observable",),
        "world",
    ),
))
