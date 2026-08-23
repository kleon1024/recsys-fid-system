"""Immutable dataset, artifact, and policy contracts for external evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import torch


IDENTITY_FIELDS = (
    "schema",
    "sequence_length",
    "feedback_names",
    "sparse_names",
    "sparse_vocabularies",
    "dense_names",
    "catalog_sha256",
)


def stream_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash large artifacts without loading them into process memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    catalog_path = dataset_dir / "random_item_catalog.pt"
    if "catalog_sha256" in manifest:
        actual = stream_sha256(catalog_path)
        if actual != manifest["catalog_sha256"]:
            raise ValueError("dataset catalog hash does not match its manifest")
    for name, record in manifest.get("splits", {}).items():
        path = dataset_dir / f"{name}.pt"
        if stream_sha256(path) != record["sha256"]:
            raise ValueError(f"dataset split hash mismatch: {name}")
    return manifest


def _normalized(value):
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def assert_artifact_compatible(
    dataset_dir: Path, artifacts: Iterable[Path]
) -> dict:
    """Fail before scoring if model and dataset identities are not closed."""
    current = load_dataset_manifest(dataset_dir)
    for artifact in artifacts:
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        trained = payload.get("dataset_manifest")
        if not isinstance(trained, dict):
            raise ValueError(f"artifact has no dataset manifest: {artifact}")
        mismatches = [
            field
            for field in IDENTITY_FIELDS
            if _normalized(trained.get(field)) != _normalized(current.get(field))
        ]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(
                f"artifact/dataset identity mismatch for {artifact.name}: {joined}"
            )
    return current


@dataclass(frozen=True)
class PolicySpec:
    """One authority for stochastic ranking behavior used by OPE and replay."""

    utility_mode: str = "raw_probability"
    temperature: float = 0.20
    uniform_mixture: float = 0.50
    minimum_standard_exposures: int = 5

    def __post_init__(self) -> None:
        if self.utility_mode not in {"raw_probability", "standardized_feed"}:
            raise ValueError(f"unsupported policy utility: {self.utility_mode}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.uniform_mixture <= 1:
            raise ValueError("uniform_mixture must be in [0, 1]")
        if self.minimum_standard_exposures < 0:
            raise ValueError("minimum_standard_exposures must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)
