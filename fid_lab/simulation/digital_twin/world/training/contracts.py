"""Contracts for held-out structural world-model data."""

from __future__ import annotations

from dataclasses import dataclass

STRUCTURAL_BRIDGE_SCHEMA = "v6-structural-neural-scm-bridge-v1"


@dataclass(frozen=True)
class StructuralBridgeConfig:
    rows: int = 300_000
    users: int = 100_000
    items: int = 200_000
    slate_width: int = 8
    ticks: int = 192
    max_extension_ticks: int = 128
    topics: int = 64
    countries: int = 12
    regions_per_country: int = 16
    embedding_dim: int = 32
    platform_seed: int = 701
    environment_seed: int = 709
    test_family_id: int = 5
    device: str = "cuda:0"

    def __post_init__(self):
        dimensions = (
            self.rows, self.users, self.items, self.slate_width, self.ticks,
            self.max_extension_ticks,
            self.topics, self.countries, self.regions_per_country,
            self.embedding_dim,
            self.test_family_id,
        )
        if min(dimensions) <= 0:
            raise ValueError("structural bridge dimensions must be positive")
        if self.rows < 5:
            raise ValueError("structural bridge requires at least five rows")

    def split_rows(self) -> dict[str, int]:
        train = int(self.rows * 0.60)
        validation = int(self.rows * 0.20)
        return {
            "train": train,
            "validation": validation,
            "test": self.rows - train - validation,
        }
