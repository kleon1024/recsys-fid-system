"""POI Detail page, module, task, and scale contracts."""

from __future__ import annotations

from dataclasses import dataclass


DETAIL_MODULES = ("related_poi", "product", "review")
DETAIL_TASKS = ("click", "deep_action", "transaction", "negative")


@dataclass(frozen=True)
class PoiDetailConfig:
    requests: int = 100_000
    users: int = 50_000
    entities_per_module: int = 8_192
    categories: int = 32
    semantic_dim: int = 16
    history_length: int = 24
    candidates_per_module: int = 8
    exposed_related: int = 4
    exposed_product: int = 2
    exposed_review: int = 2
    train_epochs: int = 3
    train_batch_requests: int = 1_024
    learning_rate: float = 1.5e-3
    seed: int = 20260824
    device: str = "cuda:0"

    @property
    def candidates(self):
        return self.candidates_per_module * len(DETAIL_MODULES)

    @property
    def exposed(self):
        return self.exposed_related + self.exposed_product + self.exposed_review

    def __post_init__(self):
        if self.requests < 400 or self.users > self.requests:
            raise ValueError("invalid POI Detail request population")
        if self.semantic_dim % 4:
            raise ValueError("semantic width must support four attention heads")
        if self.entities_per_module % self.categories:
            raise ValueError("module corpus must partition by category")
        quotas = (
            self.exposed_related, self.exposed_product, self.exposed_review
        )
        if quotas != (4, 2, 2):
            raise ValueError("POI Detail v1 requires the 4/2/2 module quota")
        if max(quotas) > self.candidates_per_module:
            raise ValueError("module exposure quota exceeds candidates")
