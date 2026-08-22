"""Different model families for surfaces with different scale and semantics."""

from __future__ import annotations

import torch
from torch import nn

from ..multitask import MultiGateMixtureOfExperts
from .contracts import SurfaceSpec


def task_heads(input_dim: int, tasks: tuple[str, ...]) -> nn.ModuleDict:
    return nn.ModuleDict({task: nn.Linear(input_dim, 1) for task in tasks})


class FeedVideoRanker(nn.Module):
    """Heavy sequence ranker for the highest-volume mixed Feed surface."""

    def __init__(self, spec: SurfaceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(len(spec.features)),
            nn.Linear(len(spec.features), 48),
            nn.ReLU(),
        )
        self.sequence_projection = nn.Linear(8, 48)
        layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            batch_first=True,
            dropout=0.0,
        )
        self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.mmoe = MultiGateMixtureOfExperts(96, spec.task_names, 4, 48)

    def forward(
        self, features: torch.Tensor, sequence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if sequence is None:
            raise ValueError("Feed ranker requires a behavior sequence")
        candidate = self.feature_projection(features)
        history = self.sequence_encoder(self.sequence_projection(sequence))[:, -1, :]
        return self.mmoe(torch.cat([candidate, history], dim=1))


class MapDetailRanker(nn.Module):
    """Geography-heavy Wide & Deep style ranker."""

    def __init__(self, spec: SurfaceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.wide = nn.Linear(len(spec.features), len(spec.tasks))
        self.deep = nn.Sequential(
            nn.Linear(len(spec.features), 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
        )
        self.heads = task_heads(24, spec.task_names)

    def forward(
        self, features: torch.Tensor, sequence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del sequence
        deep = self.deep(features)
        wide = self.wide(features)
        return {
            task: self.heads[task](deep).squeeze(1) + wide[:, index]
            for index, task in enumerate(self.spec.task_names)
        }


class YmalTwoTowerRanker(nn.Module):
    """Related-POI query/candidate representation model."""

    def __init__(self, spec: SurfaceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.split = len(spec.features) // 2
        self.query_tower = nn.Sequential(nn.Linear(self.split, 24), nn.ReLU())
        self.item_tower = nn.Sequential(
            nn.Linear(len(spec.features) - self.split, 24), nn.ReLU()
        )
        self.heads = task_heads(49, spec.task_names)

    def forward(
        self, features: torch.Tensor, sequence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del sequence
        query = self.query_tower(features[:, : self.split])
        item = self.item_tower(features[:, self.split :])
        similarity = (query * item).sum(dim=1, keepdim=True) / query.shape[1] ** 0.5
        combined = torch.cat([query, item, similarity], dim=1)
        return {
            task: self.heads[task](combined).squeeze(1)
            for task in self.spec.task_names
        }


class ProductFunnelRanker(nn.Module):
    """Task-routed model for click-to-payment funnel sparsity."""

    def __init__(self, spec: SurfaceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.projection = nn.Sequential(nn.Linear(len(spec.features), 32), nn.ReLU())
        self.mmoe = MultiGateMixtureOfExperts(32, spec.task_names, 3, 32)

    def forward(
        self, features: torch.Tensor, sequence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del sequence
        return self.mmoe(self.projection(features))


class ReviewNlpRanker(nn.Module):
    """Light ranker consuming precomputed review-language signals."""

    def __init__(self, spec: SurfaceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder_adapter = nn.Sequential(
            nn.Linear(len(spec.features), 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
        )
        self.heads = task_heads(24, spec.task_names)

    def forward(
        self, features: torch.Tensor, sequence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del sequence
        state = self.encoder_adapter(features)
        return {
            task: self.heads[task](state).squeeze(1)
            for task in self.spec.task_names
        }


def build_surface_model(spec: SurfaceSpec) -> nn.Module:
    builders = {
        "feed_poi_video": FeedVideoRanker,
        "poi_map_detail": MapDetailRanker,
        "ymal": YmalTwoTowerRanker,
        "product": ProductFunnelRanker,
        "review": ReviewNlpRanker,
    }
    return builders[spec.name](spec)
