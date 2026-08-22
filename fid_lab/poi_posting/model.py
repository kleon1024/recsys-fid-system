"""Multimodal content/POI encoder with an MMoE ranking head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from ..multitask import MultiGateMixtureOfExperts
from .contracts import PoiPostingConfig, TASKS


class PoiPostingRanker(nn.Module):
    def __init__(self, config: PoiPostingConfig) -> None:
        super().__init__()
        self.config = config
        rep = config.representation_dim
        cat = config.categorical_dim
        self.author_embedding = nn.Embedding(config.authors, cat)
        self.poi_embedding = nn.Embedding(config.pois, cat)
        self.city_embedding = nn.Embedding(config.cities, cat)
        self.category_embedding = nn.Embedding(config.categories, cat)
        self.permission_embedding = nn.Embedding(3, cat)
        self.poi_projection = nn.Linear(config.raw_semantic_dim, rep)
        self.poi_context_projection = nn.Linear(cat * 2, rep)
        input_dim = cat * 5 + rep * 3 + 5
        self.mmoe = MultiGateMixtureOfExperts(input_dim, TASKS, config.experts)

    def encode_poi(
        self,
        poi_features: torch.Tensor,
        city_id: torch.Tensor,
        category_id: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.poi_projection(poi_features)
        context = self.poi_context_projection(
            torch.cat(
                [self.city_embedding(city_id), self.category_embedding(category_id)],
                dim=1,
            )
        )
        return functional.normalize(torch.tanh(semantic + context), dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        content = functional.normalize(batch["content_features"], dim=1)
        poi = self.encode_poi(
            batch["poi_features"], batch["city_id"], batch["category_id"]
        )
        similarity = (content * poi).sum(dim=1, keepdim=True)
        inputs = torch.cat(
            [
                self.author_embedding(batch["author_id"]),
                self.poi_embedding(batch["poi_id"]),
                self.city_embedding(batch["city_id"]),
                self.category_embedding(batch["category_id"]),
                self.permission_embedding(batch["permission_id"]),
                content,
                poi,
                content * poi,
                batch["numeric_features"],
                similarity,
            ],
            dim=1,
        )
        outputs: dict[str, torch.Tensor] = {
            "similarity": similarity.squeeze(1),
        }
        outputs.update(self.mmoe(inputs))
        return outputs
