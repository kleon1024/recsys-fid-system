"""Multimodal content/POI encoder with an MMoE ranking head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .contracts import PoiPostingConfig, TASKS


class PoiPostingRanker(nn.Module):
    def __init__(self, config: PoiPostingConfig) -> None:
        super().__init__()
        self.config = config
        raw = config.raw_semantic_dim
        rep = config.representation_dim
        cat = config.categorical_dim
        self.author_embedding = nn.Embedding(config.authors, cat)
        self.poi_embedding = nn.Embedding(config.pois, cat)
        self.city_embedding = nn.Embedding(config.cities, cat)
        self.category_embedding = nn.Embedding(config.categories, cat)
        self.permission_embedding = nn.Embedding(3, cat)
        self.frame_projection = nn.Linear(raw, rep)
        self.text_projection = nn.Linear(raw, rep)
        self.poi_projection = nn.Linear(raw, rep)
        self.poi_context_projection = nn.Linear(cat * 2, rep)
        self.fusion_gate = nn.Linear(rep * 2, rep)
        input_dim = cat * 5 + rep * 3 + 5
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
            )
            for _ in range(config.experts)
        )
        self.gates = nn.ModuleDict(
            {task: nn.Linear(input_dim, config.experts) for task in TASKS}
        )
        self.heads = nn.ModuleDict({task: nn.Linear(32, 1) for task in TASKS})

    def encode_content(
        self, frames: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_states = torch.tanh(self.frame_projection(frames))
        text_state = torch.tanh(self.text_projection(text))
        attention_logits = (frame_states * text_state[:, None, :]).sum(dim=-1)
        attention = torch.softmax(attention_logits, dim=1)
        video_state = (attention[:, :, None] * frame_states).sum(dim=1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([video_state, text_state], dim=1)))
        content = gate * video_state + (1.0 - gate) * text_state
        return functional.normalize(content, dim=1), attention

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
        content, attention = self.encode_content(
            batch["frame_features"], batch["text_features"]
        )
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
        expert_states = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        outputs: dict[str, torch.Tensor] = {
            "similarity": similarity.squeeze(1),
            "frame_attention": attention,
        }
        for task in TASKS:
            gate = torch.softmax(self.gates[task](inputs), dim=1)
            state = (gate[:, :, None] * expert_states).sum(dim=1)
            outputs[task] = self.heads[task](state).squeeze(1)
        return outputs
