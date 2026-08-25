"""ANN-compatible observable Two-Tower and Multi-interest retrievers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from ...contracts import Surface
from ...platform.projection import USER_COUNTER_EVENTS
from .contracts import (
    DEFAULT_RETRIEVAL_FEATURE_CONTRACT,
    RetrievalCorpus,
    RetrievalModelConfig,
    RetrievalQueryBatch,
)


def _bucket(values: torch.Tensor, buckets: int) -> torch.Tensor:
    return torch.remainder(values.clamp_min(-1) + 1, buckets - 1) + 1


class ObservableRetrievalModel(nn.Module):
    def __init__(self, corpus: RetrievalCorpus, config: RetrievalModelConfig) -> None:
        super().__init__()
        self.config = config
        self.content_dim = corpus.content_embedding.shape[1]
        self.user = nn.Embedding(config.user_hash_buckets, config.embedding_dim)
        self.surface = nn.Embedding(16, config.embedding_dim)
        self.query_topic = nn.Embedding(
            int(corpus.topic_id.max()) + 2, config.embedding_dim,
        )
        self.query_country = nn.Embedding(
            int(corpus.country.max()) + 2, config.embedding_dim,
        )
        self.query_region = nn.Embedding(
            int(corpus.region.max()) + 2, config.embedding_dim,
        )
        self.item = nn.Embedding(config.item_hash_buckets, config.embedding_dim)
        self.item_kind = nn.Embedding(
            int(corpus.content_kind.max()) + 2, config.embedding_dim,
        )
        self.item_topic = nn.Embedding(
            int(corpus.topic_id.max()) + 2, config.embedding_dim,
        )
        self.item_creator = nn.Embedding(
            config.creator_hash_buckets, config.embedding_dim,
        )
        self.item_country = nn.Embedding(
            int(corpus.country.max()) + 2, config.embedding_dim,
        )
        self.item_region = nn.Embedding(
            int(corpus.region.max()) + 2, config.embedding_dim,
        )
        self.history_projection = nn.Linear(self.content_dim, config.hidden_dim)
        query_dense = len(USER_COUNTER_EVENTS) + len(Surface) + 2
        query_sparse = 5 * config.embedding_dim
        query_inputs = query_sparse + query_dense + config.hidden_dim
        item_sparse = 6 * config.embedding_dim
        item_inputs = item_sparse + self.content_dim + 3
        self.query_tower = nn.Sequential(
            nn.Linear(query_inputs, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.representation_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_inputs, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.representation_dim),
        )
        self.interest_queries = nn.Parameter(
            torch.randn(config.interests, config.hidden_dim) * 0.02,
        )
        for name, value in corpus.tensors.items():
            self.register_buffer(f"corpus_{name}", value.clone(), persistent=False)

    def _query_base(
        self,
        user_id: torch.Tensor,
        surface: torch.Tensor,
        event_time: torch.Tensor,
        query_topic: torch.Tensor,
        country: torch.Tensor,
        region: torch.Tensor,
        user_counts: torch.Tensor,
        surface_counts: torch.Tensor,
    ) -> torch.Tensor:
        contract = DEFAULT_RETRIEVAL_FEATURE_CONTRACT
        angle = (
            2.0 * math.pi * torch.remainder(event_time, contract.daily_ticks).float()
            / contract.daily_ticks
        )
        dense = torch.cat((
            torch.log1p(user_counts.float()) / contract.counter_log_scale,
            torch.log1p(surface_counts.float()) / contract.counter_log_scale,
            torch.sin(angle)[:, None],
            torch.cos(angle)[:, None],
        ), dim=1)
        sparse = torch.cat((
            self.user(_bucket(user_id, self.config.user_hash_buckets)),
            self.surface(_bucket(surface, 16)),
            self.query_topic(_bucket(query_topic, self.query_topic.num_embeddings)),
            self.query_country(_bucket(country, self.query_country.num_embeddings)),
            self.query_region(_bucket(region, self.query_region.num_embeddings)),
        ), dim=1)
        return torch.cat((sparse, dense), dim=1)

    def _history(self, history_item_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = history_item_id >= 0
        content = self.corpus_content_embedding[history_item_id.clamp_min(0)]
        return self.history_projection(content), valid

    def encode_query_tensors(
        self,
        *,
        user_id: torch.Tensor,
        surface: torch.Tensor,
        event_time: torch.Tensor,
        query_topic: torch.Tensor,
        country: torch.Tensor,
        region: torch.Tensor,
        user_counts: torch.Tensor,
        surface_counts: torch.Tensor,
        history_item_id: torch.Tensor,
    ) -> torch.Tensor:
        base = self._query_base(
            user_id, surface, event_time, query_topic, country, region,
            user_counts, surface_counts,
        )
        history, valid = self._history(history_item_id)
        if self.config.architecture == "multi_interest":
            logits = torch.einsum("kh,blh->bkl", self.interest_queries, history)
            logits = logits.masked_fill(~valid[:, None], -torch.inf)
            empty = ~valid.any(dim=1)
            logits[empty] = 0.0
            attention = torch.softmax(logits, dim=2).masked_fill(
                ~valid[:, None], 0.0,
            )
            interests = torch.einsum("bkl,blh->bkh", attention, history)
            expanded = base[:, None].expand(-1, self.config.interests, -1)
            return functional.normalize(
                self.query_tower(torch.cat((expanded, interests), dim=2)), dim=2,
            )
        recency = torch.arange(
            1, history.shape[1] + 1, device=history.device, dtype=history.dtype,
        )
        weight = recency[None] * valid.float()
        pooled = torch.einsum("bl,blh->bh", weight, history) / weight.sum(
            dim=1, keepdim=True,
        ).clamp_min(1.0)
        return functional.normalize(
            self.query_tower(torch.cat((base, pooled), dim=1)), dim=1,
        )

    def encode_queries(self, batch: RetrievalQueryBatch) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.encode_query_tensors(
            user_id=batch.user_id.to(device),
            surface=batch.surface.to(device),
            event_time=batch.event_time.to(device),
            query_topic=batch.query_topic.to(device),
            country=batch.user_country.to(device),
            region=batch.user_region.to(device),
            user_counts=batch.user_event_counts.to(device),
            surface_counts=batch.user_surface_counts.to(device),
            history_item_id=batch.history_item_id.to(device),
        )

    def encode_items(self, item_id: torch.Tensor) -> torch.Tensor:
        item = item_id.clamp_min(0)
        dense = torch.cat((
            self.corpus_content_embedding[item],
            self.corpus_quality_prior[item, None],
            torch.log1p(self.corpus_duration_seconds[item])[:, None] / 6.0,
            torch.tanh(self.corpus_publish_time[item].float() / 720.0)[:, None],
        ), dim=1)
        sparse = torch.cat((
            self.item(_bucket(item, self.config.item_hash_buckets)),
            self.item_kind(_bucket(
                self.corpus_content_kind[item], self.item_kind.num_embeddings,
            )),
            self.item_topic(_bucket(
                self.corpus_topic_id[item], self.item_topic.num_embeddings,
            )),
            self.item_creator(_bucket(
                self.corpus_creator_id[item], self.config.creator_hash_buckets,
            )),
            self.item_country(_bucket(
                self.corpus_country[item], self.item_country.num_embeddings,
            )),
            self.item_region(_bucket(
                self.corpus_region[item], self.item_region.num_embeddings,
            )),
        ), dim=1)
        return functional.normalize(self.item_tower(torch.cat((sparse, dense), dim=1)), dim=1)


def retrieval_scores(query: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    if query.ndim == 3:
        return torch.einsum("bkd,bnd->bkn", query, item).max(dim=1).values
    return torch.einsum("bd,bnd->bn", query, item)
