"""Adapters from existing reusable ranker blocks to the twin task contract."""

from __future__ import annotations

import torch
from deepctr_torch.layers.core import DNN
from deepctr_torch.layers.interaction import CrossNetMix, FM
from torch import nn

from ....multitask import MultiGateMixtureOfExperts
from ..platform.fids import TWIN_FID_FIELDS


SPARSE_ARCHITECTURES = frozenset({"wide_deep", "deepfm", "dcnv2"})


class SparseFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 8):
        super().__init__()
        self.deep = nn.ModuleList(
            nn.Embedding(field.buckets, embedding_dim, padding_idx=0)
            for field in TWIN_FID_FIELDS
        )
        self.wide = nn.ModuleList(
            nn.Embedding(field.buckets, 1, padding_idx=0)
            for field in TWIN_FID_FIELDS
        )

    def forward(self, buckets):
        deep = torch.stack(tuple(
            embedding(buckets[:, index])
            for index, embedding in enumerate(self.deep)
        ), dim=1)
        wide = torch.cat(tuple(
            embedding(buckets[:, index])
            for index, embedding in enumerate(self.wide)
        ), dim=1)
        return deep, wide


class DeepCTRMultiTaskNetwork(nn.Module):
    """Shared sparse tables with mature DeepCTR interaction primitives."""

    def __init__(
        self, architecture: str, dense_inputs: int, outputs: int,
        hidden: int, embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.sparse = SparseFeatureEncoder(embedding_dim)
        sparse_inputs = len(TWIN_FID_FIELDS) * embedding_dim
        combined = dense_inputs + sparse_inputs
        self.wide_head = nn.Linear(
            dense_inputs + len(TWIN_FID_FIELDS), outputs
        )
        self.deep = DNN(
            combined, (hidden * 2, hidden), activation="relu",
            dropout_rate=0.10, device="cpu",
        )
        self.deep_head = nn.Linear(hidden, outputs)
        self.fm = FM()
        self.fm_scale = nn.Parameter(torch.ones(outputs))
        self.cross = CrossNetMix(
            combined, low_rank=32, num_experts=4, layer_num=2,
            device="cpu",
        )
        self.cross_head = nn.Linear(combined + hidden, outputs)

    def forward(self, dense, buckets):
        embedding, wide = self.sparse(buckets)
        flattened = embedding.flatten(1)
        combined = torch.cat((dense, flattened), dim=1)
        wide_logit = self.wide_head(torch.cat((dense, wide), dim=1))
        deep = self.deep(combined)
        if self.architecture == "wide_deep":
            return wide_logit + self.deep_head(deep)
        if self.architecture == "deepfm":
            return (
                wide_logit + self.deep_head(deep)
                + self.fm(embedding) * self.fm_scale[None]
            )
        crossed = self.cross(combined)
        return wide_logit + self.cross_head(torch.cat((crossed, deep), dim=1))


class MMoENetwork(nn.Module):
    def __init__(
        self, inputs: int, tasks: tuple[str, ...], hidden: int,
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.mmoe = MultiGateMixtureOfExperts(
            inputs, tasks, expert_count=6, expert_dim=hidden
        )

    def forward(self, values):
        output = self.mmoe(values)
        return torch.stack(tuple(output[task] for task in self.tasks), dim=1)


def build_network(
    architecture: str,
    inputs: int,
    tasks: tuple[str, ...],
    hidden: int,
) -> nn.Module:
    outputs = len(tasks)
    if architecture in SPARSE_ARCHITECTURES:
        return DeepCTRMultiTaskNetwork(
            architecture, inputs, outputs, hidden
        )
    if architecture == "lr":
        return nn.Linear(inputs, outputs)
    if architecture == "mlp":
        return nn.Sequential(
            nn.Linear(inputs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, outputs),
        )
    if architecture == "mmoe":
        return MMoENetwork(inputs, tasks, hidden)
    raise ValueError(f"unsupported ranker architecture: {architecture}")
