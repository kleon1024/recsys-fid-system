"""Hashed sparse-FID LR with dense features and online-compatible scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..platform.features import FeatureTensorBatch
from .contracts import Lane, ProbeBatch
from .probe import ProbeArtifact


DEFAULT_FID_HASH_SIZE = 1 << 18


class HashedSparseLinearRanker(nn.Module):
    def __init__(
        self,
        dense_inputs: int,
        sparse_fields: int,
        tasks: int,
        hash_size: int = DEFAULT_FID_HASH_SIZE,
        surfaces: int = 6,
    ) -> None:
        super().__init__()
        if min(dense_inputs, sparse_fields, tasks, hash_size, surfaces) <= 0:
            raise ValueError("hashed sparse LR dimensions must be positive")
        self.dense_inputs = dense_inputs
        self.sparse_fields = sparse_fields
        self.tasks = tasks
        self.hash_size = hash_size
        self.surfaces = surfaces
        self.dense = nn.Linear(dense_inputs + surfaces, tasks)
        self.fid = nn.Embedding(hash_size, tasks, sparse=True)
        nn.init.zeros_(self.fid.weight)

    def _fid_index(self, sparse: torch.Tensor) -> torch.Tensor:
        field = torch.arange(
            self.sparse_fields, device=sparse.device, dtype=torch.long,
        )
        salt = field * 2_654_435_761
        return torch.remainder(sparse.long() + salt, self.hash_size)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            surface.long().clamp(0, self.surfaces - 1), self.surfaces,
        ).to(dense.dtype)
        dense_logit = self.dense(torch.cat((dense, one_hot), dim=1))
        sparse_logit = self.fid(self._fid_index(sparse)).sum(dim=1)
        return dense_logit + sparse_logit


@dataclass(frozen=True)
class SparseLinearArtifact:
    model: HashedSparseLinearRanker
    task_names: tuple[str, ...]
    feature_manifest_hash: str
    training_report: dict[str, object]
    dense_mean: torch.Tensor
    dense_scale: torch.Tensor
    serving_task_weights: tuple[float, ...] | None = None
    task_logit_offsets: torch.Tensor | None = None
    conditional_task_parents: tuple[int, ...] | None = None

    @property
    def model_name(self) -> str:
        return "v4-hashed-sparse-fid-lr"

    @torch.inference_mode()
    def predict_task_probabilities(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        parameter = next(self.model.parameters())
        dense = dense.to(parameter.device, parameter.dtype)
        sparse = sparse.to(parameter.device)
        surface = surface.to(parameter.device)
        mean = self.dense_mean.to(parameter.device, parameter.dtype)
        scale = self.dense_scale.to(parameter.device, parameter.dtype)
        logits = self.model((dense - mean) / scale, sparse, surface)
        offset = (
            torch.zeros(logits.shape[1], device=logits.device)
            if self.task_logit_offsets is None
            else self.task_logit_offsets.to(logits.device, logits.dtype)
        )
        raw = torch.sigmoid(logits - offset)
        probability = raw.clone()
        parents = self.conditional_task_parents or (-1,) * len(self.task_names)
        for child, parent in enumerate(parents):
            if parent >= 0:
                probability[:, child] = probability[:, parent] * raw[:, child]
        return probability

    @torch.inference_mode()
    def score(
        self, features: FeatureTensorBatch, surface: torch.Tensor,
    ) -> torch.Tensor:
        if features.manifest_hash != self.feature_manifest_hash:
            raise ValueError("sparse LR feature manifest differs")
        requests, candidates = features.dense.shape[:2]
        probability = self.predict_task_probabilities(
            features.dense.reshape(-1, features.dense.shape[2]),
            features.sparse_buckets.reshape(-1, features.sparse_buckets.shape[2]),
            surface[:, None].expand(requests, candidates).reshape(-1),
        )
        if self.serving_task_weights is None:
            score = probability[:, self.task_names.index("long_view")]
        else:
            weight = torch.tensor(
                self.serving_task_weights,
                device=probability.device,
                dtype=probability.dtype,
            )
            score = (probability * weight).sum(dim=1).clamp_min(0.0)
        return score.reshape(requests, candidates).to(features.dense.device)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema": "v4-hashed-sparse-fid-lr-v1",
            "dense_inputs": self.model.dense_inputs,
            "sparse_fields": self.model.sparse_fields,
            "tasks": self.model.tasks,
            "hash_size": self.model.hash_size,
            "surfaces": self.model.surfaces,
            "task_names": self.task_names,
            "feature_manifest_hash": self.feature_manifest_hash,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "training_report": self.training_report,
            "dense_mean": self.dense_mean.detach().cpu().clone(),
            "dense_scale": self.dense_scale.detach().cpu().clone(),
            "serving_task_weights": self.serving_task_weights,
            "task_logit_offsets": self.task_logit_offsets,
            "conditional_task_parents": self.conditional_task_parents,
        }

    @classmethod
    def from_checkpoint(cls, value: dict[str, object]) -> SparseLinearArtifact:
        if value.get("schema") != "v4-hashed-sparse-fid-lr-v1":
            raise ValueError("hashed sparse LR checkpoint schema is unsupported")
        model = HashedSparseLinearRanker(
            int(value["dense_inputs"]),
            int(value["sparse_fields"]),
            int(value["tasks"]),
            int(value["hash_size"]),
            int(value["surfaces"]),
        )
        model.load_state_dict(value["state_dict"])
        model.eval()
        return cls(
            model=model,
            task_names=tuple(value["task_names"]),
            feature_manifest_hash=str(value["feature_manifest_hash"]),
            training_report=dict(value["training_report"]),
            dense_mean=value["dense_mean"].detach().cpu().float(),
            dense_scale=value["dense_scale"].detach().cpu().float(),
            serving_task_weights=(
                None if value.get("serving_task_weights") is None
                else tuple(value["serving_task_weights"])
            ),
            task_logit_offsets=(
                None if value.get("task_logit_offsets") is None
                else value["task_logit_offsets"].detach().cpu().float()
            ),
            conditional_task_parents=(
                None if value.get("conditional_task_parents") is None
                else tuple(value["conditional_task_parents"])
            ),
        )


def _conditional_parents(task_names: tuple[str, ...]) -> tuple[int, ...]:
    hierarchy = {
        "play_3s": "play",
        "long_view": "play_3s",
        "complete": "long_view",
    }
    return tuple(
        task_names.index(hierarchy[name])
        if name in hierarchy and hierarchy[name] in task_names else -1
        for name in task_names
    )


def _warm_start_dense(
    model: HashedSparseLinearRanker,
    initial: ProbeArtifact,
    batch: ProbeBatch,
    dense_mean: torch.Tensor,
    dense_scale: torch.Tensor,
    device: torch.device,
) -> list[str]:
    if initial.feature_manifest_hash != batch.feature_manifest_hash:
        raise ValueError("warm-start feature manifest differs")
    if initial.model.inputs != model.dense_inputs:
        raise ValueError("warm-start dense input shape differs")
    warmed = []
    with torch.no_grad():
        for task in set(batch.task_names) & set(initial.task_names):
            source = initial.task_names.index(task)
            target = batch.task_names.index(task)
            old_weight = initial.model.linear.weight[source].to(device)
            adjusted = old_weight.clone()
            adjusted[:model.dense_inputs] *= (
                dense_scale / initial.dense_scale.to(device)
            )
            model.dense.weight[target].copy_(adjusted)
            model.dense.bias[target].copy_(
                initial.model.linear.bias[source].to(device)
                + (old_weight[:model.dense_inputs] * (
                    dense_mean - initial.dense_mean.to(device)
                ) / initial.dense_scale.to(device)).sum()
            )
            warmed.append(task)
    return sorted(warmed)


def train_sparse_linear(
    batch: ProbeBatch,
    *,
    lane: Lane,
    initial_dense_artifact: ProbeArtifact | None = None,
    epochs: int = 3,
    learning_rate: float = 3e-3,
    hash_size: int = DEFAULT_FID_HASH_SIZE,
    device: str | torch.device = "cpu",
    seed: int = 2_026_082_6,
    batch_size: int = 65_536,
) -> SparseLinearArtifact:
    if min(epochs, batch_size, hash_size) <= 0 or learning_rate <= 0.0:
        raise ValueError("sparse LR training configuration must be positive")
    target = torch.device(device)
    torch.manual_seed(seed)
    dense = batch.dense_features.to(target)
    dense_mean = dense.mean(dim=0)
    dense_scale = dense.std(dim=0, unbiased=False).clamp_min(1e-4)
    model = HashedSparseLinearRanker(
        dense.shape[1], batch.sparse_buckets.shape[1], len(batch.task_names),
        hash_size,
    ).to(target)
    warmed = (
        [] if initial_dense_artifact is None else _warm_start_dense(
            model, initial_dense_artifact, batch, dense_mean, dense_scale, target,
        )
    )
    parents = _conditional_parents(batch.task_names)
    labels = batch.labels.to(target)
    masks = batch.label_mask.to(target).clone()
    for child, parent in enumerate(parents):
        if parent >= 0:
            masks[:, child] &= labels[:, parent] > 0.5
    if not masks.any():
        raise ValueError("sparse LR has no mature labels")
    positive = (labels * masks).sum(dim=0)
    observed = masks.sum(dim=0)
    positive_weight = ((observed - positive) / positive.clamp_min(1.0)).clamp(1, 50)
    if "stay_value" in batch.task_names:
        positive_weight[batch.task_names.index("stay_value")] = 1.0
    dense_optimizer = torch.optim.AdamW(model.dense.parameters(), lr=learning_rate)
    sparse_optimizer = torch.optim.SparseAdam(model.fid.parameters(), lr=learning_rate)
    normalized = (dense - dense_mean) / dense_scale
    sparse = batch.sparse_buckets.to(target)
    surface = batch.surface.to(target)
    propensity = batch.joint_logging_probability.to(target)
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(dense), generator=generator, device=target)
        for start in range(0, len(order), batch_size):
            row = order[start:start + batch_size]
            logits = model(normalized[row], sparse[row], surface[row])
            element = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[row], reduction="none", pos_weight=positive_weight,
            )
            weight = (1.0 / propensity[row].clamp_min(0.05)).clamp_max(10)[:, None]
            loss = (element * masks[row] * weight).sum() / masks[row].sum().clamp_min(1)
            dense_optimizer.zero_grad(set_to_none=True)
            sparse_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            dense_optimizer.step()
            sparse_optimizer.step()
            losses.append(float(loss.detach()))
    model.eval()
    return SparseLinearArtifact(
        model=model,
        task_names=batch.task_names,
        feature_manifest_hash=batch.feature_manifest_hash,
        training_report={
            "purpose": "hashed_sparse_fid_lr_candidate",
            "lane": lane.value,
            "rows": len(batch.request_id),
            "mature_labels": int(masks.sum()),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "epochs": epochs,
            "hash_size": hash_size,
            "hash_load": float(
                torch.unique(model._fid_index(sparse)).numel() / hash_size
            ),
            "warm_started_tasks": warmed,
            "device": str(target),
            "seed": seed,
        },
        dense_mean=dense_mean.detach().cpu(),
        dense_scale=dense_scale.detach().cpu(),
        task_logit_offsets=torch.log(positive_weight).detach().cpu(),
        conditional_task_parents=parents,
    )
