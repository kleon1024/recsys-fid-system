"""Minimal LR probe proving v4 persisted samples can train and replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
from torch import nn

from ..observability import FullFlowPartitionRef
from ..platform.features import FeatureTensorBatch
from .arrow import list_column_to_tensor
from .contracts import Lane, ProbeBatch
from .sample_bus import PartitionedSampleBus


def load_probe_batch(
    bus: PartitionedSampleBus,
    refs: tuple[FullFlowPartitionRef, ...],
) -> ProbeBatch:
    if not refs:
        raise ValueError("probe requires at least one partition")
    tables = [bus.fine_examples(ref) for ref in refs]
    table = pa.concat_tables(tables)
    if not len(table):
        raise ValueError("probe partitions contain no fine examples")
    contract = bus.contract()
    sample = contract["sample_contract"]
    trace = contract["trace_manifest"]
    order = np.lexsort((
        table["ordinal"].to_numpy(),
        table["request_id"].to_numpy(),
        table["request_time"].to_numpy(),
    ))

    def scalar(name: str, dtype: torch.dtype) -> torch.Tensor:
        values = table[name].to_numpy(zero_copy_only=False)[order]
        return torch.as_tensor(values.copy(), dtype=dtype)

    dense = list_column_to_tensor(table["dense_features"], torch.float32)[order]
    sparse = list_column_to_tensor(table["sparse_buckets"], torch.long)[order]
    labels = list_column_to_tensor(table["task_label_values"], torch.float32)[order]
    masks = list_column_to_tensor(table["task_label_masks"], torch.bool)[order]
    return ProbeBatch(
        request_id=scalar("request_id", torch.long),
        user_id=scalar("user_id", torch.long),
        surface=scalar("surface", torch.long),
        request_time=scalar("request_time", torch.long),
        item_id=scalar("item_id", torch.long),
        dense_features=dense,
        sparse_buckets=sparse,
        labels=labels,
        label_mask=masks,
        joint_logging_probability=scalar(
            "joint_logging_probability", torch.float32,
        ),
        task_names=tuple(sample["task_names"]),
        dense_feature_names=tuple(sample["dense_feature_names"]),
        sparse_feature_names=tuple(sample["sparse_feature_names"]),
        feature_manifest_hash=str(trace["feature_manifest_hash"]),
        partition_content_hashes=tuple(ref.content_sha256 for ref in refs),
        event_watermark=max(ref.event_watermark for ref in refs),
    )


def feature_drift_report(
    reference: ProbeBatch,
    current: ProbeBatch,
) -> dict[str, object]:
    if reference.feature_manifest_hash != current.feature_manifest_hash:
        raise ValueError("feature drift requires one feature manifest")
    dense = {}
    for index, name in enumerate(reference.dense_feature_names):
        before = reference.dense_features[:, index].float()
        after = current.dense_features[:, index].float()
        before_mean = float(before.mean())
        after_mean = float(after.mean())
        before_std = float(before.std(unbiased=False))
        after_std = float(after.std(unbiased=False))
        dense[name] = {
            "reference_mean": before_mean,
            "current_mean": after_mean,
            "standardized_mean_shift": abs(after_mean - before_mean)
            / max(before_std, 1e-6),
            "std_ratio": after_std / max(before_std, 1e-6),
        }
    sparse = {}
    for index, name in enumerate(reference.sparse_feature_names):
        known = torch.unique(reference.sparse_buckets[:, index])
        values = current.sparse_buckets[:, index]
        sparse[name] = {
            "reference_unique": int(known.numel()),
            "current_unique": int(torch.unique(values).numel()),
            "unseen_bucket_rate": float((~torch.isin(values, known)).float().mean()),
        }
    return {
        "schema": "feature-drift-report-v1",
        "feature_manifest_hash": reference.feature_manifest_hash,
        "reference_rows": len(reference.request_id),
        "current_rows": len(current.request_id),
        "dense": dense,
        "sparse": sparse,
    }


class ProbeRanker(nn.Module):
    def __init__(self, inputs: int, tasks: int, surfaces: int = 6) -> None:
        super().__init__()
        self.inputs = inputs
        self.tasks = tasks
        self.surfaces = surfaces
        self.linear = nn.Linear(inputs + surfaces, tasks)

    def forward(self, dense: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            surface.long().clamp(0, self.surfaces - 1), self.surfaces,
        ).to(dense.dtype)
        return self.linear(torch.cat((dense, one_hot), dim=1))


@dataclass(frozen=True)
class ProbeArtifact:
    model: ProbeRanker
    task_names: tuple[str, ...]
    feature_manifest_hash: str
    training_report: dict[str, object]

    @property
    def model_name(self) -> str:
        return "v4-lr-infrastructure-probe"

    def validate_compatibility(self, expected) -> None:
        if self.feature_manifest_hash != expected.feature_manifest_hash:
            raise ValueError("probe artifact feature manifest differs")
        if self.feature_manifest_hash != expected.stage_contract_hash:
            raise ValueError("probe artifact stage contract differs")

    @torch.inference_mode()
    def score(
        self,
        features: FeatureTensorBatch,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        if features.manifest_hash != self.feature_manifest_hash:
            raise ValueError("probe scoring feature manifest differs")
        parameter = next(self.model.parameters())
        requests, candidates = features.dense.shape[:2]
        dense = features.dense.reshape(
            -1, features.dense.shape[2]
        ).to(parameter.device, parameter.dtype)
        expanded_surface = surface[:, None].expand(
            requests, candidates,
        ).reshape(-1).to(parameter.device)
        logits = self.model(dense, expanded_surface)
        task = self.task_names.index("long_view")
        score = torch.sigmoid(logits[:, task]).reshape(requests, candidates)
        return score.to(features.dense.device)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema": "v4-lr-infrastructure-probe-v1",
            "inputs": self.model.inputs,
            "tasks": self.model.tasks,
            "surfaces": self.model.surfaces,
            "task_names": self.task_names,
            "feature_manifest_hash": self.feature_manifest_hash,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "training_report": self.training_report,
        }

    @classmethod
    def from_checkpoint(cls, value: dict[str, object]) -> ProbeArtifact:
        if value.get("schema") != "v4-lr-infrastructure-probe-v1":
            raise ValueError("probe checkpoint schema is unsupported")
        model = ProbeRanker(
            int(value["inputs"]), int(value["tasks"]), int(value["surfaces"]),
        )
        model.load_state_dict(value["state_dict"])
        model.eval()
        return cls(
            model,
            tuple(value["task_names"]),
            str(value["feature_manifest_hash"]),
            dict(value["training_report"]),
        )


def train_probe(
    batch: ProbeBatch,
    *,
    lane: Lane,
    epochs: int = 2,
    learning_rate: float = 3e-3,
    device: str | torch.device = "cpu",
    seed: int = 20260825,
    batch_size: int = 65_536,
) -> ProbeArtifact:
    if epochs <= 0 or learning_rate <= 0.0 or batch_size <= 0:
        raise ValueError("probe training configuration must be positive")
    target = torch.device(device)
    torch.manual_seed(seed)
    model = ProbeRanker(
        batch.dense_features.shape[1], len(batch.task_names),
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dense = batch.dense_features.to(target)
    surface = batch.surface.to(target)
    labels = batch.labels.to(target)
    masks = batch.label_mask.to(target)
    propensity = batch.joint_logging_probability.to(target)
    if not masks.any():
        raise ValueError("probe has no mature labels")
    losses = []
    model.train()
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    optimizer_steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(dense), generator=generator, device=target)
        for start in range(0, len(order), batch_size):
            row = order[start:start + batch_size]
            logits = model(dense[row], surface[row])
            element = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[row], reduction="none",
            )
            weight = (
                1.0 / propensity[row].clamp_min(0.05)
            ).clamp_max(10.0)[:, None]
            observed = masks[row]
            loss = (
                element * observed * weight
            ).sum() / observed.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            optimizer_steps += 1
    model.eval()
    return ProbeArtifact(
        model=model,
        task_names=batch.task_names,
        feature_manifest_hash=batch.feature_manifest_hash,
        training_report={
            "purpose": "infrastructure_only_not_model_launch",
            "lane": lane.value,
            "rows": len(batch.request_id),
            "mature_labels": int(masks.sum()),
            "event_watermark": batch.event_watermark,
            "partition_content_hashes": list(batch.partition_content_hashes),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_steps": optimizer_steps,
            "device": str(target),
            "seed": seed,
        },
    )
