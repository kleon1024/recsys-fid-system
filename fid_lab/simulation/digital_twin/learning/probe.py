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
        if name not in table.column_names:
            if name == "dwell_ms":
                return torch.zeros(len(table), dtype=dtype)
            raise ValueError(f"probe sample column is missing: {name}")
        values = table[name].to_numpy(zero_copy_only=False)[order]
        return torch.as_tensor(values.copy(), dtype=dtype)

    dense = list_column_to_tensor(table["dense_features"], torch.float32)[order]
    sparse = list_column_to_tensor(table["sparse_buckets"], torch.long)[order]
    labels = list_column_to_tensor(table["task_label_values"], torch.float32)[order]
    masks = list_column_to_tensor(table["task_label_masks"], torch.bool)[order]
    applicable = list_column_to_tensor(
        table["task_label_applicable"], torch.bool,
    )[order]
    mature = list_column_to_tensor(
        table["task_label_mature"], torch.bool,
    )[order]
    surface = scalar("surface", torch.long)
    exposed = scalar("exposed", torch.bool)
    dwell_ms = scalar("dwell_ms", torch.float32)
    feed_exposure = exposed & (surface == 0)
    stay_value = torch.log1p(dwell_ms.clamp_min(0.0) / 1_000.0) / np.log1p(300.0)
    labels = torch.cat((labels, stay_value.clamp_max(1.0)[:, None]), dim=1)
    masks = torch.cat((masks, feed_exposure[:, None]), dim=1)
    applicable = torch.cat((applicable, feed_exposure[:, None]), dim=1)
    mature = torch.cat((mature, feed_exposure[:, None]), dim=1)
    return ProbeBatch(
        request_id=scalar("request_id", torch.long),
        user_id=scalar("user_id", torch.long),
        surface=surface,
        request_time=scalar("request_time", torch.long),
        item_id=scalar("item_id", torch.long),
        position=scalar("position", torch.long),
        route_id=scalar("route_id", torch.long),
        recall_score=scalar("recall_score", torch.float32),
        exposed=exposed,
        candidate_exposure_probability=scalar(
            "candidate_exposure_probability", torch.float32,
        ),
        randomized_support=scalar("randomized_support", torch.bool),
        dwell_ms=dwell_ms,
        dense_features=dense,
        sparse_buckets=sparse,
        labels=labels,
        label_mask=masks,
        label_applicable=applicable,
        label_mature=mature,
        joint_logging_probability=scalar(
            "joint_logging_probability", torch.float32,
        ),
        task_names=tuple(sample["task_names"]) + ("stay_value",),
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
    dense_mean: torch.Tensor
    dense_scale: torch.Tensor
    serving_task_weights: tuple[float, ...] | None = None
    task_logit_offsets: torch.Tensor | None = None
    conditional_task_parents: tuple[int, ...] | None = None

    @property
    def model_name(self) -> str:
        return "v4-lr-infrastructure-probe"

    def validate_compatibility(self, expected) -> None:
        if self.feature_manifest_hash != expected.feature_manifest_hash:
            raise ValueError("probe artifact feature manifest differs")
        if self.feature_manifest_hash != expected.stage_contract_hash:
            raise ValueError("probe artifact stage contract differs")

    @torch.inference_mode()
    def predict_task_probabilities(
        self, dense: torch.Tensor, surface: torch.Tensor,
    ) -> torch.Tensor:
        parameter = next(self.model.parameters())
        dense = dense.to(parameter.device, parameter.dtype)
        surface = surface.to(parameter.device)
        mean = self.dense_mean.to(parameter.device, parameter.dtype)
        scale = self.dense_scale.to(parameter.device, parameter.dtype)
        logits = self.model((dense - mean) / scale, surface)
        offset = (
            torch.zeros(logits.shape[1], device=logits.device)
            if self.task_logit_offsets is None
            else self.task_logit_offsets.to(logits.device, logits.dtype)
        )
        raw_probability = torch.sigmoid(logits - offset)
        probability = raw_probability.clone()
        parents = self.conditional_task_parents or (-1,) * len(self.task_names)
        for child, parent in enumerate(parents):
            if parent >= 0:
                probability[:, child] = (
                    probability[:, parent] * raw_probability[:, child]
                )
        return probability

    @torch.inference_mode()
    def score(
        self,
        features: FeatureTensorBatch,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        if features.manifest_hash != self.feature_manifest_hash:
            raise ValueError("probe scoring feature manifest differs")
        requests, candidates = features.dense.shape[:2]
        dense = features.dense.reshape(-1, features.dense.shape[2])
        expanded_surface = surface[:, None].expand(
            requests, candidates,
        ).reshape(-1)
        probability = self.predict_task_probabilities(dense, expanded_surface)
        if self.serving_task_weights is None:
            score = probability[:, self.task_names.index("long_view")]
        else:
            weight = torch.tensor(
                self.serving_task_weights,
                device=probability.device,
                dtype=probability.dtype,
            )
            score = (probability * weight).sum(dim=1).clamp_min(0.0)
        score = score.reshape(requests, candidates)
        return score.to(features.dense.device)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema": "v4-lr-infrastructure-probe-v4",
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
            "dense_mean": self.dense_mean.detach().cpu().clone(),
            "dense_scale": self.dense_scale.detach().cpu().clone(),
            "serving_task_weights": self.serving_task_weights,
            "task_logit_offsets": (
                None if self.task_logit_offsets is None
                else self.task_logit_offsets.detach().cpu().clone()
            ),
            "conditional_task_parents": self.conditional_task_parents,
        }

    @classmethod
    def from_checkpoint(cls, value: dict[str, object]) -> ProbeArtifact:
        schema = value.get("schema")
        if schema not in {
            "v4-lr-infrastructure-probe-v1",
            "v4-lr-infrastructure-probe-v2",
            "v4-lr-infrastructure-probe-v3",
            "v4-lr-infrastructure-probe-v4",
        }:
            raise ValueError("probe checkpoint schema is unsupported")
        model = ProbeRanker(
            int(value["inputs"]), int(value["tasks"]), int(value["surfaces"]),
        )
        model.load_state_dict(value["state_dict"])
        model.eval()
        dense_mean = value.get("dense_mean")
        dense_scale = value.get("dense_scale")
        if schema == "v4-lr-infrastructure-probe-v1":
            dense_mean = torch.zeros(model.inputs)
            dense_scale = torch.ones(model.inputs)
        return cls(
            model,
            tuple(value["task_names"]),
            str(value["feature_manifest_hash"]),
            dict(value["training_report"]),
            dense_mean.detach().cpu().float(),
            dense_scale.detach().cpu().float(),
            (
                None
                if value.get("serving_task_weights") is None
                else tuple(value["serving_task_weights"])
            ),
            (
                None
                if value.get("task_logit_offsets") is None
                else value["task_logit_offsets"].detach().cpu().float()
            ),
            (
                None
                if value.get("conditional_task_parents") is None
                else tuple(int(parent) for parent in value["conditional_task_parents"])
            ),
        )


def train_probe(
    batch: ProbeBatch,
    *,
    lane: Lane,
    initial_artifact: ProbeArtifact | None = None,
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
    dense = batch.dense_features.to(target)
    dense_mean = dense.mean(dim=0)
    dense_scale = dense.std(dim=0, unbiased=False).clamp_min(1e-4)
    model = ProbeRanker(
        batch.dense_features.shape[1], len(batch.task_names),
    ).to(target)
    conditional_names = {
        "play_3s": "play",
        "long_view": "play_3s",
        "complete": "long_view",
    }
    conditional_parents = tuple(
        batch.task_names.index(conditional_names[name])
        if name in conditional_names and conditional_names[name] in batch.task_names
        else -1
        for name in batch.task_names
    )
    conditional_children = {
        index for index, parent in enumerate(conditional_parents) if parent >= 0
    }
    warm_started_tasks: list[str] = []
    if initial_artifact is not None:
        if initial_artifact.feature_manifest_hash != batch.feature_manifest_hash:
            raise ValueError("warm-start artifact feature manifest differs")
        if initial_artifact.model.inputs != model.inputs:
            raise ValueError("warm-start artifact input shape differs")
        with torch.no_grad():
            for task in set(batch.task_names) & set(initial_artifact.task_names):
                source = initial_artifact.task_names.index(task)
                target_task = batch.task_names.index(task)
                if target_task in conditional_children:
                    continue
                old_weight = initial_artifact.model.linear.weight[source].to(target)
                old_mean = initial_artifact.dense_mean.to(target)
                old_scale = initial_artifact.dense_scale.to(target)
                adjusted_weight = old_weight.clone()
                adjusted_weight[:model.inputs] *= dense_scale / old_scale
                model.linear.weight[target_task].copy_(adjusted_weight)
                model.linear.bias[target_task].copy_(
                    initial_artifact.model.linear.bias[source].to(target)
                    + (old_weight[:model.inputs] * (
                        dense_mean - old_mean
                    ) / old_scale).sum(),
                )
                warm_started_tasks.append(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dense = (dense - dense_mean) / dense_scale
    surface = batch.surface.to(target)
    labels = batch.labels.to(target)
    masks = batch.label_mask.to(target)
    masks = masks.clone()
    for child, parent in enumerate(conditional_parents):
        if parent >= 0:
            masks[:, child] &= labels[:, parent] > 0.5
    propensity = batch.joint_logging_probability.to(target)
    if not masks.any():
        raise ValueError("probe has no mature labels")
    positive = (labels * masks).sum(dim=0)
    observed = masks.sum(dim=0)
    positive_weight = (
        (observed - positive) / positive.clamp_min(1.0)
    ).clamp(1.0, 50.0)
    if "stay_value" in batch.task_names:
        positive_weight[batch.task_names.index("stay_value")] = 1.0
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
                logits,
                labels[row],
                reduction="none",
                pos_weight=positive_weight,
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
            "warm_started_tasks": sorted(warm_started_tasks),
            "new_tasks": sorted(set(batch.task_names) - set(warm_started_tasks)),
            "conditional_task_parents": {
                batch.task_names[child]: batch.task_names[parent]
                for child, parent in enumerate(conditional_parents)
                if parent >= 0
            },
            "positive_weight": positive_weight.detach().cpu().tolist(),
            "device": str(target),
            "seed": seed,
        },
        dense_mean=dense_mean.detach().cpu(),
        dense_scale=dense_scale.detach().cpu(),
        task_logit_offsets=torch.log(positive_weight).detach().cpu(),
        conditional_task_parents=conditional_parents,
    )
