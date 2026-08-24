"""Materialize observed trace events and point-in-time training authorities."""

from __future__ import annotations

import torch

from ..exchange import TASKS, task_applicability
from ..serving.trace import RequestTrace
from .contracts import (
    TASK_MATURITY_STEPS,
    CoarseRankExampleBatch,
    FineRankExampleBatch,
    RecallExampleBatch,
    SampleManifest,
    TrainingAuthorities,
    TwinEventBatch,
)


_POSITIVE_WEIGHT = {
    "play": 0.1, "play_3s": 0.2, "long_view": 0.8, "complete": 1.0,
    "like": 1.0, "comment": 1.2, "share": 1.5, "follow": 1.8,
    "click": 0.5, "detail": 0.8, "favorite": 1.2, "add_cart": 1.5,
    "order": 2.0, "payment": 2.5, "negative": -1.0, "create": 1.2,
    "publish": 2.0,
}


def materialize_events(
    trace: RequestTrace,
    *,
    world_version: str,
    served_policy: str,
    experiment_cell: str,
    watermark_step: int,
) -> TwinEventBatch:
    gates = trace.validate()
    if not all(gates.values()):
        raise ValueError(f"request trace closure failed: {gates}")
    values = trace.tensors()
    active = values["active"].bool()
    manifest = SampleManifest(
        world_version=world_version,
        served_policy=served_policy,
        experiment_cell=experiment_cell,
        watermark_step=watermark_step,
    )
    return TwinEventBatch(
        request_id=values["request_id"][active],
        user_id=values["user_id"][active],
        step=values["step"][active],
        surface=values["surface"][active],
        request_sampling_probability=values[
            "request_sampling_probability"
        ][active],
        served_policy_id=values["served_policy_id"][active],
        experiment_cell_id=values["experiment_cell_id"][active],
        candidate_item_ids=values["recalled_item_ids"][active],
        candidate_kind=values["candidate_kind"][active],
        route=values["route"][active],
        candidate_features=values["candidate_features"][active],
        recall_score=values["recall_score"][active],
        coarse_score=values["coarse_score"][active],
        fine_score=values["fine_score"][active],
        eligible=values["eligible"][active],
        exposed_item_ids=values["exposed_item_ids"][active],
        exposed_propensity=values["exposed_propensity"][active].float(),
        selected_item_id=values["selected_item"][active],
        labels=values["labels"][active].float(),
        label_mask=values["label_mask"][active].bool(),
        history_item_ids=values["history_item"][active],
        history_steps=values["history_step"][active],
        manifest=manifest,
    )


def _mature_mask(events: TwinEventBatch) -> torch.Tensor:
    maturity = torch.tensor(
        [TASK_MATURITY_STEPS[task] for task in TASKS],
        device=events.step.device,
    )
    matured = events.step[:, None] + maturity[None] <= events.manifest.watermark_step
    return events.label_mask & matured


def _behavior_strength(events: TwinEventBatch, mature: torch.Tensor) -> torch.Tensor:
    weight = torch.tensor(
        [_POSITIVE_WEIGHT[task] for task in TASKS],
        dtype=events.labels.dtype,
        device=events.labels.device,
    )
    return (events.labels * mature.float() * weight[None]).sum(dim=1)


def _candidate_positions(events: TwinEventBatch) -> tuple[torch.Tensor, torch.Tensor]:
    match = (
        events.exposed_item_ids[:, :, None]
        == events.candidate_item_ids[:, None, :]
    )
    valid = events.exposed_item_ids >= 0
    if not (match.any(dim=2) | ~valid).all():
        raise ValueError("exposed item missing from recall candidate set")
    return match.float().argmax(dim=2), valid


def join_training_authorities(events: TwinEventBatch) -> TrainingAuthorities:
    mature = _mature_mask(events)
    strength = _behavior_strength(events, mature)
    recall_keep = strength > 0
    selected_match = (
        events.candidate_item_ids == events.selected_item_id[:, None]
    )
    sample_probability = torch.ones_like(events.recall_score)
    recall = RecallExampleBatch(
        request_id=events.request_id[recall_keep],
        user_id=events.user_id[recall_keep],
        surface=events.surface[recall_keep],
        served_policy_id=events.served_policy_id[recall_keep],
        experiment_cell_id=events.experiment_cell_id[recall_keep],
        request_sampling_probability=events.request_sampling_probability[
            recall_keep
        ],
        positive_item_id=events.selected_item_id[recall_keep],
        candidate_item_ids=events.candidate_item_ids[recall_keep],
        route=events.route[recall_keep],
        negative_mask=(~selected_match)[recall_keep],
        sampling_probability=sample_probability[recall_keep],
        behavior_strength=strength[recall_keep],
        manifest=events.manifest,
    )
    exposed = (
        events.candidate_item_ids[:, :, None]
        == events.exposed_item_ids[:, None, :]
    ).any(dim=2)
    relevance = selected_match.float() * strength.clamp_min(0.0)[:, None]
    coarse = CoarseRankExampleBatch(
        request_id=events.request_id,
        user_id=events.user_id,
        surface=events.surface,
        served_policy_id=events.served_policy_id,
        experiment_cell_id=events.experiment_cell_id,
        request_sampling_probability=events.request_sampling_probability,
        item_ids=events.candidate_item_ids,
        route=events.route,
        features=events.candidate_features,
        recall_score=events.recall_score,
        served_coarse_score=events.coarse_score,
        teacher_fine_score=events.fine_score,
        eligible=events.eligible,
        exposed=exposed,
        relevance=relevance,
        sampling_probability=sample_probability,
        manifest=events.manifest,
    )
    candidate_position, exposed_valid = _candidate_positions(events)
    gather_feature = candidate_position[:, :, None].expand(
        -1, -1, events.candidate_features.shape[2]
    )
    fine_features = events.candidate_features.gather(1, gather_feature)
    fine_scores = events.fine_score.gather(1, candidate_position)
    selected = events.exposed_item_ids == events.selected_item_id[:, None]
    label = events.labels[:, None, :] * selected[:, :, None]
    exposed_kind = events.candidate_kind.gather(1, candidate_position)
    task_defined = task_applicability(
        events.surface[:, None].expand_as(exposed_kind).reshape(-1),
        exposed_kind.reshape(-1),
    ).reshape(*exposed_kind.shape, len(TASKS))
    label_mask = (
        mature[:, None, :] & exposed_valid[:, :, None] & task_defined
    )
    positions = torch.arange(
        events.exposed_item_ids.shape[1], device=events.step.device
    )[None].expand_as(events.exposed_item_ids)
    position_propensity = 1.0 / torch.log2(positions.float() + 2.0)
    propensity = events.exposed_propensity * position_propensity
    fine = FineRankExampleBatch(
        request_id=events.request_id,
        user_id=events.user_id,
        step=events.step,
        surface=events.surface,
        served_policy_id=events.served_policy_id,
        experiment_cell_id=events.experiment_cell_id,
        request_sampling_probability=events.request_sampling_probability,
        item_ids=events.exposed_item_ids,
        positions=positions,
        examination_propensity=propensity * exposed_valid.float(),
        features=fine_features,
        served_score=fine_scores,
        labels=label,
        label_mask=label_mask,
        selected=selected,
        manifest=events.manifest,
    )
    return TrainingAuthorities(recall=recall, coarse=coarse, fine=fine)
