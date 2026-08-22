"""Event-time Joiner for ranking samples and closed/open-loop conversion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log

from .contracts import (
    BEHAVIOR_STRENGTH,
    TASK_WINDOWS_SECONDS,
    ActionEvent,
    CoarseRankExample,
    CommerceEvent,
    FineRankExample,
    OutboundClick,
    PixelEvent,
    RecallExample,
    StageDecision,
)
from .sampling import mixed_negative_sample


@dataclass(frozen=True)
class AttributionReport:
    matched_conversions: int
    duplicate_pixels: int
    orphan_pixels: int
    missing_identity: int
    late_pixels: int
    attributed_weight: float


@dataclass(frozen=True)
class JoinerReport:
    recall: tuple[RecallExample, ...]
    coarse: tuple[CoarseRankExample, ...]
    fine: tuple[FineRankExample, ...]
    duplicate_events: int
    orphan_events: int
    immature_task_labels: int
    attribution: AttributionReport


class EvolutionJoiner:
    def __init__(
        self,
        allowed_lateness_seconds: int = 300,
        pixel_half_life_seconds: int = 86_400,
    ) -> None:
        self.allowed_lateness_seconds = allowed_lateness_seconds
        self.pixel_half_life_seconds = pixel_half_life_seconds

    @staticmethod
    def _deduplicate(events, identity):
        unique = {}
        duplicates = 0
        for event in events:
            key = identity(event)
            duplicates += int(key in unique)
            unique.setdefault(key, event)
        return tuple(unique.values()), duplicates

    def _pixel_attribution(
        self,
        clicks: tuple[OutboundClick, ...],
        pixels: tuple[PixelEvent, ...],
        watermark: int,
    ) -> tuple[dict[tuple[str, int, int], float], AttributionReport]:
        clicks_by_id = {click.click_id: click for click in clicks}
        by_identity: dict[tuple[str, int], list[OutboundClick]] = defaultdict(list)
        for click in clicks:
            if click.identity is not None:
                by_identity[(click.identity, click.merchant_id)].append(click)
        weights: dict[tuple[str, int, int], float] = defaultdict(float)
        matched = orphan = missing = late = 0
        for pixel in pixels:
            if pixel.received_at > watermark:
                late += 1
                continue
            eligible: list[OutboundClick] = []
            if pixel.click_id and pixel.click_id in clicks_by_id:
                click = clicks_by_id[pixel.click_id]
                if (
                    click.merchant_id == pixel.merchant_id
                    and 0 <= pixel.event_time - click.event_time
                    <= TASK_WINDOWS_SECONDS["pixel_conversion"]
                ):
                    eligible = [click]
            elif pixel.identity is None:
                missing += 1
            else:
                eligible = [
                    click
                    for click in by_identity[(pixel.identity, pixel.merchant_id)]
                    if 0 <= pixel.event_time - click.event_time
                    <= TASK_WINDOWS_SECONDS["pixel_conversion"]
                ]
            if not eligible:
                orphan += 1
                continue
            decay = [
                exp(-log(2.0) * (pixel.event_time - click.event_time) / self.pixel_half_life_seconds)
                for click in eligible
            ]
            denominator = sum(decay)
            for click, value in zip(eligible, decay):
                weights[click.key] += value / denominator
            matched += 1
        return weights, AttributionReport(
            matched,
            0,
            orphan,
            missing,
            late,
            float(sum(weights.values())),
        )

    def _labels(
        self,
        decision: StageDecision,
        events: list[tuple[str, int, int]],
        pixel_weight: float,
        watermark: int,
    ) -> tuple[dict[str, float], dict[str, bool], int]:
        labels: dict[str, float] = {}
        masks: dict[str, bool] = {}
        immature = 0
        for task, window in TASK_WINDOWS_SECONDS.items():
            mature = watermark >= decision.impression_time + window + self.allowed_lateness_seconds
            if task == "pixel_conversion" and not decision.pixel_observable:
                mature = False
            masks[task] = mature
            immature += int(not mature)
            labels[task] = float(
                sum(
                    1.0
                    for action, event_time, received_at in events
                    if action == task
                    and decision.impression_time <= event_time <= decision.impression_time + window
                    and received_at <= event_time + self.allowed_lateness_seconds
                )
            )
        labels["pixel_conversion"] = pixel_weight
        return labels, masks, immature

    def build(
        self,
        decisions: list[StageDecision],
        actions: list[ActionEvent],
        commerce: list[CommerceEvent],
        clicks: list[OutboundClick],
        pixels: list[PixelEvent],
        watermark: int,
    ) -> JoinerReport:
        decision_by_key = {decision.key: decision for decision in decisions}
        if len(decision_by_key) != len(decisions):
            raise ValueError("stage decision keys must be unique")
        actions_unique, action_duplicates = self._deduplicate(actions, lambda value: value.event_id)
        commerce_unique, commerce_duplicates = self._deduplicate(
            commerce, lambda value: value.event_id
        )
        clicks_unique, click_duplicates = self._deduplicate(clicks, lambda value: value.click_id)
        pixels_unique, pixel_duplicates = self._deduplicate(pixels, lambda value: value.event_id)
        pixel_weights, attribution = self._pixel_attribution(
            clicks_unique, pixels_unique, watermark
        )
        attribution = AttributionReport(
            attribution.matched_conversions,
            pixel_duplicates,
            attribution.orphan_pixels,
            attribution.missing_identity,
            attribution.late_pixels,
            attribution.attributed_weight,
        )
        by_key: dict[tuple[str, int, int], list[tuple[str, int, int]]] = defaultdict(list)
        orphan = 0
        for event in (*actions_unique, *commerce_unique):
            if event.key not in decision_by_key:
                orphan += 1
                continue
            by_key[event.key].append((event.action, event.event_time, event.received_at))
        coarse: list[CoarseRankExample] = []
        fine: list[FineRankExample] = []
        recall: list[RecallExample] = []
        immature = 0
        all_item_ids = tuple(decision.video_id for decision in decisions)
        for index, decision in enumerate(decisions):
            labels, masks, immature_count = self._labels(
                decision,
                by_key[decision.key],
                pixel_weights.get(decision.key, 0.0),
                watermark,
            )
            immature += immature_count
            coarse.append(
                CoarseRankExample(
                    decision.key,
                    decision.feature_fids,
                    decision.dense_features,
                    labels,
                    masks if decision.exposed else {task: False for task in masks},
                    decision.teacher_score,
                    decision.teacher_rank,
                    decision.recall_route,
                    decision.sampling_probability,
                    decision.served_scores,
                    decision.manifest,
                )
            )
            if not decision.exposed:
                continue
            fine.append(
                FineRankExample(
                    decision.key,
                    decision.viewer_id,
                    decision.author_id,
                    decision.feature_fids,
                    decision.dense_features,
                    decision.sequence,
                    labels,
                    masks,
                    min(1.0 / max(decision.sampling_probability, 1e-6), 20.0),
                    decision.served_scores,
                    decision.manifest,
                )
            )
            positive_tasks = [
                task for task, label in labels.items() if masks[task] and label > 0.0
            ]
            if positive_tasks:
                positive = max(BEHAVIOR_STRENGTH[task] for task in positive_tasks)
                hard = tuple(
                    candidate.video_id
                    for candidate in decisions
                    if candidate.category_id == decision.category_id
                    and candidate.video_id != decision.video_id
                )
                random = tuple(item for item in all_item_ids if item != decision.video_id)
                recall.append(
                    RecallExample(
                        decision.request_id,
                        decision.viewer_id,
                        decision.video_id,
                        positive,
                        mixed_negative_sample(random, hard or random, random, 20, index + 71),
                        decision.manifest,
                    )
                )
        return JoinerReport(
            tuple(recall),
            tuple(coarse),
            tuple(fine),
            action_duplicates + commerce_duplicates + click_duplicates + pixel_duplicates,
            orphan + attribution.orphan_pixels,
            immature,
            attribution,
        )
