"""Point-in-time correct impression/action Joiner with delayed-label handling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .contracts import ActionEvent, ImpressionEvent, TASKS, TrainingExample


@dataclass(frozen=True)
class JoinerConfig:
    label_window_seconds: int = 300
    allowed_lateness_seconds: int = 60
    inverse_propensity_clip: float = 10.0
    version: str = "joiner-v1"


@dataclass(frozen=True)
class JoinReport:
    examples: tuple[TrainingExample, ...]
    immature_impressions: int
    duplicate_actions: int
    ignored_actions: int


class ExampleJoiner:
    def __init__(self, config: JoinerConfig = JoinerConfig()) -> None:
        self.config = config

    def _close_time(self, impression: ImpressionEvent) -> int:
        return (
            impression.event_time
            + self.config.label_window_seconds
            + self.config.allowed_lateness_seconds
        )

    def _valid_action(
        self, impression: ImpressionEvent, action: ActionEvent, watermark: int
    ) -> bool:
        event_deadline = impression.event_time + self.config.label_window_seconds
        receive_deadline = event_deadline + self.config.allowed_lateness_seconds
        return (
            impression.event_time <= action.event_time <= event_deadline
            and action.received_at <= min(watermark, receive_deadline)
            and action.action in TASKS
        )

    def build(
        self,
        impressions: list[ImpressionEvent],
        actions: list[ActionEvent],
        watermark: int,
    ) -> JoinReport:
        impression_by_key = {impression.key: impression for impression in impressions}
        if len(impression_by_key) != len(impressions):
            raise ValueError("impression keys must be unique")
        action_by_id: dict[str, ActionEvent] = {}
        duplicate_actions = 0
        for action in actions:
            duplicate_actions += int(action.event_id in action_by_id)
            action_by_id.setdefault(action.event_id, action)
        actions_by_key: dict[tuple[str, int], list[ActionEvent]] = defaultdict(list)
        ignored_actions = 0
        for action in action_by_id.values():
            if action.key not in impression_by_key:
                ignored_actions += 1
                continue
            actions_by_key[action.key].append(action)
        examples: list[TrainingExample] = []
        immature = 0
        for impression in sorted(impressions, key=lambda value: value.event_time):
            if self._close_time(impression) > watermark:
                immature += 1
                continue
            valid = [
                action
                for action in actions_by_key[impression.key]
                if self._valid_action(impression, action, watermark)
            ]
            ignored_actions += len(actions_by_key[impression.key]) - len(valid)
            labels = {
                task: float(any(action.action == task and action.value > 0 for action in valid))
                for task in TASKS
            }
            weight = min(
                1.0 / max(impression.propensity, 1e-6),
                self.config.inverse_propensity_clip,
            )
            examples.append(
                TrainingExample(
                    example_id=f"{impression.request_id}:{impression.item_id}",
                    user_id=impression.user_id,
                    item_id=impression.item_id,
                    impression_time=impression.event_time,
                    feature_fids=impression.feature_fids,
                    feature_buckets=impression.feature_buckets,
                    labels=labels,
                    sample_weight=weight,
                    schema_version=impression.schema_version,
                )
            )
        return JoinReport(tuple(examples), immature, duplicate_actions, ignored_actions)
