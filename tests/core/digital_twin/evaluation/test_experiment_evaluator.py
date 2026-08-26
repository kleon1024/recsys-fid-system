from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

import torch
import pytest

from fid_lab.simulation.digital_twin import AppEventBatch, EventType
from fid_lab.simulation.digital_twin.contracts import (
    PublishFailureReason,
    Surface,
    make_app_events,
)
from fid_lab.simulation.digital_twin.evaluation import (
    FactualABAccumulator,
    aa_decision,
    factual_ab_report,
)
from fid_lab.simulation.digital_twin.observability import (
    FullFlowFixtureConfig,
    build_full_flow_fixtures,
)
from fid_lab.simulation.digital_twin.experiments.launch_review.metrics import (
    StreamingExperimentMetrics,
    analyze_experiment,
)
from fid_lab.simulation.digital_twin.experiments.ranking.publish_queue_canary import (
    PublishQueueCanaryConfig,
)


def test_factual_ab_uses_request_assignment_and_user_clusters():
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=1_024,
        items=5_000,
        scenario="feed_consumption",
    ), ticks=4)
    trace = tuple(snapshot.trace for snapshot in snapshots)
    events = AppEventBatch.concatenate(tuple(
        snapshot.events for snapshot in snapshots
    ))

    report = factual_ab_report(
        trace,
        events,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    decision = aa_decision(report)

    assert report["control_users"] > 0
    assert report["treatment_users"] > 0
    assert report["cross_cell_users"] == 0
    assert report["srm_p_value"] is not None
    assert report["metrics"]["dwell_seconds"]["status"] == "estimated"
    assert decision["decision"] in {"pass", "hold"}

    accumulator = FactualABAccumulator(
        1_024, control_fraction=0.2, treatment_fraction=0.2,
    )
    for snapshot in snapshots:
        accumulator.update(snapshot.trace, snapshot.events)
    streamed = accumulator.report()
    assert streamed == report

    foreign = replace(
        events,
        event_id=events.event_id + int(events.event_id.max()) + 1,
        surface=torch.full_like(events.surface, int(Surface.SEARCH)),
        duration_ms=torch.full_like(events.duration_ms, 9_999_000),
    )
    contaminated = factual_ab_report(
        trace,
        AppEventBatch.concatenate((events, foreign)),
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    assert contaminated == report

    imbalanced = deepcopy(report)
    imbalanced["metrics"]["share"].update({
        "p_value": 0.0005,
        "ci95_low": 0.01,
        "ci95_high": 0.03,
    })
    assert aa_decision(imbalanced)["decision"] == "hold"


def test_launch_metrics_join_later_posting_events_by_feed_user_cohort():
    users = torch.tensor([0, 1, 2, 3])
    impressions = make_app_events(
        EventType.IMPRESSION,
        event_time=0,
        request_id=torch.tensor([10, 11, 12, 13]),
        user_id=users,
        surface=torch.full((4,), int(Surface.FEED)),
        experiment_cell=torch.tensor([0, 0, 1, 1]),
    )
    publishing_users = torch.tensor([0, 1, 2, 2, 3, 3])
    publications = make_app_events(
        EventType.PUBLISH,
        event_time=2,
        request_id=torch.arange(20, 26),
        user_id=publishing_users,
        surface=torch.full((6,), int(Surface.POSTING)),
        experiment_cell=torch.full((6,), -1),
    )
    metrics, sample = analyze_experiment(
        AppEventBatch.concatenate((impressions, publications)), users=4,
    )
    assert sample == {
        "control_triggered_users": 2,
        "treatment_triggered_users": 2,
    }
    assert metrics["publish"]["control_mean"] == 1.0
    assert metrics["publish"]["treatment_mean"] == 2.0

    streamed = StreamingExperimentMetrics(users=4, device="cpu")
    streamed.append(impressions)
    streamed.append(publications)
    streamed_metrics, streamed_sample = streamed.analyze()
    assert streamed_sample == sample
    assert streamed_metrics == metrics


def test_expanded_canary_preserves_standard_supply_ratio():
    with pytest.raises(ValueError, match="item/user ratio"):
        PublishQueueCanaryConfig(
            publish_checkpoint="publish.pt",
            control_fine_checkpoint="fine.pt",
            output="out",
            users=100_000,
            items=100_000,
        )


def test_expanded_canary_rejects_unsafe_memory_guards():
    with pytest.raises(ValueError, match="CUDA memory fraction"):
        PublishQueueCanaryConfig(
            publish_checkpoint="publish.pt",
            control_fine_checkpoint="fine.pt",
            output="out",
            cuda_memory_fraction=1.1,
        )


def test_publish_followup_freezes_cohort_and_only_adds_delayed_outcomes():
    metrics = StreamingExperimentMetrics(users=5, device="cpu")
    impressions = make_app_events(
        EventType.IMPRESSION,
        event_time=0,
        request_id=torch.tensor([10, 11, 12, 13]),
        user_id=torch.tensor([0, 1, 2, 3]),
        surface=torch.full((4,), int(Surface.FEED)),
        experiment_cell=torch.tensor([0, 0, 1, 1]),
    )
    dwell = make_app_events(
        EventType.DWELL,
        event_time=0,
        request_id=torch.tensor([10, 11, 12, 13]),
        user_id=torch.tensor([0, 1, 2, 3]),
        surface=torch.full((4,), int(Surface.FEED)),
        experiment_cell=torch.tensor([0, 0, 1, 1]),
        duration_ms=torch.tensor([1_000, 1_000, 2_000, 2_000]),
    )
    metrics.append(impressions)
    metrics.append(dwell)
    frozen_sample = metrics.freeze_cohort()

    followup_impression = make_app_events(
        EventType.IMPRESSION,
        event_time=1,
        request_id=torch.tensor([14]),
        user_id=torch.tensor([4]),
        surface=torch.tensor([int(Surface.FEED)]),
        experiment_cell=torch.tensor([1]),
    )
    followup_dwell = make_app_events(
        EventType.DWELL,
        event_time=1,
        request_id=torch.tensor([14]),
        user_id=torch.tensor([4]),
        surface=torch.tensor([int(Surface.FEED)]),
        experiment_cell=torch.tensor([1]),
        duration_ms=torch.tensor([9_000]),
    )
    followup_publish = make_app_events(
        EventType.PUBLISH,
        event_time=2,
        request_id=torch.tensor([20, 21]),
        user_id=torch.tensor([0, 4]),
        surface=torch.full((2,), int(Surface.POSTING)),
        experiment_cell=torch.full((2,), -1),
    )
    for events in (followup_impression, followup_dwell, followup_publish):
        metrics.append(events, cross_request_only=True)

    estimates, sample = metrics.analyze()
    assert sample == frozen_sample == {
        "control_triggered_users": 2,
        "treatment_triggered_users": 2,
    }
    assert estimates["dwell_seconds"]["control_mean"] == 1.0
    assert estimates["dwell_seconds"]["treatment_mean"] == 2.0
    assert estimates["publish"]["control_mean"] == 0.5
    assert estimates["publish"]["treatment_mean"] == 0.0


def test_publish_followup_uses_label_maturity_window():
    config = PublishQueueCanaryConfig(
        publish_checkpoint="publish.pt",
        control_fine_checkpoint="fine.pt",
        output="out",
        ticks_per_day=16,
    )
    assert config.required_followup_steps == 32
    assert config.resolved_followup_steps == 32
    with pytest.raises(ValueError, match="shorter than maturity"):
        replace(config, followup_steps=31)


def test_publish_diagnostics_expose_supply_capacity_failure():
    metrics = StreamingExperimentMetrics(users=2, device="cpu")
    failed = make_app_events(
        EventType.PUBLISH_FAILED,
        event_time=1,
        request_id=torch.tensor([30]),
        user_id=torch.tensor([0]),
        surface=torch.tensor([int(Surface.POSTING)]),
        value=torch.tensor([int(PublishFailureReason.NO_CAPACITY)]),
    )
    metrics.append(failed)
    diagnostics = metrics.diagnostics()
    assert diagnostics["funnel_event_counts"]["publish_failed"] == 1
    assert diagnostics["publish_failure_reasons"]["no_capacity"] == 1
