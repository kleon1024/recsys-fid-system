from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.evaluation import (
    evaluate_request_batch,
    support_report,
)
from fid_lab.simulation.digital_twin.observability import (
    FullFlowFixtureConfig,
    build_full_flow_fixture,
)


def test_deterministic_trace_does_not_invent_randomized_support():
    snapshot = build_full_flow_fixture(FullFlowFixtureConfig(
        users=64,
        items=800,
        scenario="feed_consumption",
    ))

    report = evaluate_request_batch(snapshot.trace, snapshot.samples.fine)

    assert report["stage"]["duplicate_request_ids"] == 0
    assert report["support"]["randomized_requests"] == 0
    assert report["support"]["randomized_supported_candidate_rows"] == 0
    assert report["support"]["factual_action_support_complete"]


def test_unsupported_challenger_is_reported_instead_of_scored():
    snapshot = build_full_flow_fixture(FullFlowFixtureConfig(
        users=64,
        items=800,
        scenario="feed_consumption",
    ))
    challenger = snapshot.trace.recall_item_id[:, -1:]

    report = support_report(snapshot.trace, challenger)

    assert report["challenger_rows"] > 0
    assert not report["candidate_ope_identified"]
    assert not report["slate_ope_identified"]


def test_request_metrics_keep_task_masks_and_grouping_explicit():
    snapshot = build_full_flow_fixture(FullFlowFixtureConfig(
        users=128,
        items=1_200,
        scenario="feed_consumption",
    ))
    labels = snapshot.samples.fine.labels
    probability = torch.full_like(labels, 0.2)

    report = evaluate_request_batch(
        snapshot.trace,
        snapshot.samples.fine,
        rank_scores=probability,
        probabilities=probability,
    )

    assert report["grouping"] == "request_id"
    assert report["tasks"]["play"]["rows"] > 0
    assert "request_gauc" in report["tasks"]["play"]
