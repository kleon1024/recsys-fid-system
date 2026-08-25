from __future__ import annotations

from fid_lab.simulation.digital_twin import AppEventBatch, RequestCandidateTrace
from fid_lab.simulation.digital_twin.evaluation import aa_decision, factual_ab_report
from fid_lab.simulation.digital_twin.observability import (
    FullFlowFixtureConfig,
    build_full_flow_fixtures,
)


def test_factual_ab_uses_request_assignment_and_user_clusters():
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=1_024,
        items=5_000,
        scenario="feed_consumption",
    ), ticks=4)
    trace = RequestCandidateTrace.concatenate(tuple(
        snapshot.trace for snapshot in snapshots
    ))
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
