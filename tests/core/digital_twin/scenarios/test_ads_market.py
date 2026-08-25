from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.contracts import ContentKind, EventType
from fid_lab.simulation.digital_twin.engine import ExperimentPlan
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
)
from fid_lab.simulation.digital_twin.platform import CascadePolicy
from fid_lab.simulation.digital_twin.platform.routes import (
    DEFAULT_BUSINESS_ROUTE_NAMES,
)
from fid_lab.simulation.digital_twin.scenarios.ads import audit_ads_market
from fid_lab.simulation.digital_twin.scenarios.ads.launch import (
    AdsLaunchConfig,
    run_ads_launch,
)


def _plan() -> ExperimentPlan:
    policy = CascadePolicy(
        "ads-market",
        1,
        1,
        1,
        enabled_routes=("random", "popular"),
        enabled_business_routes=(
            *DEFAULT_BUSINESS_ROUTE_NAMES, "ads_auction",
        ),
    )
    return ExperimentPlan.ramped_user_ab(
        active_policy=policy,
        treatment_policy=policy,
        experiment_seed=809,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )


def _kernel():
    return _build_kernel(RetrievalLadderConfig(
        users=256,
        items=3_000,
        device="cpu",
        ticks_per_day=8,
    ))[1]


def _factual_checkpoint(tmp_path):
    kernel = _kernel()
    policy = CascadePolicy(
        "organic-baseline",
        1,
        1,
        1,
        enabled_routes=("random", "popular"),
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=policy,
        treatment_policy=policy,
        experiment_seed=809,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    kernel.step(0, plan)
    store = WorldCheckpointStore(tmp_path / "checkpoints")
    checkpoint = store.save(kernel, 0, plan)
    WorldBranchRegistry(store).initialize_main(checkpoint.checkpoint_id)


def test_ads_market_reconciles_budget_spend_and_pixel_lineage():
    kernel = _kernel()
    for logical_time in range(12):
        kernel.step(logical_time, _plan())
    audit = audit_ads_market(kernel.event_log.read())
    assert audit.impressions == audit.spend_events > 0
    assert audit.clicks > audit.pixel_conversions > 0
    assert audit.billed_revenue > 0.0
    assert audit.maximum_ads_per_request == 1
    assert audit.unbudgeted_spend == 0
    assert audit.unpriced_spend == 0
    assert audit.over_bid_spend == 0
    assert audit.overspend_events == 0
    assert audit.partially_billed_impressions == 0


def test_ads_route_is_the_only_source_of_ad_candidates():
    kernel = _kernel()
    tick = kernel.step(0, _plan())
    trace = tick.candidate_trace
    ad = kernel.world.catalog.content_kind[
        trace.route_item_id.clamp_min(0)
    ] == int(ContentKind.AD)
    route = torch.zeros_like(trace.route_valid)
    route[:, trace.manifest.route_names.index("ads_auction")] = True
    assert not (trace.route_valid & ad & ~route).any()
    assert (trace.route_valid & ad & route).any()


def test_ads_budget_allocation_is_invariant_to_cell_execution_order():
    left, right = _kernel(), _kernel()
    left.step(0, _plan(), cell_order=(-1, 0, 1))
    right.step(0, _plan(), cell_order=(1, 0, -1))
    left_events = left.event_log.read()
    right_events = right.event_log.read()
    left_spend = left_events.event(EventType.AD_SPEND)
    right_spend = right_events.event(EventType.AD_SPEND)
    torch.testing.assert_close(
        left_events.event_id[left_spend], right_events.event_id[right_spend],
    )
    torch.testing.assert_close(
        left_events.value[left_spend], right_events.value[right_spend],
    )
    torch.testing.assert_close(
        left.world.supply.state.advertiser_budget,
        right.world.supply.state.advertiser_budget,
    )


def test_ads_launch_keeps_market_and_factual_stream_transactional(tmp_path):
    _factual_checkpoint(tmp_path)
    report = run_ads_launch(AdsLaunchConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        request_stream_root=str(tmp_path / "requests"),
        users=256,
        items=3_000,
        device="cpu",
        experiment_steps=2,
        minimum_triggered_users=10_000,
        maximum_attempts=1,
    ))
    review = report["review"]
    assert review["decision"] == "stop_inconclusive"
    assert review["trace_counts"]["control_ad_exposures"] == 0
    assert review["trace_counts"]["treatment_ad_exposures"] > 0
    assert review["market_audit"]["overspend_events"] == 0
    assert review["market_audit"]["partially_billed_impressions"] == 0
    assert report["request_stream_sha256"]
