from __future__ import annotations

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.engine import ExperimentPlan
from fid_lab.simulation.digital_twin.experiments.cold_start_launch import (
    ColdStartLaunchConfig,
    _decision,
    _rates,
    run_cold_start_launch,
)
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
)
from fid_lab.simulation.digital_twin.experiments.surface_route_recovery import (
    SurfaceRouteRecoveryConfig,
    recover_surface_routes,
)
from fid_lab.simulation.digital_twin.platform import CascadePolicy


def _factual_checkpoint(tmp_path):
    runtime = RetrievalLadderConfig(
        users=1_024,
        items=8_000,
        device="cpu",
        ticks_per_day=8,
        checkpoint_root=str(tmp_path / "checkpoints"),
    )
    _, kernel = _build_kernel(runtime)
    policy = CascadePolicy(
        "video-dedup-baseline",
        1,
        1,
        1,
        enabled_routes=("random", "popular"),
        feed_exposure_dedup_ticks=240,
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


def test_cold_start_launch_logs_two_level_randomized_support(tmp_path):
    _factual_checkpoint(tmp_path)
    report = run_cold_start_launch(ColdStartLaunchConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        request_stream_root=str(tmp_path / "requests"),
        users=1_024,
        items=8_000,
        device="cpu",
        experiment_steps=2,
        minimum_triggered_users=10_000,
        maximum_attempts=1,
    ))
    counts = report["review"]["traffic_counts"]
    rates = report["review"]["traffic_rates"]
    assert report["review"]["decision"] == "stop_inconclusive"
    assert counts["randomized_requests"] > 0
    assert counts["cold_start_exposures"] == counts["randomized_requests"]
    assert counts["invalid_cold_exposures"] == 0
    assert 0.0 < counts["minimum_logged_propensity"] <= 0.10
    assert 0.0 < rates["global_randomized_request_rate"] < 0.05


def test_cold_start_gate_accepts_only_supported_two_percent_lane():
    metric = {
        "control_mean": 1.0,
        "treatment_mean": 1.0,
        "absolute_delta": 0.0,
        "relative_delta": 0.0,
        "ci95_low": -0.02,
        "ci95_high": 0.02,
    }
    metrics = {"dwell_seconds": metric, "negative": metric}
    counts = {
        "feed_requests": 10_000,
        "control_requests": 2_000,
        "treatment_requests": 2_000,
        "supported_treatment_requests": 1_900,
        "randomized_requests": 200,
        "cold_start_exposures": 200,
        "invalid_cold_exposures": 0,
        "minimum_logged_propensity": 0.01,
        "maximum_logged_propensity": 0.10,
    }
    decision, _ = _decision(
        metrics,
        {"control_triggered_users": 800, "treatment_triggered_users": 800},
        counts,
        _rates(counts),
        ColdStartLaunchConfig("checkpoints", "requests"),
    )
    assert decision == "accept_layer"


def test_surface_route_recovery_invalidates_pending_cold_start_launch(tmp_path):
    _factual_checkpoint(tmp_path)
    pending = run_cold_start_launch(ColdStartLaunchConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        request_stream_root=str(tmp_path / "requests"),
        users=1_024,
        items=8_000,
        device="cpu",
        experiment_steps=1,
        minimum_triggered_users=10_000,
        maximum_attempts=2,
    ))
    assert pending["review"]["decision"] == "hold"
    recovered = recover_surface_routes(SurfaceRouteRecoveryConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        users=1_024,
        items=8_000,
        device="cpu",
        ticks_per_day=8,
    ))
    assert recovered["invalidated_launch"] == "R-LR-011"
    assert "posting_context" in recovered["business_routes"]
    store = WorldCheckpointStore(tmp_path / "checkpoints")
    _, kernel = _build_kernel(RetrievalLadderConfig(
        users=1_024, items=8_000, device="cpu", ticks_per_day=8,
    ))
    restored = store.restore(kernel, recovered["checkpoint_id"])
    assert restored.learning_cursors["cold_start_launch_v2"]["decision"] == (
        "invalidated_missing_business_route_authority"
    )
