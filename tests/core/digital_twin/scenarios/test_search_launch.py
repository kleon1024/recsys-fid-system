from __future__ import annotations

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.engine import ExperimentPlan
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
)
from fid_lab.simulation.digital_twin.platform import CascadePolicy
from fid_lab.simulation.digital_twin.scenarios.search.launch import (
    SearchLaunchConfig,
    run_search_launch,
)
from fid_lab.simulation.digital_twin.scenarios.search.metrics import (
    search_decision,
)


def _factual_checkpoint(tmp_path) -> None:
    runtime = RetrievalLadderConfig(
        users=1_024,
        items=8_000,
        device="cpu",
        ticks_per_day=8,
    )
    _, kernel = _build_kernel(runtime)
    policy = CascadePolicy(
        "search-exact-baseline",
        1,
        1,
        1,
        enabled_routes=("random", "popular"),
        enabled_business_routes=("search",),
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


def test_search_launch_isolates_semantic_route_and_keeps_factual_stream(tmp_path):
    _factual_checkpoint(tmp_path)
    report = run_search_launch(SearchLaunchConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        request_stream_root=str(tmp_path / "requests"),
        users=1_024,
        items=8_000,
        device="cpu",
        ticks_per_day=8,
        experiment_steps=2,
        minimum_triggered_users=10_000,
        maximum_attempts=1,
    ))
    review = report["review"]
    assert review["decision"] == "stop_inconclusive"
    assert review["route_counts"]["control_semantic_candidates"] == 0
    assert review["route_counts"]["treatment_semantic_candidates"] > 0
    assert review["sample"]["control_triggered_users"] > 0
    assert review["sample"]["treatment_triggered_users"] > 0
    assert report["request_stream_sha256"]
    store = WorldCheckpointStore(tmp_path / "checkpoints")
    _, kernel = _build_kernel(RetrievalLadderConfig(
        users=1_024, items=8_000, device="cpu", ticks_per_day=8,
    ))
    restored = store.restore(kernel, report["final_checkpoint_id"])
    assert "search_semantic" not in (
        restored.experiment.policies[-1].enabled_business_routes
    )


def test_search_gate_requires_supported_lift_and_guardrails():
    lift = {
        "control_mean": 0.20,
        "treatment_mean": 0.24,
        "absolute_delta": 0.04,
        "relative_delta": 0.20,
        "ci95_low": 0.01,
        "ci95_high": 0.07,
    }
    safe = {
        "control_mean": 0.30,
        "treatment_mean": 0.30,
        "absolute_delta": 0.0,
        "relative_delta": 0.0,
        "ci95_low": -0.005,
        "ci95_high": 0.005,
    }
    decision, _ = search_decision(
        {
            "success_rate": lift,
            "reformulation_rate": safe,
            "abandonment_rate": safe,
            "detail_rate": safe,
            "post_search_feed_rate": safe,
        },
        {"control_triggered_users": 500, "treatment_triggered_users": 500},
        {
            "control_requests": 600,
            "treatment_requests": 600,
            "control_semantic_candidates": 0,
            "treatment_semantic_candidates": 4_000,
        },
        200,
    )
    assert decision == "promote"
