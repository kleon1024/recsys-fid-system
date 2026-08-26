from __future__ import annotations

from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _decision,
    run_retrieval_ladder,
)


def test_retrieval_launches_change_one_route_and_preserve_factual_world():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=2_048,
        items=20_000,
        burn_in_steps=32,
        experiment_steps=16,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=2,
        max_reviews=1,
    ))
    assert len(result["reviews"]) == 1
    assert result["reviews"][0]["added_route"] == "popular"
    for review in result["reviews"]:
        control = review["control_routes"]
        treatment = review["treatment_routes"]
        assert control == ["random"]
        assert treatment == ["popular"]
        assert review["comparison_kind"] == "single_route_replacement"
        assert review["merge_policy"] == "single_route_passthrough"
        assert review["changed_owner"] == "retrieval routes only"
        stages = review["treatment_route_stage_candidates"]
        assert stages["recall"] >= stages["coarse"]
        assert stages["coarse"] >= stages["fine"]
        assert stages["fine"] >= stages["exposed"]
    assert result["final_active_routes"] == ["random"]


def test_empty_or_nonfinite_launch_cannot_promote():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=128,
        items=1_200,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.01,
        treatment_fraction=0.01,
        device="cpu",
        minimum_triggered_users=100,
    ))
    assert len(result["reviews"]) == 1
    assert result["reviews"][0]["decision"] == "hold"
    assert result["final_active_routes"] == ["random"]


def test_inconclusive_launch_stops_at_preregistered_window_limit():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=128,
        items=1_200,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.01,
        treatment_fraction=0.01,
        device="cpu",
        minimum_triggered_users=100,
        max_attempts_per_review=1,
        max_reviews=1,
    ))
    assert result["reviews"][0]["decision"] == "stop_inconclusive"
    assert not result["reviews"][0]["promoted_to_next_baseline"]


def test_significant_primary_regression_rejects_before_promotion():
    metric = {
        "control_mean": 1.0,
        "treatment_mean": 0.8,
        "absolute_delta": -0.2,
        "relative_delta": -0.2,
        "ci95_low": -0.3,
        "ci95_high": -0.1,
    }
    neutral = {**metric, "ci95_low": -0.1, "ci95_high": 0.1}
    decision, reason = _decision(
        {"dwell_seconds": metric, "negative": neutral},
        {"control_triggered_users": 2_000, "treatment_triggered_users": 2_000},
        1_500,
    )
    assert decision == "reject"
    assert reason == "stay significantly decreases"


def test_retrieval_ladder_resumes_from_registered_world_head(tmp_path):
    initial = RetrievalLadderConfig(
        users=128,
        items=1_200,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=10_000,
        checkpoint_root=str(tmp_path),
        max_reviews=1,
    )
    first = run_retrieval_ladder(initial)
    assert len(first["checkpoint_ids"]) == 2
    assert len(first["reviews"]) == 1
    resumed = run_retrieval_ladder(RetrievalLadderConfig(
        **{
            **initial.__dict__,
            "max_reviews": 1,
        },
    ))
    assert resumed["reviews"][:1] == first["reviews"]
    assert len(resumed["reviews"]) == 2
    assert resumed["reviews"][1]["added_route"] == "popular"
    assert resumed["reviews"][1]["attempt"] == 2
    assert resumed["reviews"][1]["requests"]["control"] > (
        first["reviews"][0]["requests"]["control"]
    )
    assert resumed["resumed_from_checkpoint"] == first["final_checkpoint_id"]
    assert resumed["world_branch"] == "main"
