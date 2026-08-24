from __future__ import annotations

from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    run_retrieval_ladder,
)


def test_retrieval_launches_change_one_route_and_preserve_factual_world():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=256,
        items=2_400,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=2,
    ))
    assert len(result["reviews"]) == 5
    for review in result["reviews"]:
        control = review["control_routes"]
        treatment = review["treatment_routes"]
        assert treatment[:-1] == control
        assert treatment[-1] == review["added_route"]
        assert review["changed_owner"] == "retrieval routes only"
        assert review["requests"]["control"] > 0
        assert review["requests"]["treatment"] > 0
    assert result["final_active_routes"] == ["popular"]


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
    assert all(review["decision"] == "hold" for review in result["reviews"])
    assert result["final_active_routes"] == ["popular"]
