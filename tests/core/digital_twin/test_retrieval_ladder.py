from __future__ import annotations

import json

import torch

from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _decision,
    run_retrieval_ladder,
)
from fid_lab.simulation.digital_twin.experiments.ranking.linear_launch import (
    _auc,
    _gauc,
    _save_serving_artifact,
)
from fid_lab.simulation.digital_twin.experiments.launch_review.metrics import (
    validate_aa,
)
from fid_lab.simulation.digital_twin.learning.probe import (
    ProbeArtifact,
    ProbeRanker,
)
from fid_lab.simulation.digital_twin.platform.features import (
    DEFAULT_FEATURE_MANIFEST,
)
from fid_lab.simulation.digital_twin.value_tree import task_value_weights


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
        assert treatment == ["random", "popular"]
        assert review["comparison_kind"] == "route_addition"
        assert review["merge_policy"] == "reciprocal_rank_fusion"
        assert review["treatment_route_weights"] == {
            "random": 0.1,
            "popular": 1.0,
        }
        assert review["changed_owner"] == "retrieval routes only"
        stages = review["treatment_route_stage_candidates"]
        assert stages["recall"] >= stages["coarse"]
        assert stages["coarse"] >= stages["fine"]
        assert stages["fine"] >= stages["exposed"]
    assert result["final_active_routes"] == ["random"]


def test_retrieval_ladder_can_freeze_a_learned_fine_ranker(tmp_path):
    inputs = len(DEFAULT_FEATURE_MANIFEST.dense_fields)
    artifact = ProbeArtifact(
        model=ProbeRanker(inputs, 1),
        task_names=("long_view",),
        feature_manifest_hash=DEFAULT_FEATURE_MANIFEST.manifest_hash,
        training_report={"purpose": "test"},
        dense_mean=torch.zeros(inputs),
        dense_scale=torch.ones(inputs),
        serving_task_weights=(0.45,),
        task_logit_offsets=torch.zeros(1),
    )
    checkpoint = tmp_path / "fine-ranker.pt"
    saved = _save_serving_artifact(artifact, tmp_path)
    checkpoint = tmp_path / "fine-ranker.pt"
    assert saved["path"] == str(checkpoint)
    assert len(saved["sha256"]) == 64

    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=128,
        items=1_200,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=10_000,
        max_attempts_per_review=1,
        max_reviews=1,
        fine_ranker_checkpoint=str(checkpoint),
    ))

    assert result["config"]["fine_ranker_checkpoint"] == str(checkpoint)
    assert result["reviews"][0]["added_route"] == "popular"


def test_linear_launch_auc_uses_average_rank_for_ties():
    label = torch.tensor((0.0, 1.0, 0.0, 1.0))
    assert _auc(label, torch.zeros_like(label)) == 0.5
    request = torch.tensor((1, 1, 2, 2))
    score = torch.tensor((0.0, 1.0, 1.0, 0.0))
    assert _gauc(request, label, score) == 0.5


def test_feed_value_tree_penalizes_negative_feedback():
    weights = task_value_weights(("stay_value", "long_view", "negative", "payment"))
    assert weights == (0.30, 0.35, -0.35, 0.0)


def test_aa_gate_fails_when_primary_or_guardrail_excludes_zero():
    neutral = {"ci95_low": -0.1, "ci95_high": 0.1, "control_mean": 1.0}
    shifted = {"ci95_low": 0.01, "ci95_high": 0.2, "control_mean": 1.0}
    valid, _ = validate_aa({
        "dwell_seconds": neutral,
        "negative": neutral,
    })
    invalid, reason = validate_aa({
        "dwell_seconds": shifted,
        "negative": neutral,
    })
    assert valid
    assert not invalid
    assert reason == "A/A dwell_seconds confidence interval excludes zero"


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


def test_popular_baseline_can_run_aa_then_add_a_personalized_route():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=512,
        items=12_000,
        burn_in_steps=8,
        aa_steps=4,
        experiment_steps=4,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=2,
        max_reviews=1,
        initial_route="popular",
        route_ladder=("interest_popular",),
    ))
    assert result["aa_review"] is not None
    assert result["aa_review"]["policy_routes"] == ["popular"]
    review = result["reviews"][0]
    assert review["control_routes"] == ["popular"]
    assert review["treatment_routes"] == ["popular", "interest_popular"]
    assert review["comparison_kind"] == "route_addition"


def test_promoted_multi_route_baseline_can_launch_the_next_route():
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=512,
        items=12_000,
        burn_in_steps=8,
        experiment_steps=4,
        control_fraction=0.5,
        treatment_fraction=0.5,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=2,
        max_reviews=1,
        initial_routes=("random", "popular"),
        route_ladder=("cold_start",),
    ))
    review = result["reviews"][0]
    assert review["control_routes"] == ["random", "popular"]
    assert review["treatment_routes"] == ["random", "popular", "cold_start"]


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
        experiment_steps=16,
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
    assert json.dumps(
        resumed["reviews"][:1], sort_keys=True,
    ) == json.dumps(first["reviews"], sort_keys=True)
    assert len(resumed["reviews"]) == 2
    assert resumed["reviews"][1]["added_route"] == "popular"
    assert resumed["reviews"][1]["attempt"] == 2
    assert resumed["reviews"][1]["requests"]["control"] > (
        first["reviews"][0]["requests"]["control"]
    )
    assert resumed["resumed_from_checkpoint"] == first["final_checkpoint_id"]
    assert resumed["world_branch"] == "main"
