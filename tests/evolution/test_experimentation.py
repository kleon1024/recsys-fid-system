from fid_lab.feed_loop.small_effect_ab import run_small_effect_ab
import pytest
from fid_lab.simulation.experimentation import (
    Experiment,
    ExperimentLayer,
    FeedParameters,
    Variant,
    assign_layer_numpy,
    assign_layers,
)


def test_overlapping_layers_are_balanced_and_full_chain_parameterized():
    layers = (
        ExperimentLayer(
            "rank",
            "rank-layer",
            (
                Experiment(
                    "fine-model",
                    (
                        Variant("control", 0.25, {"fine_model": "lr_v1"}),
                        Variant("treatment", 0.25, {"fine_model": "deepfm_v1"}),
                    ),
                ),
            ),
        ),
        ExperimentLayer(
            "value",
            "value-layer",
            (
                Experiment(
                    "diversity",
                    (
                        Variant("control", 0.25, {"diversity_strength": 0.0}),
                        Variant("treatment", 0.25, {"diversity_strength": 0.1}),
                    ),
                ),
            ),
        ),
    )
    assignments = [assign_layers(user, layers, FeedParameters()) for user in range(20_000)]
    rank_treatment = sum(
        value["assignments"]["rank"]["variant"] == "treatment"
        for value in assignments
    )
    value_treatment = sum(
        value["assignments"]["value"]["variant"] == "treatment"
        for value in assignments
    )
    both = sum(
        value["assignments"]["rank"]["variant"] == "treatment"
        and value["assignments"]["value"]["variant"] == "treatment"
        for value in assignments
    )
    assert 0.23 < rank_treatment / len(assignments) < 0.27
    assert 0.23 < value_treatment / len(assignments) < 0.27
    assert 0.05 < both / len(assignments) < 0.075
    assert assignments[0] == assign_layers(0, layers, FeedParameters())
    assert "model_manifest" in assignments[0]["parameters"]
    ids = __import__("numpy").arange(1_000_000, dtype="uint64")
    rank_cells, _ = assign_layer_numpy(ids, layers[0])
    value_cells, _ = assign_layer_numpy(ids, layers[1])
    assert 0.24 < (rank_cells == 1).mean() < 0.26
    assert abs(__import__("numpy").corrcoef(rank_cells == 1, value_cells == 1)[0, 1]) < 0.01


def test_vectorized_ab_recovers_effect_and_cuped_reduces_variance():
    report = run_small_effect_ab(users=200_000, relative_effects=(0.01,))[0]
    assert abs(report["cuped_relative_lift"] - 0.01) < 0.005
    assert report["variance_reduction"] > 0.25
    assert report["truth_inside_cuped_interval"]


def test_two_layers_cannot_own_the_same_parameter():
    layers = tuple(
        ExperimentLayer(
            name,
            name,
            (Experiment(name, (Variant("treatment", 0.5, {"fine_model": name}),)),),
        )
        for name in ("layer_a", "layer_b")
    )
    with pytest.raises(ValueError, match="owned by both"):
        assign_layers(1, layers, FeedParameters())
