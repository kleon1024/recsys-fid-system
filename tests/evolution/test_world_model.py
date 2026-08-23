from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fid_lab.feed_loop.world_model.contracts import WorldModelConfig
from fid_lab.feed_loop.world_model.data import (
    WorldModelSplit,
    _with_session_exit,
    load_world_split,
)
from fid_lab.feed_loop.world_model.ensemble import StructuralNoise, WorldModelEnsemble
from fid_lab.feed_loop.world_model.loss import world_model_loss
from fid_lab.feed_loop.world_model.training import (
    load_world_ensemble,
    save_world_ensemble,
)
from fid_lab.feed_loop.world_model.validation import evaluate_world_model
from fid_lab.feed_loop.world_model.benchmark.neural import (
    DINRequestRanker,
    SlateTransformerRanker,
)


def _config() -> WorldModelConfig:
    return WorldModelConfig(
        width=32, latent_dim=8, attention_heads=4, ensemble_members=2,
        batch_size=8, epochs=1,
    )


def _split(rows=16) -> WorldModelSplit:
    generator = torch.Generator().manual_seed(7)
    selected = torch.rand(rows, 28, generator=generator)
    selected[:, 12] = 0.7
    labels = torch.zeros(rows, 16)
    labels[:, 0] = torch.arange(rows) % 2
    labels[:, 1] = labels[:, 0]
    labels[:, 2] = labels[:, 0] * 12.0
    labels[:, 3] = labels[:, 0] * 0.5
    labels[:, 5] = labels[:, 0]
    labels[:, 6] = labels[:, 0]
    labels[:, 7] = labels[:, 0] * (torch.arange(rows) % 3 == 0)
    return WorldModelSplit(
        selected_features=selected,
        slate_features=torch.rand(rows, 5, 28, generator=generator),
        sequence=torch.rand(rows, 6, 8, generator=generator),
        labels=labels,
        label_masks=torch.ones_like(labels),
        weights=torch.ones(rows),
        lifecycle=torch.arange(rows) % 4,
        region=torch.arange(rows) % 10,
        user_ids=torch.arange(rows),
        request_steps=torch.arange(rows) % 6,
        exposed_index=torch.zeros(rows, dtype=torch.long),
        candidate_fine_scores=torch.rand(rows, 5, generator=generator),
        candidate_audit_utility=torch.rand(rows, 5, generator=generator),
    )


def test_neural_scm_forward_loss_and_gradient_are_finite():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    output = ensemble.members[0](batch)
    loss, tasks = world_model_loss(output, batch, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in ensemble.members[0].parameters() if parameter.grad is not None)
    assert "stay_nll" in tasks


def test_paired_structural_noise_is_repeatable_and_respects_funnel_masks():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    noise = StructuralNoise.generate(len(split), config, "cpu", 99)
    first = ensemble.sample_members(batch, noise)[0]
    second = ensemble.sample_members(batch, noise)[0]
    assert torch.equal(first["stay_seconds"], second["stay_seconds"])
    assert not (first["actions"]["play_3s"] & ~first["actions"]["play"]).any()
    assert not (first["actions"]["poi_detail"] & ~first["actions"]["anchor_click"]).any()
    assert not (first["actions"]["conversion"] & ~first["actions"]["poi_detail"]).any()
    assert not (
        first["actions"]["returned_next_session"]
        & ~first["actions"]["session_exit"]
    ).any()


def test_evaluation_fails_closed_without_randomized_causal_evidence():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    report = evaluate_world_model(
        ensemble, _split(), "cpu", "sha256:test",
        distribution_rows=16, rollout_rows=16,
    )
    assert report["decision"] == "hold_research_challenger"
    assert report["gates"]["intervention_recovery"] is False
    assert report["gates"]["policy_order"] is False


def test_artifact_round_trip_preserves_member_predictions():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    before = ensemble.predict(batch)["probability_mean"]
    with TemporaryDirectory() as directory:
        save_world_ensemble(
            ensemble, [[] for _ in ensemble.members], Path(directory),
            {"manifest_sha256": "dataset", "authority_bundle_id": "bundle"},
        )
        replay = load_world_ensemble(Path(directory), "cpu")
        after = replay.predict(batch)["probability_mean"]
    torch.testing.assert_close(before, after)


def test_ensemble_initialization_is_bound_to_the_declared_seed():
    config = _config()
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    first = WorldModelEnsemble(config).predict(batch)["probability_mean"]
    torch.manual_seed(123456)
    second = WorldModelEnsemble(config).predict(batch)["probability_mean"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_session_exit_is_point_in_time_and_last_request_is_masked():
    payload = {
        "labels": torch.zeros(4, 15),
        "label_masks": torch.ones(4, 15),
        "user_id": torch.tensor([1, 2, 1, 2]),
        "request_step": torch.tensor([0, 0, 1, 1]),
        "session_id": torch.tensor([1, 1, 2, 1]),
    }
    labels, masks = _with_session_exit(payload, 4)
    assert labels[:, 15].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert masks[:, 15].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_request_rankers_score_every_candidate_from_the_same_request():
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    assert torch.equal(batch["exposed_index"], split.exposed_index)
    for model in (DINRequestRanker(width=16), SlateTransformerRanker(width=16)):
        scores = model(batch["slate_features"], batch["sequence"])
        assert scores.shape == split.slate_features.shape[:2]
        assert torch.isfinite(scores).all()


def test_uniform_row_selection_preserves_full_split_support():
    payload = {
        "exposed_index": torch.zeros(10, dtype=torch.long),
        "exposure_propensity": torch.ones(10),
        "candidate_features": torch.rand(10, 2, 28),
        "behavior_sequence": torch.rand(10, 3, 8),
        "labels": torch.zeros(10, 15),
        "label_masks": torch.ones(10, 15),
        "lifecycle_bucket": torch.zeros(10),
        "region_bucket": torch.zeros(10),
        "user_id": torch.arange(10),
        "request_step": torch.arange(10),
        "session_id": torch.zeros(10),
        "candidate_fine_scores": torch.zeros(10, 2),
        "candidate_audit_utility": torch.zeros(10, 2),
    }
    with TemporaryDirectory() as directory:
        torch.save({"tensors": payload}, Path(directory) / "train.pt")
        split = load_world_split(Path(directory), "train", 3, "uniform")
    assert split.request_steps.tolist() == [0, 3, 6]
