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
    _sample_stay,
    load_world_ensemble,
    save_world_ensemble,
)
from fid_lab.feed_loop.world_model.validation import evaluate_world_model
from fid_lab.feed_loop.world_model.validation.evaluation import _distribution_report
from fid_lab.feed_loop.world_model.validation.synthetic import (
    synthetic_causal_validation,
)
from fid_lab.feed_loop.world_model.benchmark.neural import (
    DINRequestRanker,
    SlateTransformerRanker,
)
from fid_lab.feed_loop.world_model.external.kuairand.data.core_bridge import (
    bridge_split,
)
from fid_lab.feed_loop.world_model.external.kuairand.data.randomized import (
    RandomizedSplit,
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
    labels = torch.zeros(rows, 21)
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


def test_distribution_evaluation_ignores_unobserved_labels():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    baseline = _split()
    masks = baseline.label_masks.clone()
    masks[:, 9:] = 0.0
    first = WorldModelSplit(**{
        **baseline.__dict__, "label_masks": masks,
    })
    changed_labels = baseline.labels.clone()
    changed_labels[:, 9:] = 1.0
    second = WorldModelSplit(**{
        **baseline.__dict__, "labels": changed_labels, "label_masks": masks,
    })
    first_report = _distribution_report(ensemble, first, "cpu", len(first))
    second_report = _distribution_report(ensemble, second, "cpu", len(second))
    assert first_report["binary_ece"] == second_report["binary_ece"]
    assert first_report["joint_correlation_mae"] == second_report[
        "joint_correlation_mae"
    ]
    assert first_report["binary_ece"]["anchor_click"] is None


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


def test_stay_calibration_handles_split_smaller_than_global_limit():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    generated = _sample_stay(
        ensemble, _split(rows=5), torch.device("cpu"), limit=100_000
    )
    assert len(generated) == 5 * config.ensemble_members


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
    assert labels.shape[1] == 21
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


def test_external_bridge_preserves_observed_actions_and_masks_unknown_business_labels():
    rows = 4
    sparse = torch.tensor([
        [1, 10, 20, 30, 1, 1, 1],
        [2, 11, 21, 31, 1, 1, 1],
        [3, 12, 22, 32, 1, 1, 1],
        [4, 13, 23, 33, 1, 1, 1],
    ])
    dense = torch.zeros(rows, 11)
    dense[:, 0] = 0.7
    labels = torch.zeros(rows, 8)
    labels[:, 0] = torch.tensor([1, 0, 1, 0])
    labels[:, 1] = torch.tensor([1, 0, 0, 0])
    labels[:, 2:7] = 1
    labels[:, 7] = 0.5
    history_items = torch.zeros(rows, 64, dtype=torch.int32)
    history_items[:, -1] = sparse[:, 1].int()
    history_feedback = torch.zeros(rows, 64, 7, dtype=torch.uint8)
    history_feedback[:, -1, 0] = 1
    split = RandomizedSplit(
        sparse, dense, history_items, history_feedback, labels,
        torch.arange(rows), torch.arange(rows) * 1_000,
        torch.full((rows,), 20220422), torch.arange(10, 14),
        torch.full((rows,), 1 / 100),
    )
    catalog = {
        "sparse": sparse,
        "dense": dense,
        "raw_video_ids": torch.arange(10, 14),
        "standard_exposure_count": torch.arange(1, 5),
    }
    payload = bridge_split(split, catalog, candidates=3)
    assert payload["candidate_features"].shape == (rows, 3, 28)
    assert payload["behavior_sequence"].shape == (rows, 24, 8)
    assert payload["labels"][:, 19].tolist() == [1, 0, 1, 0]
    assert payload["label_masks"][:, 9:16].sum() == 0
    assert payload["label_masks"][:, 16:20].all()
    assert payload["labels"][:, 20].tolist() == [1, 0, 0, 0]
    assert payload["candidate_utility_source"] == (
        "unavailable_external_randomized_bridge"
    )
    assert torch.isnan(payload["candidate_audit_utility"]).all()


def test_external_bridge_cannot_claim_synthetic_policy_order():
    split = _split()
    split = WorldModelSplit(
        **{
            **split.__dict__,
            "candidate_audit_utility": torch.full_like(
                split.candidate_audit_utility, torch.nan
            ),
            "candidate_utility_source": (
                "unavailable_external_randomized_bridge"
            ),
        }
    )
    report = synthetic_causal_validation(
        WorldModelEnsemble(_config()), split, "cpu"
    )["policy_order"]
    assert report["available"] is False
    assert report["pass"] is False
    assert report["kendall_tau"] is None
