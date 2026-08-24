from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fid_lab.feed_loop.world_model.contracts import WorldModelConfig
from fid_lab.feed_loop.world_model.data import (
    WorldModelSplit,
    _with_session_exit,
    concatenate_world_splits,
    load_world_split,
)
from fid_lab.feed_loop.world_model.cli import (
    _assert_artifact_dataset,
    _combined_dataset_manifest,
)
from fid_lab.feed_loop.world_model.ensemble import StructuralNoise, WorldModelEnsemble
from fid_lab.feed_loop.world_model.feature_contract import (
    compare_feature_contracts,
)
from fid_lab.feed_loop.world_model.loss import world_model_loss
from fid_lab.feed_loop.world_model.training import (
    adapt_world_ensemble,
    adapt_structural_ensemble,
    _fit_platt,
    _sample_stay,
    load_world_ensemble,
    recenter_utility_ensemble,
    save_world_ensemble,
    train_world_ensemble,
)
from fid_lab.feed_loop.world_model.validation import evaluate_world_model
from fid_lab.feed_loop.world_model.validation.evaluation import _distribution_report
from fid_lab.feed_loop.world_model.validation.policy_evidence import (
    verify_policy_evidence,
)
from fid_lab.feed_loop.world_model.validation.boundary import (
    boundary_invariance_report,
)
from fid_lab.feed_loop.world_model.validation.support import (
    anti_exploitation_report,
    fit_support_profile,
    support_report,
)
from fid_lab.feed_loop.world_model.validation.synthetic import (
    synthetic_causal_validation,
)
from fid_lab.feed_loop.world_model.benchmark.neural import (
    DINRequestRanker,
    SlateTransformerRanker,
)
from fid_lab.feed_loop.world_model.external.kuairand.data.core_bridge import (
    KUAI_FEATURE_CONTRACT,
    KUAI_FEATURE_COVERAGE,
    adaptation_calibration_masks,
    bridge_split,
    catalog_action_features,
)
from fid_lab.simulation.digital_twin.world.neural_features import (
    V4_FEATURE_CONTRACT,
    V4_FEATURE_COVERAGE,
)
from fid_lab.simulation.digital_twin.world.training import (
    StructuralBridgeConfig,
    build_structural_bridge,
)
from fid_lab.feed_loop.world_model.external.kuairand.evaluation.neural_policy import (
    _policy_probabilities,
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
        user_ids=torch.div(torch.arange(rows), 8, rounding_mode="floor"),
        request_steps=torch.arange(rows) % 8,
        exposed_index=torch.zeros(rows, dtype=torch.long),
        candidate_fine_scores=torch.rand(rows, 5, generator=generator),
        candidate_audit_utility=torch.rand(rows, 5, generator=generator),
        event_days=torch.div(torch.arange(rows), 4, rounding_mode="floor"),
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


def test_vectorized_slate_sampling_matches_scalar_candidate_semantics():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    split = _split(rows=4)
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    rows, width = batch["slate_features"].shape[:2]
    generator = torch.Generator().manual_seed(99)
    slate_noise = StructuralNoise(
        latent=torch.randn(rows, width, config.latent_dim, generator=generator),
        mixture=torch.rand(rows, width, generator=generator),
        stay=torch.randn(rows, width, generator=generator),
        actions=torch.rand(
            rows, width, len(ensemble.members[0].action_heads), generator=generator,
        ),
    )
    vectorized = ensemble.sample_slate_members(batch, slate_noise)[0]
    scalar_batch = {
        **batch,
        "selected_features": batch["slate_features"][:, 0],
    }
    scalar_noise = StructuralNoise(
        latent=slate_noise.latent[:, 0],
        mixture=slate_noise.mixture[:, 0],
        stay=slate_noise.stay[:, 0],
        actions=slate_noise.actions[:, 0],
    )
    scalar = ensemble.sample_members(scalar_batch, scalar_noise)[0]
    torch.testing.assert_close(
        vectorized["stay_seconds"][:, 0], scalar["stay_seconds"],
    )
    for name in scalar["actions"]:
        assert torch.equal(vectorized["actions"][name][:, 0], scalar["actions"][name])


def test_evaluation_fails_closed_without_randomized_policy_evidence():
    config = _config()
    ensemble = WorldModelEnsemble(config)
    report = evaluate_world_model(
        ensemble, _split(), "cpu", "sha256:test",
        distribution_rows=16, rollout_rows=16,
    )
    assert report["decision"] == "hold_research_challenger"
    assert report["gates"]["structural_intervention_recovery"] is False
    assert report["gates"]["external_policy_order"] is False
    assert report["external_policy"]["available"] is False


def test_sequence_gate_ignores_channels_the_external_source_never_observed():
    split = _split()
    sequence = split.sequence.clone()
    sequence[:, :, 3] = 0.0
    sequence[:, :, 6:] = 0.0
    split = WorldModelSplit(**{**split.__dict__, "sequence": sequence})
    report = evaluate_world_model(
        WorldModelEnsemble(_config()), split, "cpu", "sha256:test",
        distribution_rows=16, rollout_rows=16,
    )["sequence"]
    assert 3 not in report["evaluated_event_channels"]
    assert 6 not in report["evaluated_event_channels"]
    assert 7 not in report["evaluated_event_channels"]


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


def test_world_model_is_invariant_to_order_batches_labels_and_platform_scores():
    report = boundary_invariance_report(
        WorldModelEnsemble(_config()), _split(), "cpu"
    )
    assert report["pass"] is True
    assert report["deltas"]["future_labels"] == 0.0
    assert report["deltas"]["platform_scores"] == 0.0
    assert max(report["deltas"].values()) <= report["maximum_allowed_delta"]


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


def test_platt_calibration_reduces_heldout_binary_log_loss():
    logits = torch.linspace(-1.0, 1.0, 200)
    labels = (logits > 0.35).float()
    before = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    scale, bias = _fit_platt(logits, labels)
    after = torch.nn.functional.binary_cross_entropy_with_logits(
        logits * scale + bias, labels,
    )
    assert after < before


def test_randomized_adaptation_restores_training_mode_after_validation():
    ensemble = WorldModelEnsemble(_config())
    for member in ensemble.members:
        member.eval()
    report = adapt_world_ensemble(ensemble, _split(), torch.device("cpu"))
    assert report["rows"] == 16
    assert all(member.training for member in ensemble.members)
    assert report["same_user_day_pairs"] > 0
    assert all(row["pointwise_history"] for row in report["members"])
    assert all(row["pairwise_history"] for row in report["members"])


def test_ensemble_initialization_is_bound_to_the_declared_seed():
    config = _config()
    split = _split()
    batch = split.batch(torch.arange(len(split)), torch.device("cpu"))
    first = WorldModelEnsemble(config).predict(batch)["probability_mean"]
    torch.manual_seed(123456)
    second = WorldModelEnsemble(config).predict(batch)["probability_mean"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_world_training_is_reproducible_without_global_rng_state():
    config = WorldModelConfig(
        width=16, latent_dim=8, attention_heads=4,
        ensemble_members=2, batch_size=8, epochs=1,
        randomized_adaptation_epochs=0, randomized_pairwise_epochs=0,
        structural_adaptation_epochs=0,
    )
    split = _split(rows=16)
    first = train_world_ensemble(split, split, config, "cpu")[0]
    torch.randn(1_000)
    second = train_world_ensemble(split, split, config, "cpu")[0]
    for left_member, right_member in zip(first.members, second.members):
        for name, left in left_member.state_dict().items():
            torch.testing.assert_close(left, right_member.state_dict()[name])


def test_post_structural_recenter_changes_intercept_without_rescaling_utility():
    ensemble = WorldModelEnsemble(_config())
    for member in ensemble.members:
        member.utility_calibration_shift.fill_(0.2)
    scales = [member.utility_calibration_scale.clone() for member in ensemble.members]
    report = recenter_utility_ensemble(
        ensemble, _split(rows=16), torch.device("cpu"),
    )
    assert report["method"] == "heldout_post_structural_intercept_only"
    for member_report, expected_scale, member in zip(
        report["members"], scales, ensemble.members,
    ):
        assert member_report["after_mean_error"] < abs(
            member_report["before_mean"] - member_report["target_mean"]
        )
        torch.testing.assert_close(
            member.utility_calibration_scale, expected_scale,
            rtol=0.0, atol=0.0,
        )


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
        "quality_prior": torch.linspace(0.2, 0.8, rows),
        "history_topic_by_item_hash": torch.zeros(262_145, dtype=torch.long),
    }
    catalog["history_topic_by_item_hash"][sparse[:, 1]] = sparse[:, 3]
    payload = bridge_split(split, catalog, candidates=3)
    assert payload["candidate_features"].shape == (rows, 3, 28)
    assert payload["behavior_sequence"].shape == (rows, 24, 8)
    assert payload["labels"][:, 5].tolist() == [1, 0, 0, 0]
    assert payload["label_masks"][:, 9:16].sum() == 0
    assert payload["label_masks"][:, 16:19].all()
    assert payload["label_masks"][:, 19:21].sum() == 0
    assert payload["candidate_utility_source"] == (
        "unavailable_external_randomized_bridge"
    )
    assert torch.isnan(payload["candidate_audit_utility"]).all()


def test_catalog_policy_actions_have_full_support_and_request_context():
    rows = 3
    sparse = torch.tensor([
        [1, 10, 20, 30, 1, 1, 1],
        [2, 11, 21, 31, 1, 1, 1],
        [3, 12, 22, 32, 1, 1, 1],
    ])
    dense = torch.rand(rows, 11)
    split = RandomizedSplit(
        sparse, dense, torch.zeros(rows, 64, dtype=torch.int32),
        torch.zeros(rows, 64, 7, dtype=torch.uint8), torch.zeros(rows, 8),
        torch.arange(rows), torch.arange(rows), torch.ones(rows),
        torch.arange(10, 13), torch.full((rows,), 1 / 3),
    )
    catalog = {
        "sparse": sparse,
        "dense": dense,
        "raw_video_ids": torch.arange(10, 13),
        "standard_exposure_count": torch.tensor([1, 10, 100]),
        "quality_prior": torch.tensor([0.2, 0.5, 0.8]),
        "history_topic_by_item_hash": torch.zeros(262_145, dtype=torch.long),
    }
    catalog["history_topic_by_item_hash"][sparse[:, 1]] = sparse[:, 3]
    actions = torch.tensor([[0, 1], [1, 2], [2, 0]])
    features = catalog_action_features(split, catalog, actions)
    probabilities = _policy_probabilities(catalog)
    assert features.shape == (rows, 2, 28)
    assert torch.all(features[:, :, 14:17] == 0.0)
    assert torch.all(features[:, :, 8] == (1.0 - dense[:, 8])[:, None])
    assert (probabilities > 0).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4))


def test_randomized_adaptation_and_calibration_are_user_disjoint():
    users = torch.arange(10).repeat_interleave(2)
    rows = len(users)
    split = RandomizedSplit(
        torch.zeros(rows, 7, dtype=torch.long), torch.zeros(rows, 11),
        torch.zeros(rows, 64, dtype=torch.int32),
        torch.zeros(rows, 64, 7, dtype=torch.uint8), torch.zeros(rows, 8),
        users, torch.arange(rows), torch.ones(rows), torch.arange(rows),
        torch.full((rows,), 1 / rows),
    )
    pool = users.numpy() < 8
    adaptation, calibration = adaptation_calibration_masks(split, pool)
    adaptation_users = set(users[torch.from_numpy(adaptation)].tolist())
    calibration_users = set(users[torch.from_numpy(calibration)].tolist())
    assert adaptation_users.isdisjoint(calibration_users)
    assert adaptation_users | calibration_users == set(range(8))
    assert adaptation_users
    assert calibration_users
    assert ((adaptation | calibration) == pool).all()


def test_policy_evidence_verifier_recomputes_order_and_support():
    evidence = {
        "schema": "neural-scm-kuairand-policy-evidence-v1",
        "world_model_manifest_sha256": "sha256:test",
        "policies": [
            {
                "name": name, "observed_value": value,
                "predicted_value": value + 0.01,
                "effective_sample_fraction": 0.8,
                "maximum_importance_weight": 2.0,
            }
            for name, value in zip(("a", "b", "c"), (0.3, 0.2, 0.1))
        ],
        "pairwise_observed_differences": [
            {"identified": True}, {"identified": True},
            {"identified": False},
        ],
        "gates": {"producer_gate": True},
        "decision": "pass",
        "evidence_boundary": "test",
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "policy.json"
        path.write_text(__import__("json").dumps(evidence))
        report = verify_policy_evidence(path, "sha256:test")
        mismatch = verify_policy_evidence(path, "sha256:other")
    assert report["policy_order_pass"] is True
    assert report["policy_kendall_tau"] == 1.0
    assert mismatch["policy_order_pass"] is False


def test_external_and_v4_share_semantics_but_not_evidence_coverage():
    mismatches = compare_feature_contracts(
        KUAI_FEATURE_CONTRACT, V4_FEATURE_CONTRACT
    )
    assert mismatches == []
    assert KUAI_FEATURE_CONTRACT["sha256"] == V4_FEATURE_CONTRACT["sha256"]
    assert KUAI_FEATURE_COVERAGE != V4_FEATURE_COVERAGE
    assert KUAI_FEATURE_COVERAGE["5"] == "unavailable_or_unused"
    assert V4_FEATURE_COVERAGE["5"] == "native_v4"


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


def test_structural_bridge_uses_disjoint_families_and_paired_test_worlds():
    with TemporaryDirectory() as directory:
        output = Path(directory) / "bridge"
        manifest = build_structural_bridge(
            output,
            StructuralBridgeConfig(
                rows=30, users=300, items=2_000, ticks=32, device="cpu",
            ),
        )
        train = load_world_split(output, "train")
        validation = load_world_split(output, "validation")
        test = load_world_split(output, "test")
        structural_adaptation = load_world_split(
            output, "structural_adaptation",
        )
        structural_validation = load_world_split(
            output, "structural_validation",
        )
        resumed = build_structural_bridge(
            output,
            StructuralBridgeConfig(
                rows=30, users=300, items=2_000, ticks=32, device="cpu",
            ),
        )
        with __import__("pytest").raises(ValueError, match="config mismatch"):
            build_structural_bridge(
                output,
                StructuralBridgeConfig(
                    rows=31, users=300, items=2_000, ticks=32, device="cpu",
                ),
            )
        next_output = Path(directory) / "next-bridge"
        next_manifest = build_structural_bridge(
            next_output,
            StructuralBridgeConfig(
                rows=30, users=300, items=2_000, ticks=32,
                test_family_id=7, device="cpu",
            ),
            reuse_build=output,
        )
    assert manifest["split_authority"] == (
        "multi_train_disjoint_holdout_world_families"
    )
    assert resumed["family_parts"] == manifest["family_parts"]
    assert len(manifest["family_parts"]) == 4
    assert next_manifest["families"]["test"][0]["family_id"] == 7
    for key in (
        "train:family-1", "train:family-2", "validation:family-3",
    ):
        assert (
            next_manifest["family_parts"][key]["sha256"]
            == manifest["family_parts"][key]["sha256"]
        )
    assert len({
        row["environment_seed"]
        for rows in manifest["families"].values() for row in rows
    }) == 4
    assert all(
        32 <= row["simulated_ticks"] <= 96
        and row["extension_ticks"] == row["simulated_ticks"] - 32
        and row["capture_tick_max"] >= 32 // 3
        for rows in manifest["families"].values() for row in rows
    )
    assert len(train) == 18
    assert len(validation) == 6
    assert len(test) == 6
    assert train.structural_intervention_effects is None
    assert validation.structural_intervention_effects is None
    assert test.structural_intervention_effects.shape == (6, 3)
    assert len(structural_adaptation) == 6
    assert structural_adaptation.structural_intervention_effects.shape == (6, 3)
    assert len(structural_validation) == 2
    assert structural_validation.structural_intervention_effects.shape == (2, 3)
    assert test.structural_intervention_features.shape == (6, 3, 28)
    assert test.structural_intervention_slates.shape == (6, 3, 8, 28)
    assert test.structural_intervention_sequences.shape == (6, 3, 24, 8)
    assert torch.unique(train.structural_family_ids).tolist() == [1, 2]
    assert torch.unique(test.structural_family_ids).tolist() == [5]
    structural = synthetic_causal_validation(
        WorldModelEnsemble(_config()), test, "cpu",
    )["intervention_recovery"]
    assert structural["available"] is True
    assert structural["world"] == "held_out_v4_structural_families"
    assert [row["name"] for row in structural["interventions"]] == [
        "recent_interest_signal",
        "content_quality_prior",
        "negative_history_signal",
    ]


def test_multi_source_manifest_requires_contract_parity_and_unions_coverage():
    primary = {
        "manifest_sha256": "external",
        "feature_contract_sha256": "contract",
        "feature_coverage": {"0": "observed_kuairand", "5": "unavailable"},
    }
    auxiliary = {
        "manifest_sha256": "structural",
        "feature_contract_sha256": "contract",
        "feature_coverage": {"0": "native_v4", "5": "native_v4"},
    }
    combined = _combined_dataset_manifest(primary, auxiliary)
    assert combined["source_manifest_sha256s"] == ["external", "structural"]
    assert combined["feature_coverage"] == {
        "0": "multi_source", "5": "native_v4",
    }
    auxiliary["feature_contract_sha256"] = "other"
    with __import__("pytest").raises(ValueError, match="contracts differ"):
        _combined_dataset_manifest(primary, auxiliary)


def test_mixed_training_split_does_not_smuggle_test_interventions():
    external = _split(rows=4)
    structural = WorldModelSplit(**{
        **_split(rows=3).__dict__,
        "structural_intervention_effects": torch.ones(3, 3),
    })
    mixed = concatenate_world_splits((external, structural))
    assert len(mixed) == 7
    assert mixed.structural_intervention_effects is None


def test_structural_adapter_consumes_only_explicit_paired_rows():
    split = _split(rows=8)
    split = WorldModelSplit(**{
        **split.__dict__,
        "structural_intervention_features": split.selected_features[:, None].repeat(1, 3, 1),
        "structural_intervention_slates": split.slate_features[:, None].repeat(1, 3, 1, 1),
        "structural_intervention_sequences": split.sequence[:, None].repeat(1, 3, 1, 1),
        "structural_intervention_effects": torch.zeros(8, 3),
    })
    ensemble = WorldModelEnsemble(_config())
    protected = {
        name: value.clone()
        for name, value in ensemble.state_dict().items()
        if ".utility_head." not in name
        and ".utility_feature_adapter." not in name
    }
    report = adapt_structural_ensemble(ensemble, split, torch.device("cpu"))
    assert report["method"] == "train_family_paired_effect_finetune"
    assert report["rows"] == 8
    assert all(row["history"] for row in report["members"])
    for name, expected in protected.items():
        torch.testing.assert_close(
            ensemble.state_dict()[name], expected, rtol=0.0, atol=0.0,
        )


def test_support_profile_accepts_reference_distribution_and_rejects_attack():
    split = _split(rows=64)
    profile = fit_support_profile(split, split)
    assert support_report(split, profile)["pass"] is True
    attack = anti_exploitation_report(split, profile)
    assert attack["pass"] is True
    assert attack["rejection_rate"] == 1.0


def test_multi_source_support_profile_preserves_separate_domain_components():
    external = _split(rows=40)
    structural = _split(rows=10)
    external.slate_features.zero_()
    structural.slate_features.fill_(10.0)
    profile = fit_support_profile(
        (external, structural), (external, structural),
    )
    reversed_profile = fit_support_profile(
        (structural, external), (structural, external),
    )
    assert profile["fit_rows_by_source"] == [40, 10]
    assert profile["calibration_rows_by_source"] == [40, 10]
    centers = sorted(row["center"][0] for row in profile["components"])
    reversed_centers = sorted(
        row["center"][0] for row in reversed_profile["components"]
    )
    assert centers == reversed_centers == [0.0, 10.0]


def test_structural_support_preserves_each_training_world_family():
    external = _split(rows=12)
    structural = WorldModelSplit(**{
        **_split(rows=12).__dict__,
        "structural_family_ids": torch.tensor([1] * 6 + [2] * 6),
    })
    validation = WorldModelSplit(**{
        **_split(rows=8).__dict__,
        "structural_family_ids": torch.full((8,), 3),
    })
    profile = fit_support_profile(
        (external, structural), (external, validation),
    )
    assert [row["name"] for row in profile["components"]] == [
        "source_0", "source_1_family_1", "source_1_family_2",
    ]


def test_reused_artifact_must_match_all_dataset_sources_and_coverage():
    dataset = {
        "source_manifest_sha256s": ["external", "structural"],
        "feature_contract_sha256": "contract",
        "feature_coverage": {"0": "multi_source"},
    }
    artifact = {
        "dataset_source_manifest_sha256s": ["external", "structural"],
        "feature_contract_sha256": "contract",
        "feature_coverage": {"0": "multi_source"},
    }
    _assert_artifact_dataset(artifact, dataset)
    artifact["dataset_source_manifest_sha256s"] = ["external", "other"]
    with __import__("pytest").raises(ValueError, match="dataset sources"):
        _assert_artifact_dataset(artifact, dataset)
