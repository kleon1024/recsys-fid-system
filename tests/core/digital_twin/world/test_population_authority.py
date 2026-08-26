from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fid_lab.simulation.digital_twin import (
    EventType,
    RenderedSlateBatch,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
    make_app_events,
)
from fid_lab.feed_loop.world_model.contracts import WorldModelConfig
from fid_lab.feed_loop.world_model.ensemble import WorldModelEnsemble
from fid_lab.feed_loop.world_model.training import save_world_ensemble
from fid_lab.feed_loop.world_model.feature_contract import V4_REQUIRED_FEATURES
from fid_lab.feed_loop.world_model.validation.support import SUPPORT_PROFILE_SCHEMA
from fid_lab.simulation.digital_twin.world.authority import (
    FactualResponseArtifact,
    NeuralFeedResponseAuthority,
    load_factual_response_authority,
)
from fid_lab.simulation.digital_twin.world.neural_features import (
    V4_FEATURE_CONTRACT,
    V4_FEATURE_COVERAGE,
)
from fid_lab.feed_loop.world_model.external.kuairand.data.core_bridge import (
    KUAI_FEATURE_COVERAGE,
)
from fid_lab.simulation.digital_twin.world.dynamics.population import (
    sample_population,
)
from fid_lab.simulation.digital_twin.world.dynamics.trends import TrendProcess
from fid_lab.simulation.digital_twin.world.training.shadow import (
    AuthorityShadowConfig,
    run_authority_shadow,
)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    matrix = torch.corrcoef(torch.stack((left.float(), right.float())))
    return float(matrix[0, 1])


def _wide_support_profile():
    indices = sorted(V4_REQUIRED_FEATURES)
    return {
        "schema": SUPPORT_PROFILE_SCHEMA,
        "feature_indices": indices,
        "combination": "union_of_source_components",
        "components": [{
            "name": "wide-test",
            "feature_indices": indices,
            "distance_feature_indices": indices,
            "bounded_feature_indices": [],
            "bounded_lower": [],
            "bounded_upper": [],
            "center": [0.0] * len(indices),
            "scale": [1.0] * len(indices),
            "request_distance_threshold": 100.0,
        }],
    }


def test_population_is_deterministic_correlated_and_heterogeneous():
    users = torch.arange(20_000)
    left = sample_population(
        users, topics=64, countries=12, regions_per_country=16, seed=103,
    )
    right = sample_population(
        users, topics=64, countries=12, regions_per_country=16, seed=103,
    )
    for name in left.__dataclass_fields__:
        torch.testing.assert_close(getattr(left, name), getattr(right, name))
    assert torch.unique(left.mixture).numel() == 6
    assert torch.unique(left.country).numel() == 12
    assert torch.unique(left.lifecycle_cohort).numel() == 3
    assert torch.allclose(left.weekly_activity.mean(1), torch.ones(len(users)))
    assert _correlation(left.habit, left.activity) > 0.35
    assert _correlation(left.spending_power, left.activity) < 0.45
    assert _correlation(left.satisfaction, left.fatigue) < 0.35
    assert torch.allclose(left.surface_intent.sum(1), torch.ones(len(users)))


def test_trend_process_has_exogenous_and_factual_endogenous_components():
    left = TrendProcess(regions=4, topics=8, seed=103, device="cpu")
    right = TrendProcess(regions=4, topics=8, seed=103, device="cpu")
    left.advance(7)
    right.advance(7)
    torch.testing.assert_close(left.snapshot(), right.snapshot())
    event = make_app_events(
        EventType.SHARE,
        event_time=7,
        request_id=torch.tensor([11]),
        user_id=torch.tensor([3]),
        surface=torch.tensor([0]),
        item_id=torch.tensor([5]),
        region=torch.tensor([2]),
        topic_id=torch.tensor([6]),
    )
    left.commit(event)
    left.advance(8)
    right.advance(8)
    assert left.snapshot()[2, 6] > right.snapshot()[2, 6]


def test_session_survival_can_create_churn_without_observable_churn_label():
    catalog = build_public_catalog(
        items=200,
        creators=40,
        merchants=20,
        topics=8,
        countries=4,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=101,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=2_000,
        topics=8,
        embedding_dim=8,
        countries=4,
        regions_per_country=3,
        environment_seed=103,
        future_signup_fraction=0.0,
    ), catalog)
    users = world.users
    users.active.fill_(True)
    users.satisfaction.zero_()
    users.fatigue.fill_(1.0)
    users.habit.fill_(0.01)
    users.churn_susceptibility.fill_(1.0)
    users.session_count.fill_(1)
    event = make_app_events(
        EventType.SESSION_END,
        event_time=10,
        request_id=users.user_id + 1,
        user_id=users.user_id,
        surface=torch.zeros_like(users.user_id),
    )
    world.commit(event)
    churn_rate = float(users.churned.float().mean())
    assert 0.65 < churn_rate < 0.85
    assert not hasattr(event, "churned")


def test_neural_feed_authority_is_request_keyed_and_keeps_hidden_inputs_private():
    catalog = build_public_catalog(
        items=240,
        creators=40,
        merchants=20,
        topics=8,
        countries=4,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=101,
        device="cpu",
    )
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32,
        latent_dim=8,
        attention_heads=4,
        ensemble_members=2,
        batch_size=8,
        epochs=1,
    ))
    authority = NeuralFeedResponseAuthority(
        ensemble,
        member_index=0,
        artifact_sha256="a" * 64,
        feature_contract_sha256=V4_FEATURE_CONTRACT["sha256"],
        feature_coverage=V4_FEATURE_COVERAGE,
        support_profile=_wide_support_profile(),
        inference_batch_size=3,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=8,
        topics=8,
        embedding_dim=8,
        countries=4,
        regions_per_country=3,
        environment_seed=103,
        future_signup_fraction=0.0,
    ), catalog, response_authority=authority)
    user = torch.arange(8)
    position = torch.arange(5)[None].expand(8, -1)
    item = torch.remainder(user[:, None] * 17 + position * 29, 240)
    slate = RenderedSlateBatch(
        request_id=user + 1,
        user_id=user,
        surface=torch.zeros_like(user),
        event_time=torch.zeros_like(user),
        item_ids=item,
        positions=position,
        valid=torch.ones_like(item, dtype=torch.bool),
        ui_variant=torch.zeros_like(user),
        exposure_probability=torch.ones_like(item, dtype=torch.float),
        selection_policy_kind=torch.zeros_like(user),
        exploration_rate=torch.zeros_like(user, dtype=torch.float),
        slate_log_probability=torch.zeros_like(user, dtype=torch.float),
        assignment_probability=torch.ones_like(user, dtype=torch.float),
    )
    snapshot = world.snapshot()
    first = authority.respond(snapshot, catalog, slate, 103)
    second = authority.respond(snapshot, catalog, slate, 103)
    assert torch.equal(first.event_id, second.event_id)
    assert torch.equal(first.event_type, second.event_type)
    assert first.event(EventType.IMPRESSION).sum() == 40
    assert "neural-feed" in world.manifest()["response"]
    assert not hasattr(slate, "selected_features")


def test_neural_feed_authority_rejects_training_serving_feature_skew():
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32, latent_dim=8, attention_heads=4,
        ensemble_members=2, batch_size=8, epochs=1,
    ))
    try:
        NeuralFeedResponseAuthority(
            ensemble, member_index=0, artifact_sha256="a" * 64,
            feature_contract_sha256="b" * 64,
            feature_coverage=V4_FEATURE_COVERAGE,
            support_profile=_wide_support_profile(),
        )
    except ValueError as error:
        assert "feature contract" in str(error)
    else:
        raise AssertionError("feature-contract skew must fail closed")

    try:
        NeuralFeedResponseAuthority(
            ensemble, member_index=0, artifact_sha256="a" * 64,
            feature_contract_sha256=V4_FEATURE_CONTRACT["sha256"],
            feature_coverage=KUAI_FEATURE_COVERAGE,
            support_profile=_wide_support_profile(),
        )
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("incomplete source coverage must fail closed")

    try:
        NeuralFeedResponseAuthority(
            ensemble, member_index=0, artifact_sha256="a" * 64,
            feature_contract_sha256=V4_FEATURE_CONTRACT["sha256"],
            feature_coverage=V4_FEATURE_COVERAGE,
            support_profile={},
        )
    except ValueError as error:
        assert "support profile" in str(error)
    else:
        raise AssertionError("missing support profile must fail closed")

    try:
        NeuralFeedResponseAuthority(
            ensemble, member_index=0, artifact_sha256="a" * 64,
            feature_contract_sha256=V4_FEATURE_CONTRACT["sha256"],
            feature_coverage=V4_FEATURE_COVERAGE,
            support_profile=_wide_support_profile(), inference_batch_size=0,
        )
    except ValueError as error:
        assert "batch size" in str(error)
    else:
        raise AssertionError("invalid inference batch size must fail closed")


def test_factual_response_artifact_is_content_bound_and_promoted():
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32, latent_dim=8, attention_heads=4,
        ensemble_members=2, batch_size=8, epochs=1,
    ))
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "artifact"
        save_world_ensemble(
            ensemble, [[], []], artifact,
            {
                "manifest_sha256": "dataset",
                "feature_contract_sha256": V4_FEATURE_CONTRACT["sha256"],
                "feature_coverage": V4_FEATURE_COVERAGE,
            },
            support_profile=_wide_support_profile(),
        )
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["authority_status"] = "accepted_feed_authority"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
        authority = load_factual_response_authority(
            FactualResponseArtifact(
                artifact_dir=str(artifact),
                manifest_sha256=manifest_hash,
                member_index=0,
            ),
            "cpu",
        )
        assert authority.artifact_sha256 == manifest_hash
        assert authority.manifest()["weights_sha256"] == manifest["weights_sha256"]

        try:
            load_factual_response_authority(
                FactualResponseArtifact(
                    artifact_dir=str(artifact),
                    manifest_sha256="0" * 64,
                    member_index=0,
                ),
                "cpu",
            )
        except ValueError as error:
            assert "manifest hash" in str(error)
        else:
            raise AssertionError("unregistered factual artifact must fail closed")

        (artifact / "world_model.pt").unlink()
        try:
            load_factual_response_authority(
                FactualResponseArtifact(
                    artifact_dir=str(artifact),
                    manifest_sha256=manifest_hash,
                    member_index=0,
                ),
                "cpu",
            )
        except ValueError as error:
            assert "weights" in str(error)
        else:
            raise AssertionError("missing factual weights must fail closed")


def test_neural_authority_shadow_replays_reference_cascade_without_committing():
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32, latent_dim=8, attention_heads=4,
        ensemble_members=2, batch_size=8, epochs=1,
    ))
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "artifact"
        save_world_ensemble(
            ensemble, [[], []], artifact,
            {
                "manifest_sha256": "dataset",
                "feature_contract_sha256": V4_FEATURE_CONTRACT["sha256"],
                "feature_coverage": V4_FEATURE_COVERAGE,
            },
            support_profile=_wide_support_profile(),
        )
        report = run_authority_shadow(
            artifact,
            AuthorityShadowConfig(
                users=200, items=900, ticks=2, topics=8,
                countries=4, regions_per_country=3, embedding_dim=8,
                inference_batch_size=17,
                device="cpu",
            ),
        )
    assert report["decision"] == "pass"
    assert all(report["gates"].values())
    assert report["first_run"]["trace_hash"] == report["second_run"]["trace_hash"]
    assert report["semantic_replay"]["factual"]["discrete_fields_exact"]
    assert report["semantic_replay"]["neural"]["discrete_fields_exact"]
    assert report["semantic_replay"]["neural"]["maximum_float_delta"] <= 1e-6
    assert report["semantic_replay"]["neural"]["duration_max_delta_ms"] <= 1
