from __future__ import annotations

import pytest
import torch

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    ObservableEventLog,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
)
from fid_lab.simulation.digital_twin.experiments import (
    LayeredExperimentPlan,
    PolicyLayer,
)
from fid_lab.simulation.digital_twin.platform import CascadePolicy



def _kernel(users=512, items=3_000):
    catalog = build_public_catalog(
        items=items,
        creators=150,
        merchants=60,
        advertisers=30,
        topics=16,
        countries=4,
        regions_per_country=6,
        embedding_dim=12,
        platform_seed=601,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=users,
        topics=16,
        embedding_dim=12,
        countries=4,
        regions_per_country=6,
        environment_seed=607,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=users, history_length=16),
        catalog,
        RetrievalConfig(route_k=8, merged_k=32, graph_neighbors=8),
        RankingConfig(coarse_k=16, fine_k=8, expose_k=4),
    )
    return AtomicSimulationKernel(
        world, platform,
        ObservableEventLog(allowed_lateness=world.max_reporting_lag),
    )


def _plan() -> LayeredExperimentPlan:
    return LayeredExperimentPlan(
        CascadePolicy(
            "active", 1, 1, 1,
            enabled_routes=("popular",),
        ),
        (
            PolicyLayer(
                "retrieval",
                101,
                {
                    "enabled_routes": ("popular", "geo"),
                    "recall_version_id": 2,
                },
                0.3,
                0.3,
            ),
            PolicyLayer(
                "fine",
                211,
                {"fine_version_id": 2, "sequence_weight": 0.32},
                0.3,
                0.3,
            ),
        ),
    )


def test_layers_are_stable_orthogonal_and_compile_one_policy_per_request():
    kernel = _kernel()
    first = kernel.step(0, _plan())
    trace = first.layer_assignment
    assert trace is not None
    assert trace.layer_names == ("retrieval", "fine")
    assert trace.cell_by_layer.shape == (first.rendered_requests, 2)
    assert torch.unique(trace.cell_by_layer, dim=0).shape[0] >= 4
    assert first.candidate_trace is not None
    retrieval_treatment = trace.cell_by_layer[:, 0] == 1
    candidate = first.candidate_trace
    assert (candidate.recall_version_id[retrieval_treatment] == 2).all()
    assert (candidate.recall_version_id[~retrieval_treatment] == 1).all()
    fine_treatment = trace.cell_by_layer[:, 1] == 1
    assert (candidate.fine_version_id[fine_treatment] == 2).all()
    assert (candidate.fine_version_id[~fine_treatment] == 1).all()


def test_two_layers_cannot_own_one_parameter():
    with pytest.raises(ValueError, match="owned by both"):
        LayeredExperimentPlan(
            CascadePolicy("active", 1, 1, 1),
            (
                PolicyLayer("fine-a", 1, {"fine_version_id": 2}, 0.2, 0.2),
                PolicyLayer("fine-b", 2, {"fine_version_id": 3}, 0.2, 0.2),
            ),
        )
