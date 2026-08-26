from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.catalog import build_public_catalog
from fid_lab.simulation.digital_twin.world.state import (
    UserWorldConfig,
    build_hidden_catalog_truth,
    build_hidden_users,
    topic_prototypes,
)


def test_catalog_semantics_do_not_cycle_with_contiguous_item_ids():
    catalog = build_public_catalog(
        items=20_000,
        creators=1_000,
        merchants=200,
        topics=512,
        countries=12,
        regions_per_country=16,
        embedding_dim=32,
        platform_seed=810,
        device="cpu",
    )
    adjacent_increment = (
        catalog.topic_id[1:] - catalog.topic_id[:-1] == 1
    ).float().mean()

    assert int(catalog.topic_id.unique().numel()) == 512
    assert float(adjacent_increment) < 0.02
    assert 0.69 < float(
        (catalog.content_kind == 0).float().mean()
    ) < 0.75


def _semantic_fixture():
    catalog = build_public_catalog(
        items=20_000,
        creators=1_000,
        merchants=200,
        topics=512,
        countries=12,
        regions_per_country=16,
        embedding_dim=64,
        platform_seed=810,
        device="cpu",
    )
    config = UserWorldConfig(
        users=4_000,
        topics=512,
        embedding_dim=64,
        countries=12,
        regions_per_country=16,
        environment_seed=809,
    )
    return catalog, config


def test_public_semantics_retain_topic_signal_without_becoming_identity():
    catalog, _ = _semantic_fixture()
    prototype = topic_prototypes(catalog, 512)
    positive = torch.einsum(
        "id,id->i", catalog.content_embedding, prototype[catalog.topic_id],
    ).mean()
    negative_topic = torch.remainder(catalog.topic_id + 137, 512)
    negative = torch.einsum(
        "id,id->i", catalog.content_embedding, prototype[negative_topic],
    ).mean()

    assert 0.70 < float(positive) < 0.95
    assert float(positive - negative) > 0.65


def test_private_semantics_preserve_noisy_observable_signal():
    catalog, config = _semantic_fixture()
    truth = build_hidden_catalog_truth(catalog, config.environment_seed)
    public_private = torch.einsum(
        "id,id->i", catalog.content_embedding, truth.semantic_embedding,
    ).mean()
    users = build_hidden_users(config, catalog)
    prototype = topic_prototypes(catalog, config.topics)
    primary = torch.einsum(
        "ud,ud->u", users.long_interest, prototype[users.primary_topic],
    ).mean()
    unrelated = torch.einsum(
        "ud,ud->u",
        users.long_interest,
        prototype[torch.remainder(users.primary_topic + 137, config.topics)],
    ).mean()

    assert 0.80 < float(public_private) < 0.98
    assert float(primary - unrelated) > 0.55
