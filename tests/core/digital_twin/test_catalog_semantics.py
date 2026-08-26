from __future__ import annotations

from fid_lab.simulation.digital_twin.catalog import build_public_catalog


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
