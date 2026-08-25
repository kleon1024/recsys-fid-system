from __future__ import annotations

import pytest

from fid_lab.simulation.digital_twin.assets import (
    DIGITAL_TWIN_ASSETS,
    AssetGraph,
    AssetSpec,
)


def test_default_asset_graph_declares_training_and_factual_closure():
    order = DIGITAL_TWIN_ASSETS.order()
    assert order.index("events.observable") < order.index("samples.fine")
    assert order.index("events.pending_delivery") < order.index(
        "events.observable"
    )
    assert order.index("samples.fine") < order.index("models.candidate")
    assert order.index("learning.sample_bus") < order.index(
        "learning.active_lane"
    )
    assert order.index("learning.sample_bus") < order.index(
        "learning.candidate_lane"
    )
    assert order.index("platform.feature_manifest") < order.index(
        "models.candidate"
    )
    assert order.index("events.observable") < order.index(
        "world.factual_successor"
    )
    successor = DIGITAL_TWIN_ASSETS.spec("world.factual_successor")
    assert successor.inputs == ("events.observable",)
    assert "release.decision" not in successor.inputs


def test_asset_graph_rejects_unknown_inputs_and_cycles():
    with pytest.raises(ValueError, match="unknown inputs"):
        AssetGraph((AssetSpec("output", ("missing",), "test"),))
    with pytest.raises(ValueError, match="cycle"):
        AssetGraph((
            AssetSpec("left", ("right",), "test"),
            AssetSpec("right", ("left",), "test"),
        ))
