from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import pytest
import torch

from fid_lab.simulation.digital_twin.world.training.storage import (
    load_or_create_build_state,
    write_family_part,
)


@dataclass(frozen=True)
class _Config:
    rows: int = 4


def _write_part(root, state, value):
    tensors = {"user_id": torch.tensor([value])}
    write_family_part(
        root, state, family_id=1, split="train", tensors=tensors,
        paired=None, family={"rows": 1},
    )


def test_family_partition_and_build_state_survive_interrupted_rewrite(
    tmp_path, monkeypatch,
):
    root = tmp_path / "bridge"
    state = load_or_create_build_state(root, _Config(), ((1, "train", 1),))
    _write_part(root, state, 7)
    part = root / "parts" / "train-family-1.pt"
    original_part = part.read_bytes()
    original_state = json.loads((root / "build-state.json").read_text())

    def interrupted_save(payload, path):
        path.write_bytes(b"partial")
        raise OSError("simulated storage interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="storage interruption"):
        _write_part(root, state, 9)

    assert part.read_bytes() == original_part
    assert sha256(part.read_bytes()).hexdigest() == (
        original_state["completed"]["train:family-1"]["sha256"]
    )
    assert json.loads((root / "build-state.json").read_text()) == original_state
    assert not (root / "parts" / "train-family-1.pt.tmp").exists()
