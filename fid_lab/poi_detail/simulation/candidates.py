"""Fixed Map/YMAL, product, and review candidate materialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DetailCandidates:
    entity_ids: torch.Tensor
    module_kind: torch.Tensor


def build_candidates(world):
    config, requests = world.config, world.requests
    rank = torch.arange(config.candidates_per_module, device=requests.request_id.device)
    entity_sets = []
    kind_sets = []
    for kind in range(len(world.catalogs)):
        ids = torch.remainder(
            requests.current_poi[:, None] * (17 + kind * 6)
            + requests.request_id[:, None] * (13 + kind * 4)
            + rank[None, :] * (29 + kind * 2),
            config.entities_per_module,
        )
        entity_sets.append(ids)
        kind_sets.append(torch.full_like(ids, kind))
    return DetailCandidates(
        torch.cat(entity_sets, dim=1), torch.cat(kind_sets, dim=1)
    )
