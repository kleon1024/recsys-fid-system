"""Sparse co-visit graph rebuilt outside the request path."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import torch

from ...contracts import AppEventBatch, EventType, Surface


MINIMUM_GRAPH_DWELL_MS = 3_000


class CoVisitGraphIndex:
    def __init__(self, items: int, neighbors: int, device: torch.device):
        self.items = items
        self.neighbor_count = neighbors
        self.device = device
        self._matrix = scipy.sparse.csr_matrix((items, items), dtype=np.float32)
        self._pending_source: list[np.ndarray] = []
        self._pending_target: list[np.ndarray] = []
        self.neighbor = torch.full(
            (items, neighbors), -1, device=device, dtype=torch.long,
        )
        self.score = torch.zeros(items, neighbors, device=device)
        self.version = "graph-empty"

    def update(self, events: AppEventBatch) -> None:
        dwell = (
            events.event(EventType.DWELL)
            & (events.surface == int(Surface.FEED))
            & (events.item_id >= 0)
            & (events.duration_ms >= MINIMUM_GRAPH_DWELL_MS)
        )
        if int(dwell.sum()) < 2:
            return
        request = events.request_id[dwell]
        position = events.position[dwell]
        item = events.item_id[dwell]
        order = torch.argsort(position, stable=True)
        order = order[torch.argsort(request[order], stable=True)]
        request, item = request[order], item[order]
        adjacent = request[1:] == request[:-1]
        source, target = item[:-1][adjacent], item[1:][adjacent]
        valid = source != target
        source, target = source[valid], target[valid]
        if not len(source):
            return
        self._pending_source.append(torch.cat((source, target)).cpu().numpy())
        self._pending_target.append(torch.cat((target, source)).cpu().numpy())

    def refresh(self, version: str) -> None:
        if not self._pending_source:
            return
        source = np.concatenate(self._pending_source)
        target = np.concatenate(self._pending_target)
        increment = scipy.sparse.coo_matrix(
            (np.ones(len(source), dtype=np.float32), (source, target)),
            shape=(self.items, self.items),
        ).tocsr()
        self._matrix = self._matrix + increment
        self._pending_source.clear()
        self._pending_target.clear()
        for row in np.unique(source):
            start, end = self._matrix.indptr[row:row + 2]
            columns = self._matrix.indices[start:end]
            values = self._matrix.data[start:end]
            if not len(columns):
                continue
            keep = np.argsort(-values, kind="stable")[: self.neighbor_count]
            width = len(keep)
            self.neighbor[row].fill_(-1)
            self.score[row].zero_()
            self.neighbor[row, :width] = torch.from_numpy(
                columns[keep].astype("int64")
            ).to(self.device)
            self.score[row, :width] = torch.from_numpy(values[keep]).to(self.device)
        self.version = version
