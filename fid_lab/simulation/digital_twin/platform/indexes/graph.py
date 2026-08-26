"""Sparse co-visit graph rebuilt outside the request path."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import torch

from ...contracts import AppEventBatch, EventType, Surface


STRONG_GRAPH_EVENTS = (
    EventType.LONG_VIEW,
    EventType.COMPLETE,
    EventType.LIKE,
    EventType.COMMENT,
    EventType.SHARE,
    EventType.FOLLOW,
    EventType.FAVORITE,
)


class CoVisitGraphIndex:
    def __init__(self, items: int, neighbors: int, device: torch.device):
        self.items = items
        self.neighbor_count = neighbors
        self.device = device
        self._matrix = scipy.sparse.csr_matrix((items, items), dtype=np.float32)
        self._pending_source: list[np.ndarray] = []
        self._pending_target: list[np.ndarray] = []
        self._last_item_by_user: dict[int, int] = {}
        self.neighbor = torch.full(
            (items, neighbors), -1, device=device, dtype=torch.long,
        )
        self.score = torch.zeros(items, neighbors, device=device)
        self.version = "graph-empty"

    def update(self, events: AppEventBatch) -> None:
        strong = torch.zeros_like(events.event_type, dtype=torch.bool)
        for event_type in STRONG_GRAPH_EVENTS:
            strong |= events.event(event_type)
        selected = (
            strong
            & (events.surface == int(Surface.FEED))
            & (events.user_id >= 0)
            & (events.item_id >= 0)
        )
        if not selected.any():
            return
        user = events.user_id[selected].cpu().numpy()
        item = events.item_id[selected].cpu().numpy()
        request = events.request_id[selected].cpu().numpy()
        event_time = events.event_time[selected].cpu().numpy()
        position = events.position[selected].cpu().numpy()
        order = np.lexsort((position, request, event_time, user))
        user, item, request = user[order], item[order], request[order]
        unique = np.ones(len(user), dtype=np.bool_)
        unique[1:] = (
            (user[1:] != user[:-1])
            | (request[1:] != request[:-1])
            | (item[1:] != item[:-1])
        )
        user, item = user[unique], item[unique]
        source: list[int] = []
        target: list[int] = []
        for current_user, current_item in zip(user.tolist(), item.tolist()):
            previous = self._last_item_by_user.get(current_user)
            if previous is not None and previous != current_item:
                source.append(previous)
                target.append(current_item)
            self._last_item_by_user[current_user] = current_item
        if not source:
            return
        source_array = np.asarray(source, dtype=np.int64)
        target_array = np.asarray(target, dtype=np.int64)
        self._pending_source.append(np.concatenate((source_array, target_array)))
        self._pending_target.append(np.concatenate((target_array, source_array)))

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
