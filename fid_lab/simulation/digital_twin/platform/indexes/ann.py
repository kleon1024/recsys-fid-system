"""FAISS HNSW and exact Torch item indexes behind one serving contract."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch

from ...catalog import PublicCatalog
from .contracts import RetrievalConfig


class FaissItemIndex:
    def __init__(self, catalog: PublicCatalog, config: RetrievalConfig):
        self.catalog = catalog
        self.config = config
        self.version = "unbuilt"
        self.backend = os.environ.get(
            "FID_ANN_BACKEND", "torch" if sys.platform == "darwin" else "faiss"
        )
        if self.backend not in {"faiss", "torch"}:
            raise ValueError("FID_ANN_BACKEND must be faiss or torch")
        self._index: object | None = None
        self._torch_item: torch.Tensor | None = None
        self._torch_embedding: torch.Tensor | None = None
        self._indexed_active = torch.zeros_like(catalog.active)

    def sync(self, active: torch.Tensor, version: str) -> None:
        if self.backend == "torch":
            self._torch_item = self.catalog.item_id[active]
            self._torch_embedding = self.catalog.content_embedding[active]
            self._indexed_active = active.clone()
        else:
            faiss = importlib.import_module("faiss")
            faiss.omp_set_num_threads(int(os.environ.get("FID_FAISS_THREADS", "1")))
            if (self._indexed_active & ~active).any():
                self._index = None
                self._indexed_active.zero_()
            new_item = active & ~self._indexed_active
            item = self.catalog.item_id[new_item].detach().cpu().numpy().astype("int64")
            vectors = self.catalog.content_embedding[new_item].detach().cpu().numpy()
            vectors = np.ascontiguousarray(vectors.astype("float32"))
            if self._index is None:
                base = faiss.IndexHNSWFlat(
                    self.catalog.content_embedding.shape[1],
                    self.config.hnsw_neighbors,
                    faiss.METRIC_INNER_PRODUCT,
                )
                base.hnsw.efConstruction = max(2 * self.config.hnsw_neighbors, 40)
                base.hnsw.efSearch = self.config.hnsw_ef_search
                self._index = faiss.IndexIDMap2(base)
            if len(item):
                self._index.add_with_ids(vectors, item)
                self._indexed_active |= new_item
        self.version = version

    def search(self, query: torch.Tensor, limit: int):
        if self.backend == "torch":
            if self._torch_embedding is None or self._torch_item is None:
                raise ValueError("Torch item index has not been built")
            count = min(limit, len(self._torch_item))
            score, location = torch.topk(
                query @ self._torch_embedding.T, count, dim=1,
            )
            return self._torch_item[location], score
        if self._index is None:
            raise ValueError("FAISS item index has not been built")
        query_np = np.ascontiguousarray(
            query.detach().cpu().numpy().astype("float32")
        )
        score, item = self._index.search(query_np, limit)
        return (
            torch.from_numpy(item).to(query.device),
            torch.from_numpy(score).to(query.device),
        )
