"""Versioned retrieval model artifact and device-resident serving index."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch

from .architectures import build_retriever


@dataclass
class RetrievalBundle:
    name: str
    model: torch.nn.Module
    query_mean: torch.Tensor
    query_scale: torch.Tensor
    item_mean: torch.Tensor
    item_scale: torch.Tensor
    corpus_sha256: str
    offline: dict
    item_embeddings: torch.Tensor | None = None

    def encode_query(self, query):
        return self.model.encode_query((query - self.query_mean) / self.query_scale)

    def index(self, item_features, chunk=100_000):
        values = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(item_features), chunk):
                item = item_features[start:start + chunk]
                values.append(self.model.encode_item(
                    (item - self.item_mean) / self.item_scale
                ))
        self.item_embeddings = torch.cat(values)
        return self

    def score_pool(self, query, item_ids):
        if self.item_embeddings is None:
            raise RuntimeError("retrieval bundle must index the serving corpus")
        normalized = (query - self.query_mean) / self.query_scale
        return self.model.pool_scores(normalized, self.item_embeddings[item_ids])


def corpus_hash(item_features):
    values = item_features.detach().to("cpu", torch.float32).contiguous().numpy()
    return sha256(values.tobytes()).hexdigest()


def save_bundle(bundle, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "poi-retrieval-v4-v1",
        "name": bundle.name,
        "state_dict": bundle.model.state_dict(),
        "query_mean": bundle.query_mean.cpu(),
        "query_scale": bundle.query_scale.cpu(),
        "item_mean": bundle.item_mean.cpu(),
        "item_scale": bundle.item_scale.cpu(),
        "corpus_sha256": bundle.corpus_sha256,
        "offline": bundle.offline,
    }, path)
    return {
        "artifact_file": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def load_bundle(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "poi-retrieval-v4-v1":
        raise ValueError("unsupported POI retrieval artifact")
    model = build_retriever(payload["name"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return RetrievalBundle(
        payload["name"], model,
        payload["query_mean"].to(device), payload["query_scale"].to(device),
        payload["item_mean"].to(device), payload["item_scale"].to(device),
        payload["corpus_sha256"], payload["offline"],
    )
