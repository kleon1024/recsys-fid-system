"""Two-Tower retrieval trained from actual search exposure negatives."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

from ..simulation.retrieval import item_tower_features, query_tower_features


class SearchTwoTower(nn.Module):
    def __init__(self, query_width, item_width, embedding_dim=16):
        super().__init__()
        self.query_width = query_width
        self.item_width = item_width
        self.embedding_dim = embedding_dim
        self.query = nn.Sequential(
            nn.Linear(query_width, 48), nn.ReLU(), nn.Linear(48, embedding_dim)
        )
        self.item = nn.Sequential(
            nn.Linear(item_width, 48), nn.ReLU(), nn.Linear(48, embedding_dim)
        )

    def encode_query(self, features):
        return functional.normalize(self.query(features), dim=-1)

    def encode_item(self, features):
        return functional.normalize(self.item(features), dim=-1)


@dataclass(frozen=True)
class RetrievalBundle:
    model: SearchTwoTower
    offline: dict[str, float]


def _positive_negative_pairs(world, response):
    examples = response["examples"]
    clicked = response["clicked"]
    exposed = examples.poi_ids.gather(1, examples.exposed_indices)
    selected = response["selected_poi"]
    selected_rank = (exposed == selected[:, None]).long().argmax(1)
    negative_rank = torch.remainder(
        selected_rank + 1 + world.requests.request_id, exposed.shape[1]
    )
    negative = exposed.gather(1, negative_rank[:, None]).squeeze(1)
    return torch.where(clicked)[0], selected[clicked], negative[clicked]


def train_retriever(config, world, response):
    query_features = query_tower_features(world)
    request, positive, negative = _positive_negative_pairs(world, response)
    split = int(len(request) * 0.85)
    if split < 100:
        raise ValueError("insufficient clicked searches for Two-Tower training")
    torch.manual_seed(config.seed + 700)
    model = SearchTwoTower(
        query_features.shape[1],
        item_tower_features(world, positive[:1]).shape[1],
        config.semantic_dim,
    ).to(query_features.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    generator = torch.Generator(device=query_features.device).manual_seed(
        config.seed + 701
    )
    losses = []
    for _ in range(config.train_epochs):
        order = torch.randperm(split, generator=generator, device=request.device)
        for start in range(0, split, config.train_batch_requests):
            index = order[start : start + config.train_batch_requests]
            query = model.encode_query(query_features[request[index]])
            positive_item = model.encode_item(item_tower_features(
                world, positive[index]
            ))
            negative_item = model.encode_item(item_tower_features(
                world, negative[index]
            ))
            margin = (query * positive_item).sum(1) - (
                query * negative_item
            ).sum(1)
            loss = functional.softplus(-margin).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    with torch.inference_mode():
        query = model.encode_query(query_features[request[split:]])
        positive_score = (query * model.encode_item(item_tower_features(
            world, positive[split:]
        ))).sum(1)
        negative_score = (query * model.encode_item(item_tower_features(
            world, negative[split:]
        ))).sum(1)
    return RetrievalBundle(model, {
        "training_pairs": int(split),
        "validation_pairs": int(len(request) - split),
        "pair_accuracy": float((positive_score > negative_score).float().mean()),
        "mean_margin": float((positive_score - negative_score).mean()),
        "final_loss": float(sum(losses[-20:]) / min(len(losses), 20)),
    })


def save_retriever(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = bundle.model
    torch.save({
        "schema": "local-search-two-tower-v1",
        "query_width": model.query_width,
        "item_width": model.item_width,
        "embedding_dim": model.embedding_dim,
        "state_dict": model.state_dict(),
        "offline": bundle.offline,
    }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_retriever(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "local-search-two-tower-v1":
        raise ValueError("unsupported Local Search retrieval artifact")
    model = SearchTwoTower(
        payload["query_width"], payload["item_width"], payload["embedding_dim"]
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return RetrievalBundle(model, payload["offline"])
