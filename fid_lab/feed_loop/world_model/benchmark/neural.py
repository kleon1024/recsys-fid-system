"""Candidate-conditioned DIN and slate Transformer challengers."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from ..data import WorldModelSplit


class DINRequestRanker(nn.Module):
    def __init__(self, feature_dim=28, sequence_dim=8, width=64) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.SiLU()
        )
        self.history = nn.Sequential(
            nn.LayerNorm(sequence_dim), nn.Linear(sequence_dim, width), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, slate, sequence):
        candidate = self.candidate(slate)
        history = self.history(sequence)
        attention = torch.einsum("bkd,bld->bkl", candidate, history) / math.sqrt(
            candidate.shape[-1]
        )
        mask = sequence.abs().sum(dim=2) > 0
        attention = attention.masked_fill(~mask[:, None], -1e4).softmax(dim=2)
        interest = torch.einsum("bkl,bld->bkd", attention, history)
        return self.head(
            torch.cat((candidate, interest, candidate * interest), dim=2)
        ).squeeze(2)


class SlateTransformerRanker(nn.Module):
    def __init__(self, feature_dim=28, sequence_dim=8, width=64) -> None:
        super().__init__()
        self.candidate = nn.Linear(feature_dim, width)
        self.history = nn.Linear(sequence_dim, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0.05, batch_first=True, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(layer, 2)
        self.slate_encoder = nn.TransformerEncoder(deepcopy(layer), 2)
        self.head = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, slate, sequence):
        history_mask = sequence.abs().sum(dim=2) == 0
        all_empty = history_mask.all(dim=1)
        history_mask = history_mask.clone()
        history_mask[all_empty, 0] = False
        history = self.sequence_encoder(
            self.history(sequence), src_key_padding_mask=history_mask
        )
        valid = (~history_mask).float()
        pooled = (history * valid[:, :, None]).sum(dim=1)
        pooled /= valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        candidate = self.slate_encoder(self.candidate(slate))
        context = pooled[:, None].expand(-1, candidate.shape[1], -1)
        return self.head(
            torch.cat((candidate, context, candidate * context), dim=2)
        ).squeeze(2)


def _selected_loss(model, batch, labels, weights):
    logits = model(batch["slate_features"], batch["sequence"])
    rows = torch.arange(len(logits), device=logits.device)
    selected = logits[rows, batch["exposed_index"]]
    losses = nn.functional.binary_cross_entropy_with_logits(
        selected, labels, reduction="none"
    )
    return (losses * weights).mean()


def _validation_loss(model, split, labels, device, batch_size):
    model.eval()
    total = 0.0
    batches = 0
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(split)))
            batch = split.batch(indices, device)
            target = torch.as_tensor(labels[indices], device=device)
            total += float(_selected_loss(model, batch, target, batch["weights"]))
            batches += 1
    return total / max(batches, 1)


def fit_request_ranker(model, train, train_labels, validation, validation_labels,
                       device, epochs, seed, batch_size=1_024):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        losses = []
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]
            batch = train.batch(indices, device)
            labels = torch.as_tensor(train_labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = _selected_loss(model, batch, labels, batch["weights"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = _validation_loss(
            model, validation, validation_labels, device, batch_size
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.inference_mode()
def predict_request_ranker(model, split: WorldModelSplit, device, rows=None,
                           batch_size=1_024):
    model.eval()
    count = len(split) if rows is None else min(rows, len(split))
    output = []
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(start + batch_size, count))
        batch = split.batch(indices, device)
        output.append(torch.sigmoid(
            model(batch["slate_features"], batch["sequence"])
        ).float().cpu())
    return torch.cat(output).numpy()
