"""User-day hard-pair adaptation for one-action random-exposure logs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn


def mine_user_day_pairs(split, eligible, predictions):
    groups = defaultdict(list)
    for index in np.flatnonzero(eligible):
        groups[(int(split.user_ids[index]), int(split.dates[index]))].append(index)
    pairs = []
    for indices in groups.values():
        positive = [index for index in indices if split.labels[index, 1] > 0]
        negative = [index for index in indices if split.labels[index, 1] == 0]
        if not positive or not negative:
            continue
        hard = max(negative, key=lambda index: predictions[index, 1])
        pairs.extend((index, hard) for index in positive)
    return np.asarray(pairs, dtype=np.int64)


def _forward(model, split, indices, device):
    return model(
        split.sparse[indices].to(device), split.dense[indices].to(device),
        split.history_items[indices].to(device),
        split.history_feedback[indices].to(device),
    )


def _pair_score(logits):
    return (
        0.60 * logits[:, 1]
        + 0.20 * logits[:, 0]
        + 0.20 * logits[:, 7]
        - 0.10 * logits[:, 6]
    )


def fit_pairwise_randomized(model, split, pairs, device, seed, epochs=2,
                            batch_size=512, learning_rate=5e-5):
    if not len(pairs):
        raise ValueError("no user-day positive/negative pairs")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for _ in range(epochs):
        losses = []
        order = torch.randperm(len(pairs), generator=generator).numpy()
        for start in range(0, len(pairs), batch_size):
            batch = pairs[order[start:start + batch_size]]
            positive = _forward(model, split, batch[:, 0], device)
            negative = _forward(model, split, batch[:, 1], device)
            loss = nn.functional.softplus(
                -(_pair_score(positive) - _pair_score(negative))
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history
