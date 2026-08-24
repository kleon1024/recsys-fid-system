"""Positive queries and corrected negatives from the frozen request log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .features import logged_query_features


@dataclass(frozen=True)
class RetrievalExamples:
    queries: torch.Tensor
    positive_ids: torch.Tensor
    negative_ids: torch.Tensor
    negative_probability: torch.Tensor
    weights: torch.Tensor
    high_value: torch.Tensor

    def __len__(self):
        return len(self.queries)


def _positive_mask(payload):
    labels = payload["labels"]
    return (
        (labels[:, 5] > 0) | (labels[:, 6] > 0) | (labels[:, 7] > 0)
        | (labels[:, 9] > 0)
    )


def _hard_negatives(payload, selected_rows, positive_ids, random_ids):
    item_ids = payload["candidate_item_ids"][selected_rows].long()
    scores = payload["candidate_fine_scores"][selected_rows].float()
    valid = item_ids != positive_ids[:, None]
    scores = scores.masked_fill(~valid, -1e9)
    positions = torch.topk(scores, 5, dim=1).indices
    hard = item_ids.gather(1, positions)
    hard_valid = valid.gather(1, positions)
    return torch.where(hard_valid, hard, random_ids[:, :5]), valid.sum(1).clamp_min(1)


def load_retrieval_examples(dataset_dir: Path, split: str, corpus_ids, seed=20260824):
    payload = torch.load(
        dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False
    )["tensors"]
    mask = _positive_mask(payload)
    selected_rows = torch.nonzero(mask, as_tuple=False).flatten()
    exposed = payload["exposed_index"][selected_rows].long()
    positive = payload["candidate_item_ids"][selected_rows, exposed].long()
    generator = torch.Generator().manual_seed(seed)
    random_position = torch.randint(
        len(corpus_ids), (len(positive), 8), generator=generator
    )
    random_ids = corpus_ids[random_position]
    permutation = torch.randint(
        len(positive), (len(positive), 12), generator=generator
    )
    in_batch = positive[permutation]
    in_batch = torch.where(in_batch == positive[:, None], random_ids[:, :1], in_batch)
    hard, hard_pool_size = _hard_negatives(
        payload, selected_rows, positive, random_ids
    )
    negative = torch.cat((in_batch, hard, random_ids[:, 5:]), dim=1)
    probability = torch.cat((
        torch.full((len(positive), 12), 1.0 / max(len(positive) - 1, 1)),
        hard_pool_size.reciprocal().float()[:, None].expand(-1, 5),
        torch.full((len(positive), 3), 1.0 / len(corpus_ids)),
    ), dim=1)
    labels = payload["labels"][selected_rows]
    strength = (
        1.0 + labels[:, 6] + labels[:, 7] + 2.0 * labels[:, 9]
        + 3.0 * labels[:, 10] + 4.0 * labels[:, 11] + 5.0 * labels[:, 12]
    )
    propensity = payload["exposure_propensity"][selected_rows].clamp_min(1e-4)
    weights = (strength / propensity).clamp_max(50.0)
    weights /= weights.mean()
    return RetrievalExamples(
        logged_query_features(payload, selected_rows), positive, negative,
        probability.clamp_min(1e-8), weights, labels[:, 9] > 0,
    )
