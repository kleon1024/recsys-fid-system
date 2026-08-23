"""Chunk-free vectorized generator for a sparse POI-anchor Feed vertical."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .contracts import FEED_TASKS, ScaleConfig


@dataclass(frozen=True)
class ScaleDataset:
    config: ScaleConfig
    viewer_ids: np.ndarray
    author_ids: np.ndarray
    video_ids: np.ndarray
    poi_ids: np.ndarray
    dense_features: np.ndarray
    sparse_ids: np.ndarray
    history_item_ids: np.ndarray
    sequences: np.ndarray
    sequence_mask: np.ndarray
    labels: np.ndarray
    label_probabilities: np.ndarray
    label_masks: np.ndarray
    sample_weight: np.ndarray
    served_scores: np.ndarray
    diagnostic_features: np.ndarray

    @property
    def examples(self) -> int:
        return int(self.labels.shape[0])


def _long_tail_ids(rng: np.random.Generator, size: int, cardinality: int) -> np.ndarray:
    ranks = np.minimum(rng.zipf(1.22, size=size), cardinality) - 1
    permutation = rng.permutation(cardinality)
    return permutation[ranks].astype(np.int64)


def _scaled_probability(
    base: float,
    signal: np.ndarray,
    ceiling: float = 0.95,
) -> np.ndarray:
    multiplier = np.exp(0.45 * signal)
    multiplier /= float(multiplier.mean())
    return np.clip(base * multiplier, 0.0, ceiling)


def _behavior_signals(
    config: ScaleConfig,
    dense: np.ndarray,
    sequences: np.ndarray,
    sequence_mask: np.ndarray,
    sequence_match: np.ndarray,
    viewer_ids: np.ndarray,
    video_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cross_signal = dense[:, 2] * dense[:, 5]
    if config.signal_version == "industrial-cross-sequence-v1":
        intent = 0.9 * dense[:, 2] + 0.5 * dense[:, 5] + 1.50 * cross_signal
        quality = 0.8 * dense[:, 0] + 0.6 * dense[:, 4] - 0.4 * dense[:, 9]
        quality += 1.20 * sequence_match
        diagnostics = np.stack(
            [cross_signal, sequence_match.astype(np.float32)], axis=1
        )
        return intent, quality, diagnostics.astype(np.float32)
    recency_weight = np.linspace(0.1, 1.0, sequences.shape[1], dtype=np.float32)
    denominator = (sequence_mask * recency_weight).sum(axis=1).clip(min=1e-6)
    sequence_recency = (
        sequences[:, :, 0] * sequence_mask * recency_weight
    ).sum(axis=1) / denominator
    segment_match = 1.0 - np.abs((viewer_ids % 7) - (video_ids % 7)) / 6.0
    threshold_cross = ((dense[:, 2] > 0.15) & (dense[:, 5] > -0.10)).astype(
        np.float32
    )
    periodic_cross = np.sin(dense[:, 3] * dense[:, 7])
    intent = (
        0.35 * dense[:, 2]
        + 0.25 * dense[:, 5]
        + 1.10 * np.tanh(1.4 * cross_signal)
        + 0.70 * threshold_cross
        + 0.45 * periodic_cross
        + 0.60 * segment_match
    )
    quality = (
        0.50 * dense[:, 0]
        + 0.35 * dense[:, 4]
        - 0.35 * dense[:, 9]
        + 0.75 * sequence_match
        + 0.55 * np.tanh(sequence_recency)
        + 0.40 * segment_match
    )
    diagnostics = np.stack(
        [cross_signal, sequence_match.astype(np.float32)], axis=1
    )
    return intent, quality, diagnostics.astype(np.float32)


def build_scale_dataset(config: ScaleConfig = ScaleConfig()) -> ScaleDataset:
    rng = np.random.default_rng(config.seed)
    anchored = int(rng.binomial(config.main_impressions, config.anchor_rate))
    viewer_ids = _long_tail_ids(rng, anchored, config.viewers)
    author_ids = _long_tail_ids(rng, anchored, config.authors)
    video_ids = _long_tail_ids(rng, anchored, config.videos)
    poi_ids = _long_tail_ids(rng, anchored, config.pois)
    dense = rng.normal(size=(anchored, 10)).astype(np.float32)
    history_length = rng.integers(1, 25, size=anchored)
    sequence_mask = np.arange(24)[None, :] >= (24 - history_length[:, None])
    sequences = rng.normal(0.0, 0.6, size=(anchored, 24, 8)).astype(np.float32)
    sequences[:, :, 0] += dense[:, 6, None]
    sequences *= sequence_mask[:, :, None]
    history_item_ids = rng.integers(4_096, size=(anchored, 24), dtype=np.int64)
    history_item_ids *= sequence_mask
    sequence_match = rng.random(anchored) < (1.0 / (1.0 + np.exp(-dense[:, 6])))
    history_item_ids[sequence_match, -1] = video_ids[sequence_match] % 4_096
    intent, quality, diagnostic_features = _behavior_signals(
        config,
        dense,
        sequences,
        sequence_mask,
        sequence_match,
        viewer_ids,
        video_ids,
    )
    labels = np.zeros((anchored, len(FEED_TASKS)), dtype=np.float32)
    probabilities = np.zeros_like(labels)
    long_view_probability = _scaled_probability(
        config.long_view_rate, 0.6 * quality + 0.4 * intent, 0.75
    )
    anchor_probability = _scaled_probability(config.anchor_click_rate, intent, 0.12)
    detail_conditional = _scaled_probability(config.detail_given_click, intent, 0.55)
    favorite_conditional = _scaled_probability(config.favorite_given_detail, quality, 0.35)
    order_conditional = _scaled_probability(
        config.order_given_detail, intent + quality, 0.12
    )
    negative_probability = _scaled_probability(
        config.negative_feedback_rate, dense[:, 9] - dense[:, 2], 0.08
    )
    long_view = rng.random(anchored) < long_view_probability
    anchor_click = rng.random(anchored) < anchor_probability
    detail = anchor_click & (
        rng.random(anchored) < detail_conditional
    )
    favorite = detail & (
        rng.random(anchored) < favorite_conditional
    )
    order = detail & (
        rng.random(anchored) < order_conditional
    )
    negative = rng.random(anchored) < negative_probability
    outcomes = (long_view, anchor_click, detail, favorite, order, negative)
    marginal_probabilities = (
        long_view_probability,
        anchor_probability,
        anchor_probability * detail_conditional,
        anchor_probability * detail_conditional * favorite_conditional,
        anchor_probability * detail_conditional * order_conditional,
        negative_probability,
    )
    for index, outcome in enumerate(outcomes):
        labels[:, index] = outcome
        probabilities[:, index] = marginal_probabilities[index]
    sparse_ids = np.stack(
        [viewer_ids, author_ids, video_ids, poi_ids, video_ids % 64, poi_ids % 128], axis=1
    )
    propensity = np.clip(0.06 + 0.18 / (1.0 + np.exp(-dense[:, 0])), 0.02, 0.5)
    sample_weight = np.minimum(1.0 / propensity, 10.0).astype(np.float32)
    label_masks = np.ones_like(labels, dtype=np.bool_)
    served_scores = np.stack(
        [dense[:, 0], intent, 0.5 * quality + 0.5 * intent, quality],
        axis=1,
    ).astype(np.float32)
    return ScaleDataset(
        config,
        viewer_ids,
        author_ids,
        video_ids,
        poi_ids,
        dense,
        sparse_ids,
        history_item_ids,
        sequences,
        sequence_mask,
        labels,
        probabilities,
        label_masks,
        sample_weight,
        served_scores,
        diagnostic_features,
    )


def _top_share(values: np.ndarray, fraction: float = 0.01) -> float:
    _, counts = np.unique(values, return_counts=True)
    take = max(1, int(np.ceil(len(counts) * fraction)))
    return float(np.sort(counts)[-take:].sum() / counts.sum())


def _gini(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    ordered = np.sort(counts.astype(np.float64))
    index = np.arange(1, len(ordered) + 1)
    return float((2.0 * np.sum(index * ordered) / np.sum(ordered) - len(ordered) - 1) / len(ordered))


def summarize_distribution(dataset: ScaleDataset) -> dict[str, object]:
    label_report: dict[str, dict[str, float | int]] = {}
    for index, task in enumerate(FEED_TASKS):
        positives = int(dataset.labels[:, index].sum())
        rate = positives / max(dataset.examples, 1)
        standard_error = float(np.sqrt(rate * (1.0 - rate) / max(dataset.examples, 1)))
        label_report[task] = {
            "positives": positives,
            "rate": rate,
            "standard_error": standard_error,
        }
    return {
        "assumption": asdict(dataset.config),
        "main_impressions": dataset.config.main_impressions,
        "anchored_examples": dataset.examples,
        "realized_anchor_rate": dataset.examples / dataset.config.main_impressions,
        "unique": {
            "viewers": int(np.unique(dataset.viewer_ids).size),
            "authors": int(np.unique(dataset.author_ids).size),
            "videos": int(np.unique(dataset.video_ids).size),
            "pois": int(np.unique(dataset.poi_ids).size),
        },
        "concentration": {
            "viewer_top_1pct_share": _top_share(dataset.viewer_ids),
            "author_top_1pct_share": _top_share(dataset.author_ids),
            "viewer_gini": _gini(dataset.viewer_ids),
            "author_gini": _gini(dataset.author_ids),
        },
        "labels": label_report,
    }
