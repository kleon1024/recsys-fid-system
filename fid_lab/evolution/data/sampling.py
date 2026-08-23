"""Probability-carrying negative sampling for retrieval training."""

from __future__ import annotations

import numpy as np

from .contracts import NegativeSample


SOURCE_FRACTIONS = {"in_batch": 0.60, "hard": 0.25, "random": 0.15}
SOURCE_ORDER = ("in_batch", "hard", "random")


def negative_source_counts(total: int) -> dict[str, int]:
    if total <= 0 or total % 20:
        raise ValueError("negative count must be a positive multiple of 20")
    counts = {
        source: int(total * SOURCE_FRACTIONS[source]) for source in SOURCE_ORDER
    }
    if sum(counts.values()) != total:
        raise ValueError("negative source fractions must close to the requested total")
    return counts


def expected_sampling_counts(
    probabilities: np.ndarray, total: int
) -> np.ndarray:
    """Convert conditional draw probability q(i|source) to expected count n_s*q."""
    counts = negative_source_counts(total)
    source_counts = np.concatenate(
        tuple(
            np.full(counts[source], counts[source], dtype=np.float32)
            for source in SOURCE_ORDER
        )
    )
    if probabilities.shape[-1] != total:
        raise ValueError("probability width must equal negatives per query")
    return np.clip(probabilities, 1e-8, 1.0) * source_counts


def _take(
    rng: np.random.Generator,
    pool: tuple[int, ...],
    count: int,
    source: str,
) -> list[NegativeSample]:
    if not pool or count == 0:
        return []
    replace = len(pool) < count
    selected = rng.choice(np.asarray(pool), size=count, replace=replace)
    probability = 1.0 / len(pool)
    return [NegativeSample(int(item), source, probability) for item in selected]


def mixed_negative_sample(
    in_batch: tuple[int, ...],
    hard: tuple[int, ...],
    random: tuple[int, ...],
    total: int,
    seed: int,
) -> tuple[NegativeSample, ...]:
    rng = np.random.default_rng(seed)
    counts = negative_source_counts(total)
    samples = []
    samples.extend(_take(rng, in_batch, counts["in_batch"], "in_batch"))
    samples.extend(_take(rng, hard, counts["hard"], "hard"))
    samples.extend(_take(rng, random, counts["random"], "random"))
    if len(samples) != total:
        raise ValueError("every negative pool must be non-empty")
    return tuple(samples)
