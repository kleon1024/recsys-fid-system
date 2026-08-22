"""Probability-carrying negative sampling for retrieval training."""

from __future__ import annotations

import numpy as np

from .contracts import NegativeSample


SOURCE_FRACTIONS = {"in_batch": 0.60, "hard": 0.25, "random": 0.15}


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
    probability = min(count / len(pool), 1.0) * SOURCE_FRACTIONS[source]
    return [NegativeSample(int(item), source, probability) for item in selected]


def mixed_negative_sample(
    in_batch: tuple[int, ...],
    hard: tuple[int, ...],
    random: tuple[int, ...],
    total: int,
    seed: int,
) -> tuple[NegativeSample, ...]:
    if total <= 0 or total % 20:
        raise ValueError("negative count must be a positive multiple of 20")
    rng = np.random.default_rng(seed)
    counts = {
        "in_batch": int(total * SOURCE_FRACTIONS["in_batch"]),
        "hard": int(total * SOURCE_FRACTIONS["hard"]),
        "random": int(total * SOURCE_FRACTIONS["random"]),
    }
    samples = []
    samples.extend(_take(rng, in_batch, counts["in_batch"], "in_batch"))
    samples.extend(_take(rng, hard, counts["hard"], "hard"))
    samples.extend(_take(rng, random, counts["random"], "random"))
    if len(samples) != total:
        raise ValueError("every negative pool must be non-empty")
    return tuple(samples)
