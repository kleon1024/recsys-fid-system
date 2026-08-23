"""Cluster-robust uncertainty shared by OPE and stateful shadow."""

from __future__ import annotations

import numpy as np


def cluster_interval(delta: np.ndarray, clusters: np.ndarray) -> tuple:
    mean = float(delta.mean())
    unique, inverse = np.unique(clusters, return_inverse=True)
    centered = delta - mean
    cluster_sums = np.bincount(inverse, weights=centered)
    groups, rows = len(unique), len(delta)
    variance = groups / max(groups - 1, 1) * np.square(cluster_sums).sum() / rows**2
    standard_error = float(np.sqrt(variance))
    return mean, standard_error, [
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    ]
