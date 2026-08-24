"""Shared paired A/B statistics and repeated-seed launch decisions."""

from __future__ import annotations

from math import erfc, sqrt
from dataclasses import dataclass

import numpy as np
import torch

from ..simulation.experimentation.assignment import assign_binary_torch


@dataclass
class CreatorClusterAccumulator:
    """Streaming request outcomes reduced to one mean per creator."""

    metric_names: tuple[str, ...]
    creators: int
    device: torch.device

    def __post_init__(self):
        self.control_sum = torch.zeros(
            len(self.metric_names), self.creators,
            dtype=torch.float64, device=self.device,
        )
        self.treatment_sum = torch.zeros_like(self.control_sum)
        self.counts = torch.zeros(
            self.creators, dtype=torch.float64, device=self.device
        )

    def add(self, creator_ids, control, treatment):
        self.counts.scatter_add_(
            0, creator_ids,
            torch.ones_like(creator_ids, dtype=torch.float64),
        )
        for index, name in enumerate(self.metric_names):
            self.control_sum[index].scatter_add_(
                0, creator_ids, control[name].double()
            )
            self.treatment_sum[index].scatter_add_(
                0, creator_ids, treatment[name].double()
            )

    def report(self):
        valid = self.counts > 0
        control = self.control_sum[:, valid] / self.counts[valid]
        treatment = self.treatment_sum[:, valid] / self.counts[valid]
        creator_ids = torch.arange(self.creators, device=self.device)[valid]
        paired = {
            name: paired_metric(control[index], treatment[index])
            for index, name in enumerate(self.metric_names)
        }
        randomized = {
            name: randomized_cluster_means_metric(
                control[index], treatment[index], creator_ids
            )
            for index, name in enumerate(self.metric_names)
        }
        return paired, randomized, int(valid.sum())


def paired_metric(control, treatment):
    """Return a paired normal-approximation effect report."""
    delta = (treatment - control).double().cpu().numpy()
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / sqrt(len(delta)))
    control_mean = float(control.double().mean())
    treatment_mean = float(treatment.double().mean())
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "absolute_effect": mean,
        "relative_effect": (
            None if abs(control_mean) < 1e-12 else mean / abs(control_mean)
        ),
        "standard_error": standard_error,
        "confidence_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "p_value": erfc(abs(mean / max(standard_error, 1e-12)) / sqrt(2.0)),
    }


def _cluster_means(values, cluster_ids):
    cluster_ids = cluster_ids.long()
    unique_ids, inverse = torch.unique(
        cluster_ids, sorted=True, return_inverse=True
    )
    total = torch.zeros(len(unique_ids), dtype=torch.float64, device=values.device)
    count = torch.zeros_like(total)
    total.scatter_add_(0, inverse, values.double())
    count.scatter_add_(0, inverse, torch.ones_like(values, dtype=torch.float64))
    return unique_ids, total / count


def cluster_paired_metric(control, treatment, cluster_ids):
    """Estimate paired creator-level effects without request pseudo-replication."""
    unique_ids, control_cluster = _cluster_means(control, cluster_ids)
    treatment_ids, treatment_cluster = _cluster_means(treatment, cluster_ids)
    if not torch.equal(unique_ids, treatment_ids):
        raise ValueError("paired cluster ids do not align")
    report = paired_metric(control_cluster, treatment_cluster)
    report.update({
        "clusters": len(control_cluster),
        "estimator": "same_creator_cluster_paired_difference",
    })
    return report


def cluster_randomized_metric(control, treatment, cluster_ids):
    """Observable author A/B with stable creator assignment."""
    unique_ids, control_cluster = _cluster_means(control, cluster_ids)
    treatment_ids, treatment_cluster = _cluster_means(treatment, cluster_ids)
    if not torch.equal(unique_ids, treatment_ids):
        raise ValueError("randomized cluster ids do not align")
    assigned = assign_binary_torch(unique_ids)
    left = control_cluster[~assigned].cpu().numpy()
    right = treatment_cluster[assigned].cpu().numpy()
    effect = float(right.mean() - left.mean())
    standard_error = float(np.sqrt(
        left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
    ))
    return {
        "control_mean": float(left.mean()),
        "treatment_mean": float(right.mean()),
        "absolute_effect": effect,
        "standard_error": standard_error,
        "confidence_interval": [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ],
        "control_creators": len(left),
        "treatment_creators": len(right),
        "estimator": "creator_cluster_randomized_ab",
    }


def randomized_cluster_means_metric(control_cluster, treatment_cluster, cluster_ids):
    """Randomized A/B when callers already hold one mean per cluster."""
    assigned = assign_binary_torch(cluster_ids.long())
    left = control_cluster[~assigned].double().cpu().numpy()
    right = treatment_cluster[assigned].double().cpu().numpy()
    effect = float(right.mean() - left.mean())
    standard_error = float(np.sqrt(
        left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
    ))
    return {
        "control_mean": float(left.mean()),
        "treatment_mean": float(right.mean()),
        "absolute_effect": effect,
        "standard_error": standard_error,
        "confidence_interval": [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ],
        "control_clusters": len(left),
        "treatment_clusters": len(right),
        "estimator": "cluster_randomized_ab_from_means",
    }


def aggregate_launch_rows(rows, primary_metric, value_metric):
    """Require identical comparisons and stable wins across repeated seeds."""
    controls = {row["control"] for row in rows}
    treatments = {row["treatment"] for row in rows}
    if len(controls) != 1 or len(treatments) != 1:
        return {
            "stage": rows[0]["stage"],
            "control": sorted(controls),
            "treatment": sorted(treatments),
            "decision": "hold_control_divergence",
            "seed_decisions": [row["decision"] for row in rows],
        }
    metrics = {}
    for name in rows[0]["metrics"]:
        effects = np.asarray([
            row["metrics"][name]["absolute_effect"] for row in rows
        ])
        metrics[name] = {
            "mean_effect": float(effects.mean()),
            "seed_std": float(effects.std(ddof=1 if len(effects) > 1 else 0)),
            "per_seed": effects.tolist(),
        }
    passed = [row["decision"] == "pass" for row in rows]
    if all(passed):
        decision = "pass_all_seeds"
    elif metrics[primary_metric]["mean_effect"] < 0 or (
        metrics[value_metric]["mean_effect"] < 0
    ):
        decision = "reject_mean_regression"
    else:
        decision = "hold_seed_instability"
    return {
        "stage": rows[0]["stage"],
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "seed_passes": sum(passed),
        "seed_decisions": [row["decision"] for row in rows],
        "metrics": metrics,
    }


def pooled_cluster_metrics(rows):
    """Pool identical cluster-level comparisons by inverse variance."""
    controls = {row["control"] for row in rows}
    treatments = {row["treatment"] for row in rows}
    if len(controls) != 1 or len(treatments) != 1:
        raise ValueError("cluster launch seeds must share one comparison")
    metrics = {}
    for name in rows[0]["metrics"]:
        effects = np.asarray([
            row["metrics"][name]["absolute_effect"] for row in rows
        ])
        errors = np.asarray([
            row["metrics"][name]["standard_error"] for row in rows
        ]).clip(1e-12)
        precision = errors ** -2
        effect = float((effects * precision).sum() / precision.sum())
        error = float(precision.sum() ** -0.5)
        metrics[name] = {
            "mean_effect": float(effects.mean()),
            "seed_std": float(effects.std(ddof=1)),
            "per_seed": effects.tolist(),
            "pooled_effect": effect,
            "pooled_standard_error": error,
            "pooled_confidence_interval": [
                effect - 1.96 * error, effect + 1.96 * error
            ],
        }
    return metrics
