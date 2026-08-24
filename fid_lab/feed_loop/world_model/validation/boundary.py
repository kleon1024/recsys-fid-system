"""Leakage and execution-invariance probes for the learned world authority."""

from __future__ import annotations

import torch

from ..data import WorldModelSplit


MAX_NUMERICAL_DELTA = 1e-5


def _prediction_tensor(ensemble, split, indices, device):
    prediction = ensemble.predict(split.batch(indices, device))
    return torch.cat((
        prediction["probability_mean"],
        prediction["stay_mean"][:, None],
        prediction["utility_mean"][:, None],
    ), dim=1).cpu()


def _replace(split, **changes):
    return WorldModelSplit(**{**split.__dict__, **changes})


def boundary_invariance_report(ensemble, split, device_name, limit=2_048):
    device = torch.device(device_name)
    count = min(len(split), limit)
    indices = torch.arange(count)
    baseline = _prediction_tensor(ensemble, split, indices, device)

    permutation = torch.arange(count - 1, -1, -1)
    permuted = _prediction_tensor(ensemble, split, permutation, device)
    order_delta = float((baseline - permuted.flip(0)).abs().max())

    midpoint = max(count // 2, 1)
    chunked = torch.cat((
        _prediction_tensor(ensemble, split, indices[:midpoint], device),
        _prediction_tensor(ensemble, split, indices[midpoint:], device),
    ))
    batch_delta = float((baseline - chunked).abs().max())

    labels = split.labels.clone()
    labels[:count] = 1.0 - labels[:count]
    masks = split.label_masks.clone()
    masks[:count] = 1.0 - masks[:count]
    label_changed = _replace(split, labels=labels, label_masks=masks)
    label_delta = float((
        baseline - _prediction_tensor(ensemble, label_changed, indices, device)
    ).abs().max())

    fine_scores = split.candidate_fine_scores.clone()
    audit = split.candidate_audit_utility.clone()
    fine_scores[:count] = torch.flip(fine_scores[:count], dims=(1,))
    audit[:count] = 99.0
    score_changed = _replace(
        split, candidate_fine_scores=fine_scores,
        candidate_audit_utility=audit,
    )
    score_delta = float((
        baseline - _prediction_tensor(ensemble, score_changed, indices, device)
    ).abs().max())

    gates = {
        "request_order_invariance": order_delta <= MAX_NUMERICAL_DELTA,
        "batch_partition_invariance": batch_delta <= MAX_NUMERICAL_DELTA,
        "future_label_leakage": label_delta <= MAX_NUMERICAL_DELTA,
        "platform_score_leakage": score_delta <= MAX_NUMERICAL_DELTA,
    }
    return {
        "rows": count,
        "maximum_allowed_delta": MAX_NUMERICAL_DELTA,
        "deltas": {
            "request_order": order_delta,
            "batch_partition": batch_delta,
            "future_labels": label_delta,
            "platform_scores": score_delta,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
