"""Independent closure and cascade gates for request-level Feed tensors."""

from __future__ import annotations

import torch

from ...world_model.contracts import WORLD_LABEL_COUNT


CORE_BINARY_LABELS = (0, 1, 4, 5, 6)


def validate_request_tensors(tensors, config):
    rows = len(tensors["request_id"])
    index = torch.arange(rows)
    exposed = tensors["exposed_index"].long()
    candidates = tensors["candidate_item_ids"]
    sorted_items = candidates.sort(dim=1).values
    masks = tensors["label_masks"].bool()
    labels = tensors["labels"]
    gates = {
        "request_ids_unique": len(torch.unique(tensors["request_id"])) == rows,
        "candidate_width": candidates.shape[1] == config.merged_candidates,
        "candidate_ids_unique_per_request": bool(
            (sorted_items[:, 1:] != sorted_items[:, :-1]).all()
        ),
        "coarse_budget_exact": bool(
            (tensors["candidate_coarse_mask"].sum(dim=1) == config.candidates).all()
        ),
        "exposure_survived_coarse": bool(
            tensors["candidate_coarse_mask"][index, exposed].all()
        ),
        "candidate_label_mask_selected_only": bool(
            (tensors["candidate_label_mask"].sum(dim=1) == 1).all()
            and tensors["candidate_label_mask"][index, exposed].all()
        ),
        "world_label_width": labels.shape[1] == WORLD_LABEL_COUNT,
        "session_exit_unobserved": bool((~masks[:, 15]).all()),
        "unobserved_labels_are_zero": bool((labels[~masks] == 0).all()),
        "core_binary_labels_non_degenerate": all(
            0 < int(labels[:, index].sum()) < int(masks[:, index].sum())
            for index in CORE_BINARY_LABELS
        ),
        "long_view_implies_play": bool((labels[:, 5] <= labels[:, 0]).all()),
        "quality_view_implies_long_view": bool((labels[:, 6] <= labels[:, 5]).all()),
        "detail_implies_anchor": bool((labels[:, 10] <= labels[:, 9]).all()),
        "conversion_implies_detail": bool((labels[:, 12] <= labels[:, 10]).all()),
        "content_click_matches_play": bool(torch.equal(labels[:, 19], labels[:, 0])),
        "source_long_view_matches_label": bool(torch.equal(labels[:, 20], labels[:, 5])),
        "finite_features_scores_labels": bool(
            torch.isfinite(tensors["candidate_features"]).all()
            and torch.isfinite(tensors["candidate_audit_utility"]).all()
            and torch.isfinite(labels).all()
        ),
        "propensity_inside_unit_interval": bool(
            ((tensors["exposure_propensity"] > 0)
             & (tensors["exposure_propensity"] <= 1)).all()
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"request dataset closure failed: {failed}")
    support = {
        str(index): {
            "mature": int(masks[:, index].sum()),
            "positive": int(((labels[:, index] > 0) & masks[:, index]).sum()),
            "mean": (
                float(labels[masks[:, index], index].mean())
                if masks[:, index].any() else None
            ),
        }
        for index in range(labels.shape[1])
    }
    return {"requests": rows, "gates": gates, "label_support": support}
