"""Aggregate fixed-sample retrieval seeds into one launch decision."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ORDER = ("co_visit_graph", "two_tower", "multi_interest")


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _launch_by_treatment(report):
    return {
        launch["treatment"]: launch
        for launch in report["shadow"]["launches"]
    }


def _aggregate_model(name, reports):
    control_recall = np.asarray([
        report["models"]["popular"]["recall_at_k"] for report in reports
    ])
    recall = np.asarray([
        report["models"][name]["recall_at_k"] for report in reports
    ])
    launches = [_launch_by_treatment(report)[name] for report in reports]
    stay = np.asarray([
        launch["metrics"]["stay_norm"]["absolute_delta"]
        for launch in launches
    ])
    seed_passes = sum(launch["decision"] == "pass" for launch in launches)
    stable_pass = seed_passes == len(reports) and recall.mean() >= control_recall.mean()
    if stable_pass:
        decision = "pass"
    elif np.allclose(stay, 0.0) and np.allclose(recall, control_recall):
        decision = "hold_no_ranking_delta"
    elif stay.mean() < 0 or recall.mean() < control_recall.mean():
        decision = "reject_unstable_or_regressive"
    else:
        decision = "hold_inconclusive"
    return {
        "control": "popular",
        "treatment": name,
        "decision": decision,
        "seed_passes": seed_passes,
        "offline_recall_at_k": {
            "control_mean": float(control_recall.mean()),
            "treatment_mean": float(recall.mean()),
            "treatment_std": float(recall.std(ddof=1)),
            "per_seed": recall.tolist(),
        },
        "fixed_rank_shadow_stay_norm": {
            "mean": float(stay.mean()),
            "std": float(stay.std(ddof=1)),
            "per_seed": stay.tolist(),
        },
        "all_guardrails_pass_every_seed": all(
            all(launch["gates"].values()) for launch in launches
        ),
    }


def build_retrieval_review(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text()) for path in paths]
    if len(reports) < 3:
        raise ValueError("retrieval launch review requires at least three seeds")
    configs = [report["config"] for report in reports]
    fixed = {
        (config["evaluation_seed"], config["top_k"])
        for config in configs
    }
    if len(fixed) != 1:
        raise ValueError("retrieval seeds do not share evaluation sample and Top-K")
    catalogs = {report["dataset_catalog_sha256"] for report in reports}
    fine_rankers = {report["shadow"]["fixed_fine_ranker"] for report in reports}
    worlds = {report["shadow"]["independent_world"] for report in reports}
    if len(catalogs) != 1 or len(fine_rankers) != 1 or len(worlds) != 1:
        raise ValueError("retrieval seed evidence uses different frozen authorities")
    launches = [_aggregate_model(name, reports) for name in ORDER]
    active = "popular"
    for launch in launches:
        if launch["decision"] == "pass":
            active = launch["treatment"]
    return {
        "schema": "main-feed-retrieval-launch-review-v1",
        "launch_id": "L-RECALL-EXT-001",
        "seeds": [config["seed"] for config in configs],
        "evaluation_seed": configs[0]["evaluation_seed"],
        "top_k": configs[0]["top_k"],
        "dataset_catalog_sha256": next(iter(catalogs)),
        "fixed_fine_ranker": next(iter(fine_rankers)),
        "independent_world": next(iter(worlds)),
        "launches": launches,
        "active_retrieval_control": active,
        "decision": "retain_popular_control" if active == "popular"
        else f"promote_{active}",
        "evidence": [
            {"report": str(path), "sha256": _hash(path)} for path in paths
        ],
        "evidence_boundary": (
            "Random-exposure offline retrieval plus paired fixed-ranker shadow; "
            "not stateful LT or a live A/B test."
        ),
    }
