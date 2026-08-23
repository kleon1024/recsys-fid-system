"""Fixed-ranker paired shadow for isolating retrieval candidate quality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...ope import METRIC_NAMES, policy_value_gates
from ..data.randomized import calibration_masks, load_randomized_split, subset_split
from ..evaluation.policy import calibrate_response, policy_utility
from ..evaluation.statistics import cluster_interval
from ..kernel import KuaiBehaviorKernel
from ..launch.contracts import assert_artifact_compatible, stream_sha256
from .ladder import _positive_rows


def _rules(benchmark_report, fine_report, world_key):
    benchmark = json.loads(benchmark_report.read_text())["randomized_calibration"]
    fine = json.loads(fine_report.read_text())["randomized_calibration"]
    return {
        "fine": fine["sequence_randomized_adapter"],
        "world": benchmark[world_key],
    }


def _candidate_features(split, rows, catalog, candidates):
    sparse = catalog["sparse"][candidates].clone()
    dense = catalog["dense"][candidates].clone()
    sparse[:, :, 0] = split.sparse[rows, None, 0]
    dense[:, :, 1:3] = split.dense[rows, None, 1:3]
    dense[:, :, 4:] = split.dense[rows, None, 4:]
    return sparse, dense


@torch.inference_mode()
def _route_values(split, rows, candidates, catalog, fine, world, rules,
                  batch_size):
    outputs = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        index = rows[start:stop]
        sparse, dense = _candidate_features(
            split, index, catalog, candidates[start:stop]
        )
        arguments = (
            split.sparse[index], split.dense[index], sparse, dense,
            split.history_items[index], split.history_feedback[index],
        )
        fine_response = calibrate_response(fine.score_slate(*arguments), rules["fine"])
        choice = policy_utility(fine_response, "raw_probability").argmax(dim=1)
        response = calibrate_response(world.score_slate(*arguments), rules["world"])
        batch_rows = torch.arange(len(index), device=response.probabilities.device)
        probability = response.probabilities[batch_rows, choice]
        stay = response.stay_norm[batch_rows, choice, None]
        outputs.append(torch.cat((probability, stay), dim=1).cpu())
    return torch.cat(outputs)


def _comparison(control, treatment, users):
    delta = treatment.numpy() - control.numpy()
    metrics = {}
    for index, name in enumerate(METRIC_NAMES):
        mean, standard_error, interval = cluster_interval(delta[:, index], users)
        metrics[name] = {
            "absolute_delta": mean,
            "cluster_standard_error": standard_error,
            "confidence_interval_95": interval,
            "control_mean": float(control[:, index].mean()),
            "treatment_mean": float(treatment[:, index].mean()),
        }
    gates = policy_value_gates(
        metrics, {"paired_shadow": {"effective_sample_fraction": 1.0}}
    )
    return {
        "metrics": metrics,
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "hold_or_reject",
    }


def run_retrieval_shadow(ladder, dataset_dir: Path, fine_artifact: Path,
                         world_artifact: Path, benchmark_report: Path,
                         fine_report: Path, world_key="sequence_transformer",
                         device="cuda:0", batch_size=128):
    manifest = assert_artifact_compatible(
        dataset_dir, (fine_artifact, world_artifact)
    )
    randomized = load_randomized_split(dataset_dir, "random_test")
    _, evaluation = calibration_masks(
        randomized, ladder["config"]["evaluation_seed"]
    )
    split = subset_split(randomized, np.flatnonzero(evaluation))
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    rows, _ = _positive_rows(split, catalog)
    if not torch.equal(rows, ladder["test_rows"]):
        raise ValueError("retrieval shadow rows differ from offline evaluation")
    fine = KuaiBehaviorKernel.load(fine_artifact, device)
    world = KuaiBehaviorKernel.load(world_artifact, device)
    rules = _rules(benchmark_report, fine_report, world_key)
    route_values = {
        name: _route_values(
            split, rows, candidates, catalog, fine, world, rules, batch_size
        )
        for name, candidates in ladder["candidate_sets"].items()
    }
    order = ("popular", "co_visit_graph", "two_tower", "multi_interest")
    launches = []
    control = order[0]
    users = split.user_ids[rows].numpy()
    for treatment in order[1:]:
        result = _comparison(
            route_values[control], route_values[treatment], users
        )
        launches.append({
            "control": control, "treatment": treatment, **result,
        })
    return {
        "schema": "kuairand-retrieval-fixed-rank-shadow-v1",
        "queries": len(rows),
        "users": int(np.unique(users).size),
        "fixed_fine_ranker": stream_sha256(fine_artifact),
        "independent_world": stream_sha256(world_artifact),
        "dataset_catalog_sha256": manifest["catalog_sha256"],
        "launches": launches,
        "frozen_retrieval_control": control,
        "provisional_passes": [
            row["treatment"] for row in launches if row["decision"] == "pass"
        ],
        "evidence_boundary": (
            "Paired one-request shadow under a fixed fine ranker and independent "
            "Feed world; not stateful LT or a live A/B test."
        ),
    }
