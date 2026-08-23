"""Paired stateful shadow replay over an independent external world kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .kuairand.data.sequence import load_sequence_split
from .kuairand.evaluation.policy import RAW_SELECTION_WEIGHTS, policy_utility
from .kuairand.kernel import KuaiBehaviorKernel
from .kuairand.launch.contracts import assert_artifact_compatible, stream_sha256


METRIC_NAMES = ("click", "long_view", "like", "hate", "stay_norm")
SELECTION_WEIGHTS = RAW_SELECTION_WEIGHTS
MAX_INTERACTION_REGRESSION = 0.005
GUARD_TOLERANCES = {
    "click": 0.02,
    "long_view": 0.02,
    "like": 0.02,
    "hate": 0.0005,
    "stay_norm": 0.005,
}


def _selection_value(response):
    return policy_utility(response, "raw_probability")


def _selected_metrics(response, choice):
    rows = torch.arange(len(choice), device=choice.device)
    probability = response.probabilities[rows, choice]
    return torch.stack((
        probability[:, 0], probability[:, 1], probability[:, 2],
        probability[:, 6], response.stay_norm[rows, choice],
    ), dim=1)


def treatment_guard_mask(base, candidate, choice):
    rows = torch.arange(len(choice), device=choice.device)
    selected = base.probabilities[rows, choice]
    selected_stay = base.stay_norm[rows, choice]
    eligible = (
        base.probabilities[:, :, 0] >= selected[:, None, 0] - GUARD_TOLERANCES["click"]
    ) & (
        base.probabilities[:, :, 1]
        >= selected[:, None, 1] - GUARD_TOLERANCES["long_view"]
    ) & (
        base.probabilities[:, :, 2] >= selected[:, None, 2] - GUARD_TOLERANCES["like"]
    ) & (
        candidate.probabilities[:, :, 6]
        <= selected[:, None, 6] + GUARD_TOLERANCES["hate"]
    ) & (
        candidate.stay_norm
        >= selected_stay[:, None] - GUARD_TOLERANCES["stay_norm"]
    )
    eligible[rows, choice] = True
    return eligible


def _arm_step(policy, world, request_sparse, request_dense, candidate_sparse,
              candidate_dense, history_items, history_feedback, seed, guard=None):
    policy_response = policy.score_slate(
        request_sparse, request_dense, candidate_sparse, candidate_dense,
        history_items, history_feedback,
    )
    policy_value = _selection_value(policy_response)
    if guard is not None:
        base = guard.score_slate(
            request_sparse, request_dense, candidate_sparse, candidate_dense,
            history_items, history_feedback,
        )
        base_choice = _selection_value(base).argmax(dim=1)
        policy_value = policy_value.masked_fill(
            ~treatment_guard_mask(base, policy_response, base_choice), -1e9
        )
    choice = policy_value.argmax(dim=1)
    world_response = world.score_slate(
        request_sparse, request_dense, candidate_sparse, candidate_dense,
        history_items, history_feedback,
    )
    metrics = _selected_metrics(world_response, choice)
    actions, _ = world.sample_selected(world_response, choice, seed)
    choice_cpu = choice.cpu()
    rows = torch.arange(len(choice_cpu))
    items = candidate_sparse[rows, choice_cpu, 1]
    history = world.advance_history(
        history_items, history_feedback, items, actions.cpu()
    )
    return metrics.cpu(), history


def _candidate_slate(split, start, stop, candidates, step, seed):
    generator = torch.Generator().manual_seed(seed + step * 10_007 + start)
    indices = torch.randint(
        len(split), (stop - start, candidates), generator=generator
    )
    return split.sparse[indices], split.dense[indices]


def run_paired_replay(dataset_dir, control_artifact, treatment_artifact,
                      world_artifact, device_name="cuda:0", users=5_000,
                      candidates=20, steps=8, batch_size=256, seed=20260824):
    dataset_manifest = assert_artifact_compatible(
        dataset_dir, (control_artifact, treatment_artifact, world_artifact)
    )
    split = load_sequence_split(dataset_dir, "test", users)
    control = KuaiBehaviorKernel.load(control_artifact, device_name)
    treatment = KuaiBehaviorKernel.load(treatment_artifact, device_name)
    world = KuaiBehaviorKernel.load(world_artifact, device_name)
    totals = {
        "control": torch.zeros(len(split), len(METRIC_NAMES)),
        "treatment": torch.zeros(len(split), len(METRIC_NAMES)),
    }
    histories = {
        name: (split.history_items.clone(), split.history_feedback.clone())
        for name in totals
    }
    policies = {"control": control, "treatment": treatment}
    for step in range(steps):
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            candidate_sparse, candidate_dense = _candidate_slate(
                split, start, stop, candidates, step, seed
            )
            for name in ("control", "treatment"):
                items, feedback = histories[name]
                metrics, updated = _arm_step(
                    policies[name], world, split.sparse[start:stop],
                    split.dense[start:stop], candidate_sparse, candidate_dense,
                    items[start:stop], feedback[start:stop],
                    seed + step * 100_003 + start,
                    control if name == "treatment" else None,
                )
                totals[name][start:stop] += metrics
                items[start:stop], feedback[start:stop] = updated
    report = _replay_report(totals, steps, candidates)
    report["artifacts"] = {
        "dataset_manifest": dataset_manifest,
        "control_sha256": stream_sha256(control_artifact),
        "treatment_sha256": stream_sha256(treatment_artifact),
        "world_sha256": stream_sha256(world_artifact),
    }
    report["candidate_seed"] = seed
    return report


def _replay_report(totals, steps, candidates):
    delta = (totals["treatment"] - totals["control"]) / steps
    report = {}
    for index, name in enumerate(METRIC_NAMES):
        values = delta[:, index].numpy()
        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
        mean = float(values.mean())
        report[name] = {
            "absolute_delta": mean,
            "standard_error": standard_error,
            "confidence_interval_95": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
            "control_mean": float((totals["control"][:, index] / steps).mean()),
        }
    gates = {
        "long_view_nonnegative": report["long_view"]["confidence_interval_95"][0] >= 0,
        "stay_nonnegative": report["stay_norm"]["confidence_interval_95"][0] >= 0,
        "hate_nonpositive": report["hate"]["confidence_interval_95"][1] <= 0,
        "click_guardrail": (
            report["click"]["confidence_interval_95"][0]
            >= -MAX_INTERACTION_REGRESSION
        ),
        "like_guardrail": (
            report["like"]["confidence_interval_95"][0]
            >= -MAX_INTERACTION_REGRESSION
        ),
    }
    return {
        "schema": "kuairand-external-stateful-shadow-ab-v1",
        "users": len(delta), "steps": steps, "candidates": candidates,
        "metrics": report, "gates": gates,
        "decision": "shadow_ab_pass" if all(gates.values()) else "shadow_ab_hold",
        "selection_value": {
            "weights": SELECTION_WEIGHTS,
            "semantics": "ranking utility only, not unified LT",
            "control_feasibility_tolerances": GUARD_TOLERANCES,
        },
        "evidence_boundary": (
            "Independent-seed learned-world replay; not randomized-log causal evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--treatment-artifact", type=Path, required=True)
    parser.add_argument("--world-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--users", type=int, default=5_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    report = run_paired_replay(
        args.dataset_dir, args.control_artifact, args.treatment_artifact,
        args.world_artifact, args.device, args.users, args.candidates, args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "metrics": report["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
