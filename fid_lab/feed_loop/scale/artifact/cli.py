"""Run a published stateful model launch in the GPU tensor environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import TensorArtifactPolicy
from ..tensor_engine import TensorFeedConfig, combine_tensor_ab, run_tensor_feed


def _semantic_parity(source, control, ab, candidate_name):
    launch = source["launches"][f"lr_to_{candidate_name}"]
    mappings = {
        "stay_per_exposure": "stay_per_exposure",
        "long_view_rate": "long_view_rate",
        "quality_long_view_rate": "quality_long_view_rate",
    }
    metrics = {}
    for semantic_name, tensor_name in mappings.items():
        semantic = launch["metrics"][semantic_name]["control_mean"]
        tensor = control["metrics"][tensor_name]
        gap = (tensor - semantic) / semantic
        metrics[semantic_name] = {
            "semantic_control": semantic,
            "tensor_control": tensor,
            "relative_gap": gap,
            "within_10pct": abs(gap) <= 0.10,
        }
    effects = {}
    for name in ("stay_per_exposure", "quality_long_view_rate"):
        semantic = launch["metrics"][name]["true_relative_itt"]
        tensor = ab[name]["relative_lift"]
        effects[name] = {
            "semantic_true_relative_itt": semantic,
            "tensor_relative_lift": tensor,
            "absolute_gap": abs(tensor - semantic),
            "within_2pp": abs(tensor - semantic) <= 0.02,
        }
    return {
        "distribution": metrics,
        "treatment_effect": effects,
        "passed": all(value["within_10pct"] for value in metrics.values())
        and all(value["within_2pp"] for value in effects.values()),
    }


def _launch_decision(parity, ab):
    if not parity["passed"]:
        return "reject_semantic_tensor_parity"
    negative = ab["negative_rate"]
    quality = ab["quality_long_view_rate"]
    stay = ab["stay_per_exposure"]
    lt_value = ab["lt_value_per_user"]
    if negative["relative_lift"] > 0 and negative["p_value"] < 0.05:
        return "reject_negative_feedback"
    if quality["relative_lift"] < -0.01:
        return (
            "reject_quality_long_view_guardrail"
            if quality["p_value"] < 0.05
            else "hold_quality_long_view_risk"
        )
    if stay["relative_lift"] < 0 and stay["p_value"] < 0.05:
        return "reject_primary_regression"
    if lt_value["relative_lift"] < -0.01:
        return "reject_lt_value_guardrail"
    if stay["relative_lift"] > 0 and stay["p_value"] < 0.05:
        return "pass_primary_metric"
    return "hold_underpowered_or_neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    report_path = Path(args.report)
    artifact_dir = Path(args.artifact_dir)
    config = TensorFeedConfig(
        users=args.users,
        steps=args.steps,
        batch_users=args.batch_users,
        device=args.device,
        signal_version="heterogeneous-nonlinear-v2",
    )
    control_policy = TensorArtifactPolicy(
        report_path, artifact_dir, args.device, treatment=False
    )
    treatment_policy = TensorArtifactPolicy(
        report_path, artifact_dir, args.device, treatment=True
    )
    control = run_tensor_feed(config, control_policy)
    treatment = run_tensor_feed(config, treatment_policy)
    ab = combine_tensor_ab(control, treatment)
    semantic_source = json.loads(report_path.read_text())
    parity = _semantic_parity(
        semantic_source, control, ab, treatment_policy.name
    )
    report = {
        "artifact_manifest": treatment_policy.manifest,
        "common_random_numbers": True,
        "control": control,
        "treatment": treatment,
        "ab": ab,
        "semantic_parity": parity,
        "launch_decision": _launch_decision(parity, ab),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
