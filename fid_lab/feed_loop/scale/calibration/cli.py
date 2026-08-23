"""Run the externally calibrated V3 Feed world and its first launch audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from ....value import unified_lt_launch_decision
from ..tensor_engine import (
    DEFAULT_GPU_BATCH_USERS,
    LOCAL_INTENT_RANKER,
    PERSONALIZED,
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)


def _alignment(calibration, simulated):
    profile = calibration["profile"]
    observed = {
        "play_rate": profile["play_threshold_rate"]["positive"],
        "play_3s_rate": profile["play_threshold_rate"]["three_seconds"],
        "long_view_rate": profile["feedback_rate"]["long_view"],
        "like_rate": profile["feedback_rate"]["is_like"],
        "negative_rate": profile["feedback_rate"]["is_hate"],
        "stay_per_exposure": profile["mean_play_seconds"],
    }
    return {
        name: {
            "observed": value,
            "simulated": simulated[name],
            "relative_error": (simulated[name] - value) / value,
        }
        for name, value in observed.items()
    }


def run_calibrated_launch(config, calibration_path):
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes)
    if calibration["schema"] != "kuairand-standard-calibration-v1":
        raise ValueError("unsupported external calibration schema")
    control = run_tensor_feed(config, PERSONALIZED)
    treatment = run_tensor_feed(config, LOCAL_INTENT_RANKER)
    metrics = combine_tensor_ab(control, treatment)
    return {
        "suite": "externally-calibrated-feed-v3",
        "launch_id": "L-SIMULATOR-004",
        "config": asdict(config),
        "calibration": {
            "schema": calibration["schema"],
            "sha256": sha256(calibration_bytes).hexdigest(),
            "source": calibration["source"],
            "causal_boundary": calibration["causal_boundary"],
            "alignment": _alignment(calibration, control["metrics"]),
        },
        "control": control,
        "treatment": treatment,
        "ab": metrics,
        "decision": unified_lt_launch_decision(metrics["lt_value_per_user"]),
        "evidence_boundary": (
            "Public standard-policy logs calibrate marginals only. This synthetic "
            "A/B is not unbiased OPE or production lift evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users,
        steps=args.steps,
        batch_users=args.batch_users,
        device=args.device,
        signal_version="kuairand-calibrated-v3",
        trace_users=8,
    )
    report = run_calibrated_launch(config, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "alignment": report["calibration"]["alignment"],
        "lt": report["ab"]["lt_value_per_user"],
    }, indent=2))


if __name__ == "__main__":
    main()
