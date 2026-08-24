"""Common-random Feed A/B gate for a V4 request-aware model artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..model_ladder.v4.serving import TensorV4RequestPolicy
from ..tensor_engine import TensorFeedConfig, combine_tensor_ab, run_tensor_feed
from ..tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..tensor_runtime.contracts import (
    EXTERNAL_MIXTURE_FEED_VERSION,
    LOCAL_NEURAL_SIGNAL_VERSION,
)
from ...tensor_policies import PERSONALIZED


def _gate(policy, world, config):
    control_world = run_tensor_feed(
        config, PERSONALIZED, behavior_world=world
    )
    treatment_world = run_tensor_feed(
        config, policy, behavior_world=world
    )
    ab = combine_tensor_ab(control_world, treatment_world)
    lt = ab["lt_value_per_user"]
    negative = ab["negative_rate"]
    duration = ab["selected_duration_per_exposure"]
    gates = {
        "behavior_world_lineage_matches": (
            policy.behavior_world == world.describe()
        ),
        "lt_confidence_interval_nonnegative": lt["confidence_interval"][0] >= 0,
        "negative_absolute_delta_below_5bp": (
            negative["confidence_interval"][1] <= 0.0005
        ),
        "selected_duration_relative_shift_below_5pct": (
            duration["relative_lift"] is not None
            and abs(duration["relative_lift"]) <= 0.05
        ),
    }
    return {
        "schema": "v4-request-model-common-random-launch-review-v1",
        "config": asdict(config),
        "control": PERSONALIZED.name,
        "treatment": policy.describe(),
        "ab": ab,
        "gates": gates,
        "decision": "launch" if all(gates.values()) else "hold",
        "performance": {
            "control": control_world["performance"],
            "treatment": treatment_world["performance"],
        },
        "evidence_boundary": (
            "Synthetic common-random Feed A/B; not production lift evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("din", "transformer", "mmoe", "ple"), required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--behavior-artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--external-dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-weight", type=float)
    parser.add_argument("--base-tolerance", type=float, default=0.05)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-users", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users, steps=args.steps, candidates=12,
        route_candidates=16, route_oversample=4, merged_candidates=64,
        audit_candidates=32, catalog_items=200_000, catalog_creators=25_000,
        batch_users=args.batch_users, signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        local_signal_version=LOCAL_NEURAL_SIGNAL_VERSION,
        behavior_sequence_length=64, device=args.device,
        retain_paired_user_metrics=True,
    )
    world = ExternalSequenceMixtureWorld(
        args.behavior_artifact, args.calibration_report,
        args.external_dataset_dir, args.device, config.seed,
    )
    policy = TensorV4RequestPolicy(
        args.model, args.training_report, args.artifact_dir, args.device,
        args.blend_weight, args.base_tolerance,
    )
    report = _gate(policy, world, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "lt": report["ab"]["lt_value_per_user"],
        "stay": report["ab"]["stay_per_exposure"],
        "negative": report["ab"]["negative_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
