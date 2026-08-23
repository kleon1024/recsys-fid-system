"""Materialize the current candidate-cascade GPU acceptance report."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..tensor_engine import (
    LOCAL_INTENT_RANKER,
    PERSONALIZED,
    TensorFeedConfig,
    combine_tensor_ab,
    combine_tensor_trigger_ab,
    run_tensor_feed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=200_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users,
        steps=args.steps,
        batch_users=args.batch_users,
        device=args.device,
        signal_version="heterogeneous-nonlinear-v2",
        trace_users=4,
    )
    worlds = {
        policy.name: run_tensor_feed(config, policy, trigger_kind="post_search")
        for policy in (PERSONALIZED, LOCAL_INTENT_RANKER)
    }
    control = worlds[PERSONALIZED.name]
    treatment = worlds[LOCAL_INTENT_RANKER.name]
    report = {
        "suite": "credible-feed-digital-twin-cascade-v3",
        "config": asdict(config),
        "control": control,
        "treatment": treatment,
        "ab": combine_tensor_ab(control, treatment),
        "trigger_analysis": combine_tensor_trigger_ab(control, treatment),
        "evidence_boundary": (
            "Synthetic tensor DGP validates cascade mechanics and effect recovery, "
            "not production lift."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "suite": report["suite"],
        "candidate_graph": config.candidate_graph_version,
    }, indent=2))


if __name__ == "__main__":
    main()
