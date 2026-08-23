"""Materialize a small request-level candidate dataset from the stateful world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.experiment import build_feed_joiner
from ...simulation.policies import HeuristicPolicy
from ...simulation.population import run_population
from .request_dataset import build_request_candidate_dataset, materialize_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = SimulationConfig(
        users=args.users,
        items=args.items,
        candidates=args.candidates,
        joiner_users=args.users,
        signal_version="heterogeneous-nonlinear-v2",
    )
    catalog = build_catalog(config)
    policy = HeuristicPolicy()
    trajectories = run_population(config, catalog, policy, range(config.users))
    assigned = np.zeros(config.users, dtype=bool)
    joined = build_feed_joiner(
        config, catalog, trajectories, (policy, policy), assigned
    )
    dataset = build_request_candidate_dataset(
        trajectories,
        catalog,
        joined,
        {"model": policy.name, "feature": "stateful-v2"},
    )
    print(json.dumps(materialize_dataset(dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
