"""Render the Local posting-supply switchback report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .switchback import (
    SupplySwitchbackConfig,
    calibrate_supply_switchback,
    run_supply_switchback,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=int, default=100)
    parser.add_argument("--periods", type=int, default=28)
    parser.add_argument("--users-per-city-period", type=int, default=10_000)
    parser.add_argument("--calibration-runs", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_supply_switchback(
        SupplySwitchbackConfig(
            cities=arguments.cities,
            periods=arguments.periods,
            users_per_city_period=arguments.users_per_city_period,
        )
    )
    if arguments.calibration_runs:
        report["estimator_calibration"] = calibrate_supply_switchback(
            SupplySwitchbackConfig(
                cities=arguments.cities,
                periods=arguments.periods,
                users_per_city_period=arguments.users_per_city_period,
            ),
            arguments.calibration_runs,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
