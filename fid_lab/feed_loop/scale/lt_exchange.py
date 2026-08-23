"""LT commercialization exchange-rate statistics over frozen trajectories."""

from __future__ import annotations

from math import erfc, sqrt

import torch

from ...value import DEFAULT_LT_CONFIG


LT_EXCHANGE_RATE_SCENARIOS = (0.0, 0.10, 0.25, 1.0)


def accumulate_lt_exchange_components(stats, user_ids, user_metrics) -> None:
    bucket = torch.remainder(user_ids * 1_664_525 + 1_013_904_223, 2**31)
    assigned = bucket < 2**30
    accepted = user_metrics[:, 18].to(torch.float64)
    base_rate = DEFAULT_LT_CONFIG.rates[
        "accepted_commercialization_unit"
    ].unit_value
    noncommercial = user_metrics[:, 19].to(torch.float64) - base_rate * accepted
    for cell, mask in enumerate((~assigned, assigned)):
        x = noncommercial[mask]
        y = accepted[mask]
        stats[cell] += torch.stack(
            (
                mask.sum(),
                x.sum(),
                x.square().sum(),
                y.sum(),
                y.square().sum(),
                (x * y).sum(),
            )
        )


def render_lt_exchange_components(stats):
    report = {}
    for cell, cell_name in enumerate(("control", "treatment")):
        count, sum_x, sum_x2, sum_y, sum_y2, sum_xy = stats[cell]
        mean_x = sum_x / count
        mean_y = sum_y / count
        report[cell_name] = {
            "users": int(count),
            "noncommercial_mean": float(mean_x),
            "noncommercial_variance": float(
                (sum_x2 - sum_x.square() / count) / (count - 1.0)
            ),
            "accepted_commercialization_mean": float(mean_y),
            "accepted_commercialization_variance": float(
                (sum_y2 - sum_y.square() / count) / (count - 1.0)
            ),
            "covariance": float(
                (sum_xy - sum_x * sum_y / count) / (count - 1.0)
            ),
        }
    return report


def combine_lt_exchange_sensitivity(
    control_report,
    treatment_report,
    rates=LT_EXCHANGE_RATE_SCENARIOS,
):
    control = control_report["lt_exchange_components"]["control"]
    treatment = treatment_report["lt_exchange_components"]["treatment"]
    base_rate = DEFAULT_LT_CONFIG.rates[
        "accepted_commercialization_unit"
    ].unit_value
    output = {}
    for rate in rates:
        cells = []
        for source in (control, treatment):
            mean = (
                source["noncommercial_mean"]
                + rate * source["accepted_commercialization_mean"]
            )
            variance = (
                source["noncommercial_variance"]
                + rate**2 * source["accepted_commercialization_variance"]
                + 2.0 * rate * source["covariance"]
            )
            cells.append((mean, max(variance, 0.0), source["users"]))
        control_mean, control_variance, control_users = cells[0]
        treatment_mean, treatment_variance, treatment_users = cells[1]
        effect = treatment_mean - control_mean
        standard_error = sqrt(
            control_variance / control_users
            + treatment_variance / treatment_users
        )
        control_commerce = control_report["metrics"][
            "accepted_platform_commercialization_per_user"
        ]
        treatment_commerce = treatment_report["metrics"][
            "accepted_platform_commercialization_per_user"
        ]
        control_truth = (
            control_report["metrics"]["lt_value_per_user"]
            - base_rate * control_commerce
            + rate * control_commerce
        )
        treatment_truth = (
            treatment_report["metrics"]["lt_value_per_user"]
            - base_rate * treatment_commerce
            + rate * treatment_commerce
        )
        output[f"{rate:g}"] = {
            "exchange_rate": rate,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "absolute_lift": effect,
            "relative_lift": effect / control_mean if abs(control_mean) > 1e-12 else None,
            "standard_error": standard_error,
            "confidence_interval": (
                effect - 1.96 * standard_error,
                effect + 1.96 * standard_error,
            ),
            "p_value": erfc(
                abs(effect / max(standard_error, 1e-12)) / sqrt(2.0)
            ),
            "known_absolute_effect": treatment_truth - control_truth,
            "known_relative_effect": (
                (treatment_truth - control_truth) / control_truth
                if abs(control_truth) > 1e-12
                else None
            ),
        }
    return output
