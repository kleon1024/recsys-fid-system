"""Single launch-decision authority for exchanged platform LT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .container import LTMetricContainer


@dataclass(frozen=True)
class LTIncrement:
    point: float
    lower: float
    upper: float


def lt_increment(metric: Mapping[str, object], *, pooled: bool = False) -> LTIncrement:
    """Normalize ordinary and pooled A/B output into one LT interval."""
    prefix = "pooled_" if pooled else ""
    interval = metric[f"{prefix}confidence_interval"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("LT confidence interval must contain lower and upper bounds")
    point_key = f"{prefix}absolute_lift"
    if point_key in metric:
        point = float(metric[point_key])
    elif not pooled:
        point = float(metric["treatment_mean"]) - float(metric["control_mean"])
    else:
        raise ValueError("pooled LT metric is missing pooled_absolute_lift")
    return LTIncrement(point, float(interval[0]), float(interval[1]))


def unified_lt_launch_decision(
    metric: Mapping[str, object],
    *,
    pooled: bool = False,
    evidence_ready: bool = True,
    hard_constraint_failure: str | None = None,
) -> str:
    """Gate growth changes on the confidence interval of exchanged LT only.

    Safety, legal, privacy, and integrity constraints may fail independently.
    Unexchanged business or quality metrics are diagnostics, not launch gates.
    """
    if hard_constraint_failure:
        return hard_constraint_failure
    if not evidence_ready:
        return "reject_lt_evidence_invalid"
    increment = lt_increment(metric, pooled=pooled)
    if increment.point > 0.0 and increment.lower >= 0.0:
        return "pass_unified_lt_nonnegative"
    if increment.upper < 0.0:
        return "reject_unified_lt_negative"
    return "hold_unified_lt_uncertain"


def unified_lt_exchange_report(
    metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Render the exchange contract, component effects, and total gate evidence."""
    component_names = (
        "lt_stay_per_user",
        "lt_active_days_per_user",
        "accepted_platform_commercialization_per_user",
    )
    total = lt_increment(metrics["lt_value_per_user"])
    contract = LTMetricContainer().manifest()
    rates = contract["rates"]
    accepted = all(
        not str(rate["evidence"]).startswith("synthetic_")
        for rate in rates.values()
    )
    return {
        "contract": contract,
        "components": {name: metrics[name] for name in component_names},
        "total": metrics["lt_value_per_user"],
        "criterion": "point_estimate > 0 and 95% CI lower_bound >= 0",
        "overall_nonnegative": total.point > 0.0 and total.lower >= 0.0,
        "production_exchange_authority_accepted": accepted,
        "production_readiness": (
            "eligible_for_hard_constraint_review" if accepted else "hold_synthetic_rates"
        ),
    }
