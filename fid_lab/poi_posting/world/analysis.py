"""Inverse-variance pooled launch decisions for creator-cluster experiments."""

from __future__ import annotations

from ...launches.statistics import pooled_cluster_metrics


def aggregate_creator_launch(rows):
    controls = {row["control"] for row in rows}
    treatments = {row["treatment"] for row in rows}
    if len(controls) != 1 or len(treatments) != 1:
        raise ValueError("creator launch seeds must share one comparison")
    metrics = pooled_cluster_metrics(rows)
    gates = {
        "publish_positive": metrics["publish_rate"]["pooled_confidence_interval"][0] > 0,
        "platform_lt_nonnegative": (
            metrics["platform_lt_per_request"]["pooled_confidence_interval"][0] >= 0
        ),
        "relevant_supply_nonnegative": (
            metrics["relevant_supply_per_request"]["pooled_confidence_interval"][0]
            >= -0.0002
        ),
        "negative_guardrail": (
            metrics["selected_content_negative_risk"]["pooled_confidence_interval"][1]
            <= 0.0002
        ),
        "direction_consistency": (
            sum(value > 0 for value in metrics["publish_rate"]["per_seed"]) >= 2
            and sum(
                value >= 0
                for value in metrics["platform_lt_per_request"]["per_seed"]
            ) >= 2
        ),
    }
    if "audit_oracle_recall" in metrics:
        gates["recall_nonnegative"] = (
            metrics["audit_oracle_recall"]["pooled_confidence_interval"][0] >= 0
        )
    if all(gates.values()):
        decision = "pass_pooled_three_seeds"
    elif (
        metrics["publish_rate"]["pooled_effect"] < 0
        or metrics["platform_lt_per_request"]["pooled_effect"] < 0
    ):
        decision = "reject_mean_regression"
    else:
        decision = "hold_uncertainty"
    return {
        "stage": rows[0]["stage"],
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "gates": gates,
        "metrics": metrics,
        "seed_decisions": [row["decision"] for row in rows],
        "creator_online_by_seed": [row["creator_online_ab"] for row in rows],
    }
