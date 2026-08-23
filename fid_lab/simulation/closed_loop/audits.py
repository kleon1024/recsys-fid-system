"""Independent candidate, cascade, and behavior-distribution audits."""

from __future__ import annotations

import numpy as np


def candidate_policy_audit(rows, policies) -> dict[str, object]:
    results = {}
    choices = {}
    for policy in policies:
        selected_probability = []
        oracle_probability = []
        policy_choices = []
        for row in rows:
            features = np.asarray(row.candidate_features, dtype=np.float32)
            oracle = np.asarray(row.candidate_oracle_long_view, dtype=np.float32)
            choice = int(np.argmax(policy.score(features)))
            policy_choices.append(choice)
            selected_probability.append(float(oracle[choice]))
            oracle_probability.append(float(oracle.max()))
        selected = np.asarray(selected_probability)
        oracle = np.asarray(oracle_probability)
        results[policy.name] = {
            "chosen_true_long_view_probability": float(selected.mean()),
            "oracle_probability": float(oracle.mean()),
            "oracle_regret": float((oracle - selected).mean()),
            "oracle_top1_rate": float(np.mean(np.isclose(selected, oracle))),
        }
        choices[policy.name] = np.asarray(policy_choices)
    ordered = [policy.name for policy in policies]
    changes = {
        f"{left}_to_{right}": float(np.mean(choices[left] != choices[right]))
        for left, right in zip(ordered, ordered[1:])
    }
    return {"policies": results, "adjacent_top1_change_rate": changes}


def cascade_audit(rows) -> dict[str, object]:
    route_counts: dict[str, int] = {}
    candidates = 0
    for row in rows:
        for routes in row.candidate_routes:
            candidates += 1
            for route in routes:
                route_counts[route] = route_counts.get(route, 0) + 1
    return {
        "mean_recalled_after_merge": float(np.mean([row.recall_count for row in rows])),
        "mean_after_coarse": float(np.mean([row.coarse_count for row in rows])),
        "route_candidate_coverage": {
            route: count / candidates for route, count in sorted(route_counts.items())
        },
    }


def behavior_distribution(rows) -> dict[str, object]:
    responses = [row.response for row in rows]
    exposures = len(responses)
    stays = np.asarray([response.stay_seconds for response in responses])
    rates = {
        name: float(np.mean([getattr(response, attribute) for response in responses]))
        for name, attribute in {
            "play_rate": "play",
            "play_3s_rate": "play_3s",
            "slide_rate": "slide",
            "long_view_rate": "long_view",
            "quality_long_view_rate": "high_quality_long_view",
            "like_rate": "like",
            "favorite_rate": "favorite",
            "comment_rate": "comment",
            "share_rate": "share",
            "poi_video_rate": "anchor_impression",
            "negative_rate": "negative_feedback",
        }.items()
    }
    anchors = sum(response.anchor_impression for response in responses)
    plays = sum(response.play for response in responses)
    clicks = sum(response.anchor_click for response in responses)
    details = sum(response.poi_detail for response in responses)
    payments = sum(response.payment for response in responses)
    return {
        "exposures": exposures,
        **rates,
        "stay_seconds_mean": float(stays.mean()),
        "stay_seconds_p50": float(np.quantile(stays, 0.50)),
        "stay_seconds_p90": float(np.quantile(stays, 0.90)),
        "play_3s_given_play": sum(response.play_3s for response in responses) / plays
        if plays
        else None,
        "slide_given_play": sum(response.slide for response in responses) / plays
        if plays
        else None,
        "anchor_ctr": clicks / anchors if anchors else None,
        "poi_detail_per_anchor_click": details / clicks if clicks else None,
        "payment_per_poi_detail": payments / details if details else None,
        "long_view_probability_calibration_gap": float(
            np.mean([response.probabilities["long_view"] for response in responses])
            - rates["long_view_rate"]
        ),
    }


