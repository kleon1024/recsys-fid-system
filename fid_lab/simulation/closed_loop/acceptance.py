"""Acceptance gates for the closed-loop simulator contract."""

from __future__ import annotations


def simulator_acceptance(report: dict[str, object]) -> dict[str, object]:
    distribution = report["behavior_distribution"]
    cascade = report["cascade"]
    joiner = report["joiner"]
    launches = report["ab_ladder"]
    progression = (
        "popular_baseline_to_quality_affinity_rule",
        "quality_affinity_rule_to_lr_basic_features",
    )
    checks = {
        "behavior_rates_plausible": (
            0.85 <= distribution["play_rate"] <= 0.99
            and 0.05 <= distribution["slide_rate"] <= 0.35
            and 0.10 <= distribution["long_view_rate"] <= 0.50
            and 0.02 <= distribution["quality_long_view_rate"] <= 0.20
        ),
        "long_view_probability_calibrated": abs(
            distribution["long_view_probability_calibration_gap"]
        )
        <= 0.03,
        "all_recall_routes_reach_coarse": set(
            cascade["route_candidate_coverage"]
        )
        == {
            "ann",
            "graph",
            "geo",
            "fresh",
            "long_tail",
            "popular",
            "post_search",
            "retarget",
        },
        "cascade_budget_enforced": (
            cascade["mean_recalled_after_merge"] > cascade["mean_after_coarse"]
            and cascade["mean_after_coarse"] == report["config"]["candidates"]
        ),
        "simple_ladder_demonstrates_positive_algorithm_impact": all(
            launches[name]["metrics"]["stay_per_exposure"]["true_itt"] > 0.0
            for name in progression
        ),
        "shadow_replay_exact": report["offline_online_max_score_delta"] < 1e-6,
        "coarse_and_exposure_samples_close": (
            joiner["coarse_examples"]
            == report["config"]["candidates"] * joiner["fine_examples"]
        ),
        "request_candidate_authority_closes": (
            joiner["request_candidate_dataset"]["one_exposure_per_request"]
            and joiner["request_candidate_dataset"]["candidate_decisions"]
            == joiner["request_candidate_dataset"]["mature_label_rows"]
            and joiner["request_candidate_dataset"]["requests"]
            == joiner["fine_examples"]
        ),
        "ab_estimators_recover_assignment_distribution": all(
            all(
                metric["truth_inside_randomization_interval"]
                for metric in launch["randomization_audit"].values()
            )
            for launch in launches.values()
        ),
        "experiment_logs_feed_round_two": (
            report["policy_iteration"]["training_examples_after"]
            > report["policy_iteration"]["training_examples_before"]
        ),
    }
    if "supply_iteration" in report:
        supply = report["supply_iteration"]
        checks["posting_supply_reaches_distribution"] = (
            supply["treatment_posting"]["published_videos"] > 0
            and supply["treatment_catalog_poi_items"]
            > supply["control_catalog_poi_items"]
            and any(
                abs(metric["absolute_effect"]) > 1e-9
                for metric in supply["supply_only_paired_world_effects"].values()
            )
        )
    return {"passed": all(checks.values()), "checks": checks}


