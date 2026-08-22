"""Closed-loop policy iteration, replay, Joiner, and user-level A/B."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..evolution.data.contracts import (
    ActionEvent,
    CommerceEvent,
    OutboundClick,
    PixelEvent,
    StageDecision,
)
from ..evolution.data.joiner import EvolutionJoiner
from ..evolution.evaluation.metrics import binary_metrics, grouped_auc
from .ab import experiment_metrics, launch_decision, randomization_audit
from .contracts import SimulationConfig, Trajectory
from .environment import build_catalog
from .policies import (
    HeuristicPolicy,
    PopularPolicy,
    fit_logistic_policy,
    fit_policies,
    serialized_replay_deltas,
)
from .population import run_population
from .local.supply import run_supply_iteration


def _training_data(config: SimulationConfig, catalog):
    policy = HeuristicPolicy()
    trajectories = run_population(
        config, catalog, policy, range(config.users), explore=True
    )
    rows = [row for trajectory in trajectories for row in trajectory.rows]
    return rows, *_example_arrays(rows, config)


def _example_arrays(rows, config: SimulationConfig):
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([row.response.long_view for row in rows], dtype=np.int8)
    users = np.asarray([row.user_id for row in rows], dtype=np.int64)
    sessions = np.asarray(
        [row.user_id * config.max_sessions + row.session_id for row in rows], dtype=np.int64
    )
    propensities = np.asarray([row.selection_probability for row in rows], dtype=np.float32)
    return features, labels, users, sessions, propensities


def _candidate_policy_audit(rows, policies) -> dict[str, object]:
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


def _cascade_audit(rows) -> dict[str, object]:
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


def _behavior_distribution(rows) -> dict[str, object]:
    responses = [row.response for row in rows]
    exposures = len(responses)
    stays = np.asarray([response.stay_seconds for response in responses])
    rates = {
        name: float(np.mean([getattr(response, attribute) for response in responses]))
        for name, attribute in {
            "play_rate": "play",
            "play_3s_rate": "play_3s",
            "slide_rate": "slide",
            "lt_rate": "long_view",
            "hlt_rate": "high_quality_long_view",
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
            - rates["lt_rate"]
        ),
    }


def _row_decisions(row, catalog, model_name: str, history, timestamp: int, observable: bool):
    order = np.argsort(-np.asarray(row.candidate_scores))
    rank_by_index = np.empty(len(order), dtype=int)
    rank_by_index[order] = np.arange(1, len(order) + 1)
    decisions = []
    for candidate_index, (candidate_id, features, score, propensity, routes) in enumerate(
        zip(
            row.candidate_ids,
            row.candidate_features,
            row.candidate_scores,
            row.candidate_propensities,
            row.candidate_routes,
        )
    ):
        decisions.append(
            StageDecision(
                row.request_id,
                row.user_id,
                int(catalog.author[candidate_id]),
                candidate_id,
                int(catalog.poi[candidate_id]),
                timestamp,
                int(catalog.category[candidate_id]),
                int(catalog.city[candidate_id]),
                (
                    row.user_id % 1024,
                    int(catalog.author[candidate_id]),
                    candidate_id,
                    int(catalog.poi[candidate_id]),
                    int(catalog.category[candidate_id]),
                    int(catalog.city[candidate_id]),
                ),
                features,
                tuple(history[-8:]),
                "+".join(routes),
                propensity,
                score,
                int(rank_by_index[candidate_index]),
                candidate_id == row.item_id,
                observable,
                {"fine": score, "value_tree": score},
                {"feature": "stateful-v1", "model": model_name, "code": "simulation-v1"},
            )
        )
    return decisions


def _row_events(row, catalog, timestamp: int, observable: bool):
    actions = []
    commerce = []
    clicks = []
    pixels = []
    item_id = row.item_id
    response = row.response
    action_specs = (
        ("play", response.play, 1, 1.0),
        ("play_3s", response.play_3s, 3, 1.0),
        ("stay_seconds", response.play, 4, response.stay_seconds),
        ("slide", response.slide, 5, 1.0),
        ("long_view", response.long_view, 30, 1.0),
        ("high_quality_long_view", response.high_quality_long_view, 31, 1.0),
        ("like", response.like, 35, 1.0),
        ("favorite", response.favorite, 36, 1.0),
        ("comment", response.comment, 37, 1.0),
        ("share", response.share, 38, 1.0),
        ("negative_feedback", response.negative_feedback, 39, 1.0),
        ("anchor_impression", response.anchor_impression, 40, 1.0),
        ("anchor_click", response.anchor_click, 41, 1.0),
        ("poi_detail", response.poi_detail, 50, 1.0),
        ("poi_favorite", response.poi_favorite, 70, 1.0),
    )
    actions.extend(
        ActionEvent(
            f"{row.request_id}-{name}", row.request_id, item_id,
            int(catalog.poi[item_id]), name, timestamp + delay, timestamp + delay + 2,
            value,
        )
        for name, happened, delay, value in action_specs
        if happened
    )
    open_loop = item_id % 4 == 0
    if response.order and not open_loop:
        commerce_specs = (("order", response.order, 300), ("payment", response.payment, 360))
        commerce.extend(
            CommerceEvent(
                f"{row.request_id}-{name}", row.request_id, item_id,
                int(catalog.poi[item_id]), name, f"o-{row.request_id}",
                f"p-{row.request_id}" if response.payment else None,
                timestamp + delay, timestamp + delay + 3
            )
            for name, happened, delay in commerce_specs
            if happened
        )
    if response.order and open_loop:
        click_id = f"click-{row.request_id}"
        identity = f"identity-{row.user_id}" if observable else None
        clicks.append(
            OutboundClick(
                click_id, row.request_id, item_id, int(catalog.poi[item_id]), identity,
                int(catalog.author[item_id]), timestamp + 100
            )
        )
        if response.pixel_conversion:
            pixels.append(
                PixelEvent(
                    f"pixel-{row.request_id}", f"conversion-{row.request_id}", identity,
                    int(catalog.author[item_id]), timestamp + 900, timestamp + 930,
                    click_id if observable else None
                )
            )
    return actions, commerce, clicks, pixels


def build_feed_joiner(
    config: SimulationConfig,
    catalog,
    observed: list[Trajectory],
    policies,
    assigned: np.ndarray,
):
    decisions: list[StageDecision] = []
    actions: list[ActionEvent] = []
    commerce: list[CommerceEvent] = []
    clicks: list[OutboundClick] = []
    pixels: list[PixelEvent] = []
    for user_index, trajectory in enumerate(observed[: config.joiner_users]):
        policy = policies[int(assigned[user_index])]
        history: list[tuple[float, ...]] = []
        for row in trajectory.rows:
            timestamp = 1_800_000_000 + user_index * 100_000 + row.session_id * 1_000 + row.request_index * 10
            observable = user_index % 10 != 0
            decisions.extend(
                _row_decisions(row, catalog, policy.name, history, timestamp, observable)
            )
            row_actions, row_commerce, row_clicks, row_pixels = _row_events(
                row, catalog, timestamp, observable
            )
            actions.extend(row_actions)
            commerce.extend(row_commerce)
            clicks.extend(row_clicks)
            pixels.extend(row_pixels)
            history.append(row.features[:8])
    report = EvolutionJoiner().build(
        decisions, actions, commerce, clicks, pixels, watermark=1_900_000_000
    )
    return report


def _joiner_report(config, catalog, observed, policies, assigned):
    report = build_feed_joiner(config, catalog, observed, policies, assigned)
    tasks = tuple(report.fine[0].labels) if report.fine else ()
    label_rates = {}
    mask_coverage = {}
    for task in tasks:
        eligible = [example for example in report.fine if example.label_masks[task]]
        mask_coverage[task] = len(eligible) / len(report.fine)
        label_rates[task] = (
            float(np.mean([example.labels[task] for example in eligible]))
            if eligible
            else None
        )
    return {
        "stage_decisions": len(report.coarse),
        "recall_examples": len(report.recall),
        "coarse_examples": len(report.coarse),
        "fine_examples": len(report.fine),
        "duplicate_events": report.duplicate_events,
        "orphan_events": report.orphan_events,
        "immature_task_labels": report.immature_task_labels,
        "fine_label_rates": label_rates,
        "fine_label_mask_coverage": mask_coverage,
        "pixel_attribution": asdict(report.attribution),
    }


def _fit_ladder(config, features, labels, user_ids, session_ids, propensities):
    split = int(len(labels) * 0.75)
    inverse_propensity = np.minimum(1.0 / np.maximum(propensities[:split], 1e-4), 20.0)
    policies = fit_policies(
        features[:split], labels[:split], config.seed, sample_weight=inverse_propensity
    )
    basic_columns = (0, 1, 2, 3, 5, 8, 9)
    sequence_columns = basic_columns + (4, 11)
    basic_lr = fit_logistic_policy(
        "lr_basic_features",
        features[:split],
        labels[:split],
        basic_columns,
        config.seed,
        inverse_propensity,
    )
    sequence_lr = fit_logistic_policy(
        "lr_plus_sequence",
        features[:split],
        labels[:split],
        sequence_columns,
        config.seed,
        inverse_propensity,
    )
    learned_policies = (basic_lr, sequence_lr, *policies)
    replay_deltas = serialized_replay_deltas(learned_policies, features[split:])
    ladder = (PopularPolicy(), HeuristicPolicy(), *learned_policies)
    offline = {}
    for policy in learned_policies:
        scores = policy.score(features[split:])
        offline[policy.name] = {
            **binary_metrics(labels[split:], scores),
            "user_gauc": grouped_auc(labels[split:], scores, user_ids[split:]),
            "session_gauc": grouped_auc(labels[split:], scores, session_ids[split:]),
        }
    return policies, ladder, offline, replay_deltas, inverse_propensity, split


def _evaluate_ladder(config, catalog, ladder, final_policies):
    experiment_user_ids = np.arange(config.users) + 10_000_000
    trajectories = {
        policy.name: run_population(
            config, catalog, policy, experiment_user_ids
        )
        for policy in ladder
    }
    ab_ladder = {}
    assignments = {}
    final_assignment = None
    final_potential = None
    for launch_index, (control_policy, treatment_policy) in enumerate(
        zip(ladder, ladder[1:])
    ):
        assignment_rng = np.random.default_rng(config.seed + 77 + launch_index)
        assigned = assignment_rng.random(config.users) < 0.5
        launch_name = f"{control_policy.name}_to_{treatment_policy.name}"
        assignments[launch_name] = assigned
        control = trajectories[control_policy.name]
        treatment = trajectories[treatment_policy.name]
        metrics, potential = experiment_metrics(control, treatment, assigned)
        ab_ladder[launch_name] = {
            "control": control_policy.name,
            "treatment": treatment_policy.name,
            "assignment": {
                "control_users": int((~assigned).sum()),
                "treatment_users": int(assigned.sum()),
            },
            "metrics": metrics,
            "randomization_audit": randomization_audit(
                potential, config.seed + 991 + launch_index
            ),
            "decision": launch_decision(metrics),
        }
        if treatment_policy is final_policies[1]:
            final_assignment = assigned
            final_potential = potential
    if final_assignment is None or final_potential is None:
        raise RuntimeError("final LR-to-XGBoost launch was not evaluated")
    control = trajectories[final_policies[0].name]
    treatment = trajectories[final_policies[1].name]
    metrics = ab_ladder[
        f"{final_policies[0].name}_to_{final_policies[1].name}"
    ]["metrics"]
    final_randomization_audit = ab_ladder[
        f"{final_policies[0].name}_to_{final_policies[1].name}"
    ]["randomization_audit"]
    observed = [
        treatment[i] if final_assignment[i] else control[i]
        for i in range(config.users)
    ]
    return (
        ab_ladder,
        final_assignment,
        metrics,
        final_randomization_audit,
        observed,
        trajectories,
        assignments,
    )


def _second_training_round(config, catalog, original_rows, ladder, trajectories, assignments):
    basic = next(policy for policy in ladder if policy.name == "lr_basic_features")
    sequence = next(policy for policy in ladder if policy.name == "lr_plus_sequence")
    launch_name = "lr_basic_features_to_lr_plus_sequence"
    assigned = assignments[launch_name]
    experiment_rows = [
        row
        for user_index, treatment_assigned in enumerate(assigned)
        for row in (
            trajectories[sequence.name][user_index].rows
            if treatment_assigned
            else trajectories[basic.name][user_index].rows
        )
    ]
    combined_rows = [*original_rows, *experiment_rows]
    features, labels, _, _, propensities = _example_arrays(combined_rows, config)
    original_count = len(original_rows)
    weights = np.ones(len(combined_rows), dtype=np.float32)
    weights[:original_count] = np.minimum(
        1.0 / np.maximum(propensities[:original_count], 1e-4), 20.0
    )
    round_two = fit_logistic_policy(
        "lr_plus_sequence_round2",
        features,
        labels,
        (0, 1, 2, 3, 5, 8, 9, 4, 11),
        config.seed + 2,
        weights,
    )
    audit_users = np.arange(config.users) + 20_000_000
    audit_trajectories = run_population(
        config, catalog, HeuristicPolicy(), audit_users
    )
    audit_rows = [row for trajectory in audit_trajectories for row in trajectory.rows]
    audit_features = np.asarray(
        [row.features for row in audit_rows], dtype=np.float32
    )
    fresh_users = np.arange(config.users) + 30_000_000
    control = run_population(config, catalog, sequence, fresh_users)
    treatment = run_population(config, catalog, round_two, fresh_users)
    round_assignment = np.random.default_rng(config.seed + 404).random(config.users) < 0.5
    metrics, potential = experiment_metrics(control, treatment, round_assignment)
    observed = [
        treatment[index] if round_assignment[index] else control[index]
        for index in range(config.users)
    ]
    return {
        "training_examples_before": original_count,
        "mature_experiment_examples": len(experiment_rows),
        "training_examples_after": len(combined_rows),
        "candidate_policy_audit": _candidate_policy_audit(
            audit_rows, (sequence, round_two)
        ),
        "shadow_replay_score_delta": serialized_replay_deltas(
            (round_two,), audit_features
        )[round_two.name],
        "ab_metrics": metrics,
        "randomization_audit": randomization_audit(potential, config.seed + 1404),
        "decision": launch_decision(metrics),
        "joiner": _joiner_report(
            config, catalog, observed, (sequence, round_two), round_assignment
        ),
    }


def _simulator_acceptance(report: dict[str, object]) -> dict[str, object]:
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
            and 0.10 <= distribution["lt_rate"] <= 0.50
            and 0.02 <= distribution["hlt_rate"] <= 0.20
        ),
        "long_view_probability_calibrated": abs(
            distribution["long_view_probability_calibration_gap"]
        )
        <= 0.03,
        "all_six_recall_routes_reach_coarse": set(
            cascade["route_candidate_coverage"]
        )
        == {"ann", "graph", "geo", "fresh", "long_tail", "popular"},
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


def run_closed_loop_experiment(
    config: SimulationConfig = SimulationConfig(),
    include_local: bool = False,
) -> dict[str, object]:
    catalog = build_catalog(config)
    rows, features, labels, user_ids, session_ids, propensities = _training_data(
        config, catalog
    )
    policies, ladder, offline, replay_deltas, inverse_propensity, split = _fit_ladder(
        config, features, labels, user_ids, session_ids, propensities
    )
    (
        ab_ladder,
        final_assignment,
        metrics,
        randomization_audit,
        observed,
        trajectories,
        assignments,
    ) = _evaluate_ladder(config, catalog, ladder, policies)
    policy_iteration = _second_training_round(
        config, catalog, rows, ladder, trajectories, assignments
    )
    sequence_policy = next(
        policy for policy in ladder if policy.name == "lr_plus_sequence"
    )
    report = {
        "runtime": {
            "environment_contract": "Gymnasium",
            "reference_wheel": "sardine-rec==1.0.8",
            "protocol": ("request", "session", "cross_session"),
        },
        "config": asdict(config),
        "logging_policy": "quality/affinity rule with randomized exploration",
        "training_examples": len(rows),
        "long_view_prevalence": float(labels.mean()),
        "propensity": {
            "minimum": float(propensities.min()),
            "median": float(np.median(propensities)),
            "inverse_weight_effective_sample_size": float(
                inverse_propensity.sum() ** 2 / np.square(inverse_propensity).sum()
            ),
        },
        "offline": offline,
        "candidate_policy_audit": _candidate_policy_audit(rows[split:], ladder),
        "cascade": _cascade_audit(rows),
        "behavior_distribution": _behavior_distribution(rows),
        "policy_runtime": {
            policy.name: {
                "training_device": policy.training_device,
                "serving_device": policy.serving_device,
            }
            for policy in policies
        },
        "shadow_replay_score_delta": replay_deltas,
        "offline_online_max_score_delta": max(replay_deltas.values()),
        "ab_assignment": {
            "control_users": int((~final_assignment).sum()),
            "treatment_users": int(final_assignment.sum()),
        },
        "ab_metrics": metrics,
        "ab_ladder": ab_ladder,
        "policy_iteration": policy_iteration,
        "single_experiment_truth_covered": all(
            metric["confidence_interval"][0]
            <= metric["true_itt"]
            <= metric["confidence_interval"][1]
            for metric in metrics.values()
        ),
        "randomization_audit": randomization_audit,
        "estimator_audit_passed": all(
            value["truth_inside_randomization_interval"]
            for value in randomization_audit.values()
        ),
        "launch_decision": launch_decision(metrics),
        "joiner": _joiner_report(
            config, catalog, observed, policies, final_assignment
        ),
        "limitations": (
            "The environment validates mechanics and estimator recovery under explicit dynamics; "
            "it does not establish real production lift without logged-data calibration and a live randomized test."
        ),
    }
    if include_local:
        report["supply_iteration"] = run_supply_iteration(
            config,
            catalog,
            trajectories[sequence_policy.name],
            sequence_policy,
        )
    report["simulator_acceptance"] = _simulator_acceptance(report)
    return report
