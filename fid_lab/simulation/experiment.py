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
from ..evolution.evaluation.ab_simulator import metric_lift
from ..evolution.evaluation.metrics import binary_metrics, grouped_auc
from .contracts import SimulationConfig, TraceRow, Trajectory
from .environment import StatefulFeedEnv, build_catalog
from .policies import HeuristicPolicy, fit_policies, serialized_replay_delta


def _run_user(env: StatefulFeedEnv, policy, user_id: int, explore: bool = False) -> Trajectory:
    observation, _ = env.reset(seed=env.config.seed + user_id, options={"user_id": user_id})
    rows = []
    watch = 0.0
    clicks = orders = negatives = returned_sessions = 0
    discounted_value = 0.0
    completed_sessions = 1
    terminated = False
    while not terminated:
        source_session = env.session_id
        source_request = env.request_index
        candidate_ids = env.candidates.copy()
        scores = policy.score(observation)
        decision_rng = np.random.default_rng(
            env.config.seed + user_id * 65_537 + source_session * 101 + source_request
        )
        if explore and decision_rng.random() < env.config.exploration_rate:
            action = int(decision_rng.integers(0, len(scores)))
        else:
            action = int(np.argmax(scores))
        _, _, terminated, _, info = env.step(action)
        response = info["response"]
        returned = bool(info["session_end"] and info["returned"])
        request_id = f"u{user_id}-s{source_session}-r{source_request}"
        rows.append(
            TraceRow(
                user_id,
                source_session,
                source_request,
                request_id,
                int(candidate_ids[action]),
                tuple(int(value) for value in candidate_ids),
                tuple(tuple(float(value) for value in candidate) for candidate in observation),
                tuple(float(value) for value in scores),
                tuple(float(value) for value in observation[action]),
                float(scores[action]),
                response,
                returned,
            )
        )
        watch += response.watch_minutes
        clicks += int(response.anchor_click)
        orders += int(response.payment)
        negatives += int(response.negative_feedback)
        discounted_value += (
            response.watch_minutes
            + 2.0 * response.anchor_click
            + 12.0 * response.payment
            - 4.0 * response.negative_feedback
        ) * (0.92**source_session)
        if returned:
            returned_sessions += 1
            completed_sessions += 1
        observation = env._observation() if not terminated else observation
    return Trajectory(
        tuple(rows), completed_sessions, returned_sessions, watch, clicks, orders, negatives, discounted_value
    )


def _training_data(config: SimulationConfig, env: StatefulFeedEnv):
    rows = []
    policy = HeuristicPolicy()
    for user_id in range(config.users):
        rows.extend(_run_user(env, policy, user_id, explore=True).rows)
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([row.response.long_view for row in rows], dtype=np.int8)
    users = np.asarray([row.user_id for row in rows], dtype=np.int64)
    sessions = np.asarray(
        [row.user_id * config.max_sessions + row.session_id for row in rows], dtype=np.int64
    )
    return rows, features, labels, users, sessions


def _bayesian_bootstrap_probability(control: np.ndarray, treatment: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    positive = 0
    draws = 800
    for _ in range(draws):
        control_weight = rng.exponential(size=len(control))
        treatment_weight = rng.exponential(size=len(treatment))
        control_mean = float(np.average(control, weights=control_weight))
        treatment_mean = float(np.average(treatment, weights=treatment_weight))
        positive += int(treatment_mean > control_mean)
    return positive / draws


def _experiment_metrics(control: list[Trajectory], treatment: list[Trajectory], assigned: np.ndarray):
    fields = {
        "watch_minutes": "watch_minutes",
        "anchor_clicks": "anchor_clicks",
        "orders": "orders",
        "negative_feedback": "negative_feedback",
        "sessions": "sessions",
        "long_term_value": "discounted_value",
    }
    report = {}
    potential_outcomes = {}
    for name, attribute in fields.items():
        zero = np.asarray([getattr(value, attribute) for value in control], dtype=float)
        one = np.asarray([getattr(value, attribute) for value in treatment], dtype=float)
        potential_outcomes[name] = (zero, one)
        lift = asdict(metric_lift(zero[~assigned], one[assigned], zero, one))
        lift["posterior_probability_positive"] = _bayesian_bootstrap_probability(
            zero[~assigned], one[assigned], 91 + len(report)
        )
        report[name] = lift
    return report, potential_outcomes


def _randomization_audit(potential_outcomes, seed: int, draws: int = 500):
    """Verify the user-level A/B estimator over assignments without rerunning users."""
    rng = np.random.default_rng(seed)
    report = {}
    for name, (zero, one) in potential_outcomes.items():
        estimates = []
        users = len(zero)
        treatment_count = users // 2
        for _ in range(draws):
            treatment_index = rng.choice(users, treatment_count, replace=False)
            assigned = np.zeros(users, dtype=bool)
            assigned[treatment_index] = True
            estimates.append(float(one[assigned].mean() - zero[~assigned].mean()))
        estimates_array = np.asarray(estimates)
        true_itt = float((one - zero).mean())
        interval = np.quantile(estimates_array, (0.025, 0.975))
        report[name] = {
            "true_itt": true_itt,
            "mean_estimate": float(estimates_array.mean()),
            "estimator_bias": float(estimates_array.mean() - true_itt),
            "randomization_interval": tuple(float(value) for value in interval),
            "truth_inside_randomization_interval": bool(interval[0] <= true_itt <= interval[1]),
        }
    return report


def _row_decisions(row, catalog, model_name: str, history, timestamp: int, observable: bool):
    order = np.argsort(-np.asarray(row.candidate_scores))
    rank_by_index = np.empty(len(order), dtype=int)
    rank_by_index[order] = np.arange(1, len(order) + 1)
    decisions = []
    for candidate_index, (candidate_id, features, score) in enumerate(
        zip(row.candidate_ids, row.candidate_features, row.candidate_scores)
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
                "ann+graph+geo+fresh",
                1.0,
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
        ("long_view", response.long_view, 30),
        ("anchor_click", response.anchor_click, 40),
        ("detail", response.detail, 50),
        ("favorite", response.favorite, 70),
    )
    actions.extend(
        ActionEvent(
            f"{row.request_id}-{name}", row.request_id, item_id,
            int(catalog.poi[item_id]), name, timestamp + delay, timestamp + delay + 2
        )
        for name, happened, delay in action_specs
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


def _joiner_report(
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
    return {
        "stage_decisions": len(decisions),
        "recall_examples": len(report.recall),
        "coarse_examples": len(report.coarse),
        "fine_examples": len(report.fine),
        "duplicate_events": report.duplicate_events,
        "orphan_events": report.orphan_events,
        "immature_task_labels": report.immature_task_labels,
        "pixel_attribution": asdict(report.attribution),
    }


def run_closed_loop_experiment(config: SimulationConfig = SimulationConfig()) -> dict[str, object]:
    catalog = build_catalog(config)
    env = StatefulFeedEnv(config, catalog)
    rows, features, labels, user_ids, session_ids = _training_data(config, env)
    split = int(len(labels) * 0.75)
    policies = fit_policies(features[:split], labels[:split], config.seed)
    replay_delta = serialized_replay_delta(policies, features[split:])
    offline = {}
    for policy in policies:
        scores = policy.score(features[split:])
        offline[policy.name] = {
            **binary_metrics(labels[split:], scores),
            "user_gauc": grouped_auc(labels[split:], scores, user_ids[split:]),
            "session_gauc": grouped_auc(labels[split:], scores, session_ids[split:]),
        }
    experiment_user_ids = np.arange(config.users) + 10_000_000
    control = [_run_user(env, policies[0], int(user_id)) for user_id in experiment_user_ids]
    treatment = [_run_user(env, policies[1], int(user_id)) for user_id in experiment_user_ids]
    assignment_rng = np.random.default_rng(config.seed + 77)
    assigned = assignment_rng.random(config.users) < 0.5
    observed = [treatment[i] if assigned[i] else control[i] for i in range(config.users)]
    metrics, potential_outcomes = _experiment_metrics(control, treatment, assigned)
    randomization_audit = _randomization_audit(
        potential_outcomes, config.seed + 991
    )
    negative_guardrail = metrics["negative_feedback"]
    launch_decision = (
        "reject: negative-feedback guardrail regressed"
        if negative_guardrail["absolute_lift"] > 0.0
        and negative_guardrail["p_value"] < 0.05
        else "continue staged rollout: no significant guardrail regression"
    )
    return {
        "runtime": {
            "environment_contract": "Gymnasium",
            "reference_wheel": "sardine-rec==1.0.8",
            "protocol": ("request", "session", "cross_session"),
        },
        "config": asdict(config),
        "logging_policy": "quality/affinity rule with randomized exploration",
        "training_examples": len(rows),
        "long_view_prevalence": float(labels.mean()),
        "offline": offline,
        "policy_runtime": {
            policy.name: {
                "training_device": policy.training_device,
                "serving_device": policy.serving_device,
            }
            for policy in policies
        },
        "offline_online_max_score_delta": replay_delta,
        "ab_assignment": {
            "control_users": int((~assigned).sum()),
            "treatment_users": int(assigned.sum()),
        },
        "ab_metrics": metrics,
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
        "launch_decision": launch_decision,
        "joiner": _joiner_report(config, catalog, observed, policies, assigned),
        "limitations": (
            "The environment validates mechanics and estimator recovery under explicit dynamics; "
            "it does not establish real production lift without logged-data calibration and a live randomized test."
        ),
    }
