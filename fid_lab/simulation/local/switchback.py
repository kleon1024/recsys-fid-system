"""City-period switchback for posting-supply interference and platform LT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt

import numpy as np

from ...value import LTMetricContainer


@dataclass(frozen=True)
class SupplySwitchbackConfig:
    cities: int = 100
    periods: int = 28
    block_periods: int = 2
    users_per_city_period: int = 10_000
    seed: int = 20260823


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


def _assign(config: SupplySwitchbackConfig) -> tuple[np.ndarray, np.ndarray]:
    city = np.arange(config.cities)[:, None]
    period = np.arange(config.periods)[None, :]
    start = (city * 1_664_525 + config.seed) % 2
    block = period // config.block_periods
    treatment = (start + block) % 2 == 1
    switched = np.zeros_like(treatment)
    switched[:, 1:] = treatment[:, 1:] != treatment[:, :-1]
    return treatment, switched


def _potential_worlds(config: SupplySwitchbackConfig):
    rng = np.random.default_rng(config.seed)
    city_demand = rng.normal(0.0, 0.55, (config.cities, 1))
    period = np.arange(config.periods)[None, :]
    season = 0.16 * np.sin(2.0 * np.pi * period / 7.0)
    shock = rng.normal(0.0, 0.08, (config.cities, config.periods))
    base_penetration = _sigmoid(-4.25 + 0.28 * city_demand + season + shock)
    treatment_penetration = _sigmoid(
        -4.25 + 0.28 * city_demand + season + shock + 0.20
    )
    base_quality = np.clip(0.55 + 0.08 * city_demand + 0.03 * shock, 0.1, 0.95)
    treatment_quality = np.clip(base_quality + 0.025, 0.1, 0.98)
    supply_zero = base_penetration * base_quality
    current_supply_one = treatment_penetration * treatment_quality
    supply_delta = current_supply_one - supply_zero
    # The first period after a switch is excluded. Remaining treatment periods
    # observe both current supply and a bounded carryover from the prior period.
    supply_one = supply_zero + 1.25 * supply_delta

    stay_zero = 42.0 + 0.9 * city_demand + 0.25 * season + 0.12 * shock
    stay_one = stay_zero + 0.75 * supply_delta
    active_zero = _sigmoid(-0.62 + 0.12 * city_demand + 0.05 * season)
    active_one = np.clip(active_zero + 0.0025 * supply_delta, 0.0, 1.0)
    poi_vv_zero = 0.16 + 0.22 * supply_zero
    poi_vv_one = 0.16 + 0.22 * supply_one
    local_commerce_zero = 0.018 + 0.10 * supply_zero
    local_commerce_one = 0.018 + 0.10 * supply_one
    return {
        "posting_penetration": (base_penetration, treatment_penetration),
        "quality_adjusted_supply": (supply_zero, supply_one),
        "poi_video_vv_per_user": (poi_vv_zero, poi_vv_one),
        "stay_seconds_per_user": (stay_zero, stay_one),
        "active_days_per_user": (active_zero, active_one),
        "local_commercialization_per_user": (
            local_commerce_zero,
            local_commerce_one,
        ),
    }


def _lt_world(stay: np.ndarray, active: np.ndarray) -> np.ndarray:
    container = LTMetricContainer()
    stay_rate = container.config.rates["stay_minute"].unit_value
    active_rate = container.config.rates["active_day"].unit_value
    return stay / 60.0 * stay_rate + active * active_rate


class _TwoWayFixedEffectEstimator:
    """Reusable two-way FE design with city-clustered CR1 uncertainty."""

    def __init__(self, shape, assigned: np.ndarray, eligible: np.ndarray) -> None:
        city, period = np.indices(shape)
        self.keep = eligible.ravel()
        treatment = assigned.ravel()[self.keep].astype(float)
        self.city_index = city.ravel()[self.keep]
        period_index = period.ravel()[self.keep]
        columns = [np.ones(len(treatment)), treatment]
        columns.extend(
            (self.city_index == value).astype(float) for value in range(1, shape[0])
        )
        columns.extend(
            (period_index == value).astype(float) for value in range(1, shape[1])
        )
        self.design = np.column_stack(columns)
        self.solver = np.linalg.pinv(self.design)
        self.bread = np.linalg.pinv(self.design.T @ self.design)
        self.clusters = np.unique(self.city_index)
        observations, parameters = self.design.shape
        self.correction = len(self.clusters) / (len(self.clusters) - 1.0)
        self.correction *= (observations - 1.0) / (observations - parameters)

    def fit(self, outcome: np.ndarray) -> tuple[float, float, int]:
        y = outcome.ravel()[self.keep]
        coefficient = self.solver @ y
        residual = y - self.design @ coefficient
        meat = np.zeros((self.design.shape[1], self.design.shape[1]))
        for cluster in self.clusters:
            mask = self.city_index == cluster
            score = self.design[mask].T @ residual[mask]
            meat += np.outer(score, score)
        covariance = self.correction * self.bread @ meat @ self.bread
        return (
            float(coefficient[1]),
            sqrt(max(float(covariance[1, 1]), 0.0)),
            len(self.clusters),
        )


def _estimate_metric(zero, one, observed, assigned, eligible, estimator):
    estimate, standard_error, cities = estimator.fit(observed)
    noiseless_observed = np.where(assigned, one, zero)
    true_effect, _, _ = estimator.fit(noiseless_observed)
    interval = (
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "p_value": erfc(abs(estimate / max(standard_error, 1e-12)) / sqrt(2.0)),
        "confidence_interval": interval,
        "known_dgp_effect": true_effect,
        "truth_inside_confidence_interval": bool(
            interval[0] <= true_effect <= interval[1]
        ),
        "relative_effect": true_effect / float(zero[eligible].mean()),
        "cities": cities,
    }


def calibrate_supply_switchback(
    config: SupplySwitchbackConfig = SupplySwitchbackConfig(),
    simulations: int = 200,
) -> dict[str, object]:
    """Repeated-DGP coverage audit for the exact production estimator."""
    if simulations < 2:
        raise ValueError("switchback calibration requires at least two simulations")
    records: dict[str, list[tuple[float, float, bool, float]]] = {}
    for offset in range(simulations):
        report = run_supply_switchback(
            SupplySwitchbackConfig(
                cities=config.cities,
                periods=config.periods,
                block_periods=config.block_periods,
                users_per_city_period=config.users_per_city_period,
                seed=config.seed + offset,
            )
        )
        for name, metric in report["metrics"].items():
            records.setdefault(name, []).append(
                (
                    metric["estimate"],
                    metric["known_dgp_effect"],
                    metric["truth_inside_confidence_interval"],
                    metric["p_value"],
                )
            )
    metrics = {}
    for name, values in records.items():
        estimates = np.asarray([value[0] for value in values])
        truths = np.asarray([value[1] for value in values])
        error = estimates - truths
        metrics[name] = {
            "coverage_rate": float(np.mean([value[2] for value in values])),
            "mean_estimate": float(estimates.mean()),
            "mean_known_dgp_effect": float(truths.mean()),
            "bias": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "significant_rate": float(np.mean([value[3] < 0.05 for value in values])),
        }
    return {
        "simulations": simulations,
        "estimator": "two-way fixed effects with city-clustered CR1 standard errors",
        "metrics": metrics,
    }


def run_supply_switchback(
    config: SupplySwitchbackConfig = SupplySwitchbackConfig(),
) -> dict[str, object]:
    assigned, switched = _assign(config)
    worlds = _potential_worlds(config)
    worlds["lt_value_per_user"] = (
        _lt_world(
            worlds["stay_seconds_per_user"][0],
            worlds["active_days_per_user"][0],
        ),
        _lt_world(
            worlds["stay_seconds_per_user"][1],
            worlds["active_days_per_user"][1],
        ),
    )
    eligible = ~switched
    selected = {
        name: np.where(assigned, one, zero)
        for name, (zero, one) in worlds.items()
    }
    rng = np.random.default_rng(config.seed + 77)
    users = config.users_per_city_period
    selected["posting_penetration"] += rng.normal(
        0.0,
        np.sqrt(
            np.clip(selected["posting_penetration"] * (1.0 - selected["posting_penetration"]), 0.0, 1.0)
            / users
        ),
    )
    selected["quality_adjusted_supply"] += rng.normal(
        0.0, 0.08 / sqrt(users), selected["quality_adjusted_supply"].shape
    )
    selected["poi_video_vv_per_user"] += rng.normal(
        0.0,
        np.sqrt(np.clip(selected["poi_video_vv_per_user"], 0.0, None) / users),
    )
    selected["stay_seconds_per_user"] += rng.normal(
        0.0, 35.0 / sqrt(users), selected["stay_seconds_per_user"].shape
    )
    selected["active_days_per_user"] += rng.normal(
        0.0,
        np.sqrt(
            np.clip(selected["active_days_per_user"] * (1.0 - selected["active_days_per_user"]), 0.0, 1.0)
            / users
        ),
    )
    selected["local_commercialization_per_user"] += rng.normal(
        0.0,
        np.sqrt(
            np.clip(selected["local_commercialization_per_user"], 0.0, None)
            / users
        ),
    )
    selected["lt_value_per_user"] = _lt_world(
        selected["stay_seconds_per_user"], selected["active_days_per_user"]
    )
    estimator = _TwoWayFixedEffectEstimator(assigned.shape, assigned, eligible)
    metrics = {
        name: _estimate_metric(
            zero, one, selected[name], assigned, eligible, estimator
        )
        for name, (zero, one) in worlds.items()
    }
    lt = metrics["lt_value_per_user"]
    if not lt["truth_inside_confidence_interval"]:
        decision = "hold_estimator_miss"
    elif lt["estimate"] > 0.0 and lt["p_value"] < 0.05:
        decision = "pass_platform_lt"
    else:
        decision = "hold_platform_lt"
    return {
        "launch_id": "L-LOCAL-SUPPLY-SWITCHBACK-002",
        "config": asdict(config),
        "assignment": "city-period alternating blocks",
        "estimator": "two-way fixed effects with city-clustered CR1 standard errors",
        "washout": "first period after every block switch excluded",
        "effective_user_periods": int(eligible.sum())
        * config.users_per_city_period,
        "metrics": metrics,
        "decision": decision,
        "invariant": (
            "Local supply and commercialization remain business metrics; "
            "only stay and active-day effects enter LT."
        ),
    }
