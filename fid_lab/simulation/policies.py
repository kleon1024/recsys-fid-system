"""Thin policy adapters around mature scikit-learn and XGBoost models."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier, XGBRegressor


class HeuristicPolicy:
    name = "quality_affinity_rule"

    def score(self, features: np.ndarray) -> np.ndarray:
        return (
            1.8 * features[:, 0]
            + 0.7 * features[:, 1]
            + 0.2 * features[:, 3]
            + 0.1 * features[:, 5]
        )


class PopularPolicy:
    name = "popular_baseline"

    def score(self, features: np.ndarray) -> np.ndarray:
        return features[:, 3]


class ParameterizedPolicy:
    """Bind one resolved full-chain snapshot to the actual fine/value score."""

    def __init__(self, base, parameters) -> None:
        if parameters.fine_model != base.name:
            raise ValueError(
                f"resolved fine_model {parameters.fine_model} does not match {base.name}"
            )
        self.base = base
        self.parameters = parameters
        self.name = parameters.fine_model

    def score(self, features: np.ndarray) -> np.ndarray:
        raw = self.base.score(features) / self.parameters.calibration_temperature
        interaction_proxy = 0.5 * features[:, 0] + 0.5 * features[:, 1]
        negative_proxy = 0.5 * features[:, 7] - 0.5 * features[:, 0]
        value = (
            self.parameters.stay_weight * features[:, 0]
            + self.parameters.long_view_weight * features[:, 4]
            + self.parameters.hlt_weight * features[:, 1]
            + self.parameters.interaction_weight * interaction_proxy
            + self.parameters.negative_weight * negative_proxy
        )
        diversity_penalty = self.parameters.diversity_strength * features[:, 11]
        return raw + 0.10 * value - diversity_penalty


class LocalConstrainedValuePolicy:
    """Boost Local value only inside an explicit Feed-value tolerance."""

    name = "feed_guarded_local_value_tree"

    def __init__(
        self,
        feed_policy,
        local_weight: float = 0.15,
        maximum_feed_score_loss: float = 0.03,
    ) -> None:
        self.feed_policy = feed_policy
        self.local_weight = local_weight
        self.maximum_feed_score_loss = maximum_feed_score_loss

    def score(self, features: np.ndarray) -> np.ndarray:
        feed_score = self.feed_policy.score(features)
        local_proxy = features[:, 13] * (
            0.35 * features[:, 0]
            + 0.25 * features[:, 2]
            + 0.20 * features[:, 5]
            + 0.20 * features[:, 1]
        )
        eligible = feed_score >= float(feed_score.max()) - self.maximum_feed_score_loss
        return feed_score + self.local_weight * local_proxy * eligible


class LocalIntentPolicy:
    """Incremental Local rank policy with explicit signal-ablation controls."""

    def __init__(
        self,
        feed_policy,
        name: str,
        local_weight: float,
        search_weight: float = 0.0,
        retarget_weight: float = 0.0,
        intent_quality_weight: float = 0.0,
        embedding_correction_weight: float = 0.0,
    ) -> None:
        self.feed_policy = feed_policy
        self.name = name
        self.local_weight = local_weight
        self.search_weight = search_weight
        self.retarget_weight = retarget_weight
        self.intent_quality_weight = intent_quality_weight
        self.embedding_correction_weight = embedding_correction_weight

    def score(self, features: np.ndarray) -> np.ndarray:
        feed_score = self.feed_policy.score(features)
        poi = features[:, 13]
        static_local = (
            0.24 * features[:, 0]
            + 0.18 * features[:, 2]
            + 0.14 * features[:, 5]
            + 0.16 * features[:, 20]
            + 0.14 * features[:, 21]
            + 0.14 * features[:, 22]
        )
        intent = (
            static_local
            + self.search_weight * features[:, 18]
            + self.retarget_weight * features[:, 19]
        )
        intent_match = np.clip(features[:, 18] + features[:, 19], 0.0, 1.0)
        intent_quality = (
            0.35 * features[:, 23]
            + 0.25 * features[:, 1]
            + 0.20 * features[:, 20]
            + 0.10 * features[:, 2]
            + 0.10 * features[:, 5]
        )
        correction = features[:, 23] - features[:, 0]
        return (
            feed_score
            + self.local_weight * poi * intent
            + self.intent_quality_weight * poi * intent_match * intent_quality
            + self.embedding_correction_weight
            * poi
            * intent_match
            * correction
        )


class LearnedPolicy:
    def __init__(
        self,
        name: str,
        model,
        training_device: str,
        serving_device: str,
        columns: tuple[int, ...] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.training_device = training_device
        self.serving_device = serving_device
        self.columns = columns

    def score(self, features: np.ndarray) -> np.ndarray:
        model_features = features if self.columns is None else features[:, self.columns]
        return self.model.predict_proba(model_features)[:, 1]


class LearnedRegressionPolicy(LearnedPolicy):
    def score(self, features: np.ndarray) -> np.ndarray:
        model_features = features if self.columns is None else features[:, self.columns]
        return self.model.predict(model_features)


class GuardedBlendPolicy:
    def __init__(
        self,
        name: str,
        base,
        challenger,
        candidates: int,
        base_score_tolerance: float,
    ) -> None:
        self.name = name
        self.base = base
        self.challenger = challenger
        self.candidates = candidates
        self.base_score_tolerance = base_score_tolerance

    def score(self, features: np.ndarray) -> np.ndarray:
        base = self.base.score(features)
        challenger = self.challenger.score(features)
        if len(features) % self.candidates:
            return base + 0.05 * challenger
        base_matrix = base.reshape(-1, self.candidates)
        challenger_matrix = challenger.reshape(-1, self.candidates)
        eligible = base_matrix >= (
            base_matrix.max(axis=1, keepdims=True) - self.base_score_tolerance
        )
        constrained = np.where(eligible, challenger_matrix, -1e6)
        return constrained.reshape(-1)


def fit_logistic_policy(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    columns: tuple[int, ...],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LearnedPolicy:
    model = LogisticRegression(max_iter=300, random_state=seed)
    model.fit(features[:, columns], labels, sample_weight=sample_weight)
    return LearnedPolicy(name, model, "cpu", "cpu", columns)


def fit_policies(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logistic = LogisticRegression(max_iter=300, random_state=seed)
    xgboost = XGBClassifier(
        n_estimators=140,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=device,
        n_jobs=4,
        random_state=seed,
    )
    logistic.fit(features, labels, sample_weight=sample_weight)
    xgboost.fit(features, labels, sample_weight=sample_weight)
    # Candidate batches are small NumPy arrays. Make the CPU serving contract
    # explicit after GPU training instead of accepting an implicit device fallback.
    xgboost.set_params(device="cpu", n_jobs=1)
    return (
        LearnedPolicy("logistic_regression", logistic, "cpu", "cpu"),
        LearnedPolicy("xgboost", xgboost, device, "cpu"),
    )


def fit_xgboost_policy(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
    name: str = "xgboost",
) -> LearnedPolicy:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XGBClassifier(
        n_estimators=140,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=device,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    model.set_params(device="cpu", n_jobs=1)
    return LearnedPolicy(name, model, device, "cpu")


def fit_xgboost_regression_policy(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
    name: str = "xgboost_stay",
) -> LearnedRegressionPolicy:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XGBRegressor(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=device,
        n_jobs=4,
        random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(features, targets, sample_weight=sample_weight)
    model.set_params(device="cpu", n_jobs=1)
    return LearnedRegressionPolicy(name, model, device, "cpu")


def serialized_replay_deltas(policies, features: np.ndarray) -> dict[str, float]:
    report = {}
    with TemporaryDirectory() as directory:
        for policy in policies:
            before = policy.score(features)
            model_features = (
                features if policy.columns is None else features[:, policy.columns]
            )
            if isinstance(policy.model, XGBClassifier):
                path = Path(directory) / f"{policy.name}.json"
                policy.model.save_model(path)
                loaded = XGBClassifier(device="cpu", n_jobs=1)
                loaded.load_model(path)
            else:
                path = Path(directory) / f"{policy.name}.joblib"
                joblib.dump(policy.model, path)
                loaded = joblib.load(path)
            after = loaded.predict_proba(model_features)[:, 1]
            report[policy.name] = float(np.max(np.abs(before - after)))
    return report
