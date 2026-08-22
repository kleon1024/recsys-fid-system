"""Thin policy adapters around mature scikit-learn and XGBoost models."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier


class HeuristicPolicy:
    name = "quality_affinity_rule"

    def score(self, features: np.ndarray) -> np.ndarray:
        return (
            1.8 * features[:, 0]
            + 0.7 * features[:, 1]
            + 0.2 * features[:, 3]
            + 0.1 * features[:, 4]
            - 0.5 * features[:, 6]
        )


class LearnedPolicy:
    def __init__(self, name: str, model, training_device: str, serving_device: str) -> None:
        self.name = name
        self.model = model
        self.training_device = training_device
        self.serving_device = serving_device

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]


def fit_policies(features: np.ndarray, labels: np.ndarray, seed: int):
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
    logistic.fit(features, labels)
    xgboost.fit(features, labels)
    # Candidate batches are small NumPy arrays. Make the CPU serving contract
    # explicit after GPU training instead of accepting an implicit device fallback.
    xgboost.set_params(device="cpu")
    return (
        LearnedPolicy("logistic_regression", logistic, "cpu", "cpu"),
        LearnedPolicy("xgboost", xgboost, device, "cpu"),
    )


def serialized_replay_delta(policies, features: np.ndarray) -> float:
    control, treatment = policies
    before = (control.score(features), treatment.score(features))
    with TemporaryDirectory() as directory:
        logistic_path = Path(directory) / "control.joblib"
        xgboost_path = Path(directory) / "treatment.json"
        joblib.dump(control.model, logistic_path)
        treatment.model.save_model(xgboost_path)
        loaded_logistic = joblib.load(logistic_path)
        loaded_xgboost = XGBClassifier(device="cpu")
        loaded_xgboost.load_model(xgboost_path)
        after = (
            loaded_logistic.predict_proba(features)[:, 1],
            loaded_xgboost.predict_proba(features)[:, 1],
        )
    return max(float(np.max(np.abs(left - right))) for left, right in zip(before, after))
