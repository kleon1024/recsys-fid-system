"""Connect actual model scores to slate choice and simulated online outcomes."""

from __future__ import annotations

from dataclasses import asdict
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ...scale.synthetic import ScaleDataset
from .ab_simulator import metric_lift
from .metrics import binary_metrics


def _features(dataset: ScaleDataset) -> np.ndarray:
    sparse = (dataset.sparse_ids % 128).astype(np.float32) / 127.0
    return np.concatenate([sparse, dataset.dense_features], axis=1)


def _fit_models(features: np.ndarray, labels: np.ndarray, train_end: int):
    logistic = LogisticRegression(max_iter=200, random_state=29)
    xgboost = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        tree_method="hist",
        n_jobs=4,
        random_state=29,
    )
    logistic.fit(features[:train_end], labels[:train_end])
    xgboost.fit(features[:train_end], labels[:train_end])
    return logistic, xgboost


def _replay(models, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    logistic, xgboost = models
    offline_control = logistic.predict_proba(features)[:, 1]
    offline_treatment = xgboost.predict_proba(features)[:, 1]
    with TemporaryDirectory() as directory:
        logistic_path = f"{directory}/logistic.joblib"
        xgboost_path = f"{directory}/xgboost.json"
        joblib.dump(logistic, logistic_path)
        xgboost.save_model(xgboost_path)
        served_control = joblib.load(logistic_path).predict_proba(features)[:, 1]
        served_model = XGBClassifier()
        served_model.load_model(xgboost_path)
        served_treatment = served_model.predict_proba(features)[:, 1]
    delta = max(
        float(np.max(np.abs(offline_control - served_control))),
        float(np.max(np.abs(offline_treatment - served_treatment))),
    )
    return served_control, served_treatment, delta


def run_model_coupled_ab(
    dataset: ScaleDataset,
    slate_size: int = 20,
    seed: int = 20260823,
) -> dict[str, object]:
    features = _features(dataset)
    train_end = int(dataset.examples * 0.70)
    test_start = int(dataset.examples * 0.85)
    labels = dataset.labels[:, 0]
    models = _fit_models(features, labels, train_end)
    control_score, treatment_score, replay_delta = _replay(
        models, features[test_start:]
    )
    usable = len(control_score) // slate_size * slate_size
    control_matrix = control_score[:usable].reshape(-1, slate_size)
    treatment_matrix = treatment_score[:usable].reshape(-1, slate_size)
    control_choice = control_matrix.argmax(axis=1)
    treatment_choice = treatment_matrix.argmax(axis=1)
    rows = np.arange(len(control_choice))
    task_indices = {"long_view": 0, "order": 4, "negative_feedback": 5}
    probabilities = dataset.label_probabilities[test_start : test_start + usable]
    probabilities = probabilities.reshape(-1, slate_size, probabilities.shape[1])
    rng = np.random.default_rng(seed)
    assigned = rng.random(len(rows)) < 0.5
    metrics = {}
    for task, task_index in task_indices.items():
        control_probability = probabilities[rows, control_choice, task_index]
        treatment_probability = probabilities[rows, treatment_choice, task_index]
        common_random = rng.random(len(rows))
        control_outcome = (common_random < control_probability).astype(float)
        treatment_outcome = (common_random < treatment_probability).astype(float)
        metrics[task] = asdict(
            metric_lift(
                control_outcome[~assigned],
                treatment_outcome[assigned],
                control_probability,
                treatment_probability,
            )
        )
    test_labels = labels[test_start : test_start + usable]
    return {
        "control_model": "scikit-learn LogisticRegression",
        "treatment_model": "XGBoost",
        "candidate_budget": slate_size,
        "requests": len(rows),
        "offline_auc": {
            "control": binary_metrics(test_labels, control_score[:usable])["auc"],
            "treatment": binary_metrics(test_labels, treatment_score[:usable])["auc"],
        },
        "offline_online_max_score_delta": replay_delta,
        "different_top_choice_rate": float((control_choice != treatment_choice).mean()),
        "metrics": metrics,
        "all_truth_covered": all(
            values["confidence_interval"][0]
            <= values["true_itt"]
            <= values["confidence_interval"][1]
            for values in metrics.values()
        ),
    }
