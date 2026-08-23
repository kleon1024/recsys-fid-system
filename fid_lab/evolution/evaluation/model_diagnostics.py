"""Oracle headroom and feature-ablation diagnostics for the synthetic DGP."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ...scale.synthetic import ScaleDataset
from .metrics import binary_metrics


def _fit_auc(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    model,
) -> float:
    model.fit(train_x, train_y)
    parameters = model.get_params()
    if str(parameters.get("device", "cpu")).startswith("cuda"):
        model.set_params(device="cpu", n_jobs=1)
    return binary_metrics(test_y, model.predict_proba(test_x)[:, 1])["auc"]


def diagnose_signal(
    dataset: ScaleDataset,
    device: str = "cpu",
) -> dict[str, object]:
    train_end = int(dataset.examples * 0.70)
    test_start = int(dataset.examples * 0.85)
    labels = dataset.labels[:, 0]
    dense = dataset.dense_features
    engineered = np.concatenate([dense, dataset.diagnostic_features], axis=1)
    sparse = (dataset.sparse_ids % 128).astype(np.float32) / 127.0
    full = np.concatenate([engineered, sparse], axis=1)
    test_labels = labels[test_start:]
    oracle_auc = binary_metrics(
        test_labels, dataset.label_probabilities[test_start:, 0]
    )["auc"]
    models = {
        "linear_dense": (
            dense,
            LogisticRegression(max_iter=200, random_state=17),
        ),
        "linear_with_known_cross_and_sequence": (
            engineered,
            LogisticRegression(max_iter=200, random_state=17),
        ),
        "xgboost_full": (
            full,
            XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                tree_method="hist",
                device=device,
                n_jobs=4,
                random_state=17,
            ),
        ),
    }
    results = {
        name: _fit_auc(
            features[:train_end],
            labels[:train_end],
            features[test_start:],
            test_labels,
            model,
        )
        for name, (features, model) in models.items()
    }
    return {
        "oracle_auc": oracle_auc,
        "models": results,
        "headroom_from_best_observed": oracle_auc - max(results.values()),
        "positive_rate": float(test_labels.mean()),
        "cross_signal_correlation": float(
            np.corrcoef(dataset.diagnostic_features[test_start:, 0], test_labels)[0, 1]
        ),
        "sequence_match_lift": float(
            test_labels[dataset.diagnostic_features[test_start:, 1] == 1].mean()
            - test_labels[dataset.diagnostic_features[test_start:, 1] == 0].mean()
        ),
        "training_device": device,
        "prediction_device": "cpu",
    }
