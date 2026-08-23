"""Random-exposure calibration authority for KuaiRand behavior heads."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..contracts import FEEDBACK_NAMES
from ..data.randomized import calibration_masks


def binary_rule(labels, scores):
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    positives = int(labels.sum())
    if positives >= 50 and len(labels) - positives >= 50:
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(logit(clipped)[:, None], labels)
        return {
            "method": "platt", "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
    observed = np.clip(labels.mean(), 1e-6, 1.0 - 1e-6)
    logits = logit(clipped)
    lower, upper = -20.0, 20.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if expit(logits + midpoint).mean() < observed:
            lower = midpoint
        else:
            upper = midpoint
    return {
        "method": "prior_intercept", "coefficient": 1.0,
        "intercept": float((lower + upper) / 2.0),
    }


def apply_binary(scores, rule):
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return expit(rule["coefficient"] * logit(clipped) + rule["intercept"])


def randomized_calibration(split, predictions, seed):
    calibration, evaluation = calibration_masks(split, seed)
    labels = split.labels.numpy()
    report = {}
    for index, name in enumerate(FEEDBACK_NAMES):
        rule = binary_rule(
            labels[calibration, index], predictions[calibration, index]
        )
        scores = apply_binary(predictions[evaluation, index], rule)
        target = labels[evaluation, index]
        report[name] = {
            "rule": rule,
            "calibration_positives": int(labels[calibration, index].sum()),
            "evaluation_positives": int(target.sum()),
            "auc": float(roc_auc_score(target, scores)),
            "log_loss": float(log_loss(target, scores)),
            "brier": float(brier_score_loss(target, scores)),
            "predicted_rate": float(scores.mean()),
            "observed_rate": float(target.mean()),
        }
    stay_model = LinearRegression().fit(
        predictions[calibration, 7:8], labels[calibration, 7]
    )
    stay_scores = np.clip(
        stay_model.predict(predictions[evaluation, 7:8]), 0.0, 1.0
    )
    report["stay_norm"] = {
        "rule": {
            "method": "linear", "coefficient": float(stay_model.coef_[0]),
            "intercept": float(stay_model.intercept_),
        },
        "mae": float(np.abs(stay_scores - labels[evaluation, 7]).mean()),
        "predicted_mean": float(stay_scores.mean()),
        "observed_mean": float(labels[evaluation, 7].mean()),
    }
    return report
