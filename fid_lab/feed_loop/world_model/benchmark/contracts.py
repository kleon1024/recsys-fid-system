"""Acceptance semantics for proving that V4 exposes usable model capacity."""

from __future__ import annotations


MIN_CONTEXT_DELTA_TO_BASELINE_STD = 0.05
MIN_AUC_CAPACITY_GAIN = 0.0005
MIN_REQUEST_REGRET_REDUCTION = 0.02

TABULAR_DEEP_MODELS = ("wide_deep", "deepfm", "dcnv2")
REQUEST_MODELS = ("din_request", "slate_transformer")


def capacity_gates(context: dict, models: dict) -> dict[str, bool]:
    logistic_auc = models["logistic_regression"]["auc"]
    deep_auc = max(models[name]["auc"] for name in TABULAR_DEEP_MODELS)
    tabular_auc = max(
        models[name]["auc"] for name in ("logistic_regression", "xgboost")
    )
    request_auc = max(models[name]["auc"] for name in REQUEST_MODELS)
    tabular_regret = min(
        models[name]["request"]["oracle_regret"]
        for name in ("logistic_regression", "xgboost", *TABULAR_DEEP_MODELS)
    )
    request_regret = min(
        models[name]["request"]["oracle_regret"] for name in REQUEST_MODELS
    )
    return {
        "sequence_context_material": context["permuted_sequence"][
            "relative_to_baseline_std"
        ] >= MIN_CONTEXT_DELTA_TO_BASELINE_STD,
        "slate_context_material": context["selected_only_slate"][
            "relative_to_baseline_std"
        ] >= MIN_CONTEXT_DELTA_TO_BASELINE_STD,
        "deep_interaction_gain_over_logistic": (
            deep_auc >= logistic_auc + MIN_AUC_CAPACITY_GAIN
        ),
        "request_model_auc_gain": (
            request_auc >= tabular_auc + MIN_AUC_CAPACITY_GAIN
        ),
        "request_model_regret_gain": (
            request_regret
            <= tabular_regret * (1.0 - MIN_REQUEST_REGRET_REDUCTION)
        ),
    }
