"""Render deterministic README charts from checked report authorities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "launches"
BENCHMARKS = ROOT / "reports" / "benchmarks"
OUTPUT = ROOT / "docs" / "assets"
COLORS = ("#2563eb", "#f59e0b", "#059669", "#dc2626", "#7c3aed")


def _load(name: str) -> dict[str, object]:
    return json.loads((REPORTS / name).read_text())


def _load_benchmark(name: str) -> dict[str, object]:
    return json.loads((BENCHMARKS / name).read_text())


def _setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "recsys-fid-system-v1",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / name,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "recsys-fid-system report renderer"},
    )
    plt.close(fig)


def model_quality() -> None:
    report = _load("2026-08-23-feed-model-ladder.json")
    order = ("lr_full_feed", "wide_deep", "deepfm", "dcnv2", "mmoe_value_tree")
    labels = ("LR", "W&D", "DeepFM", "DCNv2", "MMoE")
    auc = [report["offline"][name]["auc"] for name in order]
    regret = [
        report["offline"][name]["candidate"]["oracle_regret"] for name in order
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    x = np.arange(len(labels))
    axes[0].bar(x, auc, color=COLORS[0], width=0.66)
    axes[0].set_ylim(0.55, 0.74)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("Offline discrimination")
    axes[1].bar(x, regret, color=COLORS[3], width=0.66)
    axes[1].set_ylim(0.0, 0.17)
    axes[1].set_ylabel("Oracle regret (lower is better)")
    axes[1].set_title("Candidate-choice quality")
    for axis, values in zip(axes, (auc, regret)):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(index, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)
    figure.suptitle("Why Logistic Regression remains the serving authority", y=1.02)
    figure.tight_layout()
    _save(figure, "model-quality.svg")


def training_loss() -> None:
    report = _load("2026-08-23-feed-model-ladder.json")
    online = _load("2026-08-23-feed-online-learning.json")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for index, (name, label) in enumerate(
        (("wide_deep", "W&D"), ("deepfm", "DeepFM"), ("dcnv2", "DCNv2"))
    ):
        values = report["offline"][name]["loss_history"]
        axes[0].plot(range(1, len(values) + 1), values, marker="o", label=label,
                     color=COLORS[index])
    axes[0].set_title("Single-task ranker training")
    axes[0].set_xlabel("Recorded epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].legend(frameon=False)
    mmoe = report["offline"]["mmoe_value_tree"]["loss_history"]
    axes[1].plot(range(1, len(mmoe) + 1), mmoe, marker="o", color=COLORS[4],
                 label="MMoE")
    ps_values = (
        online["training"]["mean_loss_first"],
        online["training"]["mean_loss_last"],
    )
    axes[1].plot((1, len(mmoe)), ps_values, marker="s", linestyle="--",
                 color=COLORS[1], label="Streaming PS")
    axes[1].set_title("Multi-task and streaming training")
    axes[1].set_xlabel("Recorded epoch / endpoint")
    axes[1].set_ylabel("Training loss")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle("Loss falls even when online policy quality does not improve", y=1.02)
    figure.tight_layout()
    _save(figure, "training-loss.svg")


def model_ab_impact() -> None:
    report = _load("2026-08-23-feed-model-ladder.json")
    launches = (
        ("lr_to_wide_deep", "W&D"),
        ("lr_to_deepfm", "DeepFM"),
        ("lr_to_dcnv2", "DCNv2"),
        ("lr_to_mmoe_value_tree", "MMoE"),
    )
    metrics = (
        ("stay_per_exposure", "Stay / exposure"),
        ("quality_long_view_rate", "Quality long-view"),
        ("local_value_tree_score", "Local Value Tree"),
        ("lt_value", "Platform LT"),
    )
    values = np.asarray(
        [
            [report["launches"][launch]["metrics"][metric]["relative_lift"] * 100
             for metric, _ in metrics]
            for launch, _ in launches
        ]
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(metrics))
    width = 0.18
    for index, (_, label) in enumerate(launches):
        axis.bar(x + (index - 1.5) * width, values[index], width, label=label,
                 color=COLORS[index])
    axis.axhline(0, color="#475569", linewidth=1)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Observed A/B relative lift (%)")
    axis.set_title("Every advanced fine-rank candidate violates a primary or guardrail metric")
    axis.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save(figure, "model-ab-impact.svg")


def cascade_and_local() -> None:
    coarse = _load("2026-08-23-coarse-cascade-ladder.json")
    local = _load("2026-08-23-local-intent-ranker-scale-10m.json")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    first = coarse["replicates"][0]["launches"][0]
    recall = first["metrics"]["coarse_feed_oracle_recall"]
    regret = first["metrics"]["fine_oracle_regret_per_exposure"]
    axes[0].bar((0, 1), (recall["control_mean"], recall["treatment_mean"]),
                color=(COLORS[3], COLORS[2]), width=0.62)
    axes[0].set_xticks((0, 1), ("Quality-only", "LR coarse"))
    axes[0].set_ylabel("Oracle Top-K preservation")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_title(
        f"Coarse repair: regret {regret['control_mean']:.3f} → {regret['treatment_mean']:.3f}"
    )
    names = ("Intent ranker v4", "Load expansion v5")
    metric_names = (
        ("stay_per_exposure", "Stay"),
        ("local_value_tree_score_per_exposure", "Local tree"),
        ("lt_value_per_user", "Platform LT"),
    )
    local_values = np.asarray(
        [
            [launch["metrics"][metric]["pooled_relative_lift"] * 100
             for metric, _ in metric_names]
            for launch in local["aggregate"]
        ]
    )
    x = np.arange(len(metric_names))
    axes[1].bar(x - 0.18, local_values[0], 0.36, label=names[0], color=COLORS[0])
    axes[1].bar(x + 0.18, local_values[1], 0.36, label=names[1], color=COLORS[1])
    axes[1].axhline(0, color="#475569", linewidth=1)
    axes[1].set_xticks(x, [label for _, label in metric_names])
    axes[1].set_ylabel("Pooled relative lift (%)")
    axes[1].set_title("Local business gain does not automatically become LT")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.tight_layout()
    _save(figure, "cascade-local-tradeoff.svg")


def model_scale() -> None:
    small = _load_benchmark("2026-08-23-model-v2-1m-gpu.json")
    large = _load_benchmark("2026-08-23-model-v2-10m-gpu.json")
    labels = ("LR", "XGBoost", "W&D", "DeepFM", "DCNv2", "DIN", "MMoE", "PLE")
    small_auc = [result["metrics"]["auc"] for result in small["ranking"]]
    large_auc = [result["metrics"]["auc"] for result in large["ranking"]]
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(labels))
    axis.bar(x - 0.19, small_auc, 0.38, label="1M impressions / 20K samples",
             color=COLORS[0])
    axis.bar(x + 0.19, large_auc, 0.38, label="10M impressions / 200K samples",
             color=COLORS[2])
    axis.set_ylim(0.56, 0.625)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Test AUC")
    axis.set_title("Nonlinear DGP: model headroom appears only after sample scale increases")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, (left, right) in enumerate(zip(small_auc, large_auc)):
        axis.text(index, max(left, right) + 0.0013, f"{right-left:+.3f}",
                  ha="center", fontsize=8)
    figure.tight_layout()
    _save(figure, "model-scale.svg")


def tensor_migration() -> None:
    report = _load("2026-08-23-tensor-artifact-v2-1m-gpu.json")
    parity = report["semantic_parity"]
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    distribution = parity["distribution"]
    labels = ("Stay", "Long view", "Quality view")
    gaps = [value["relative_gap"] * 100 for value in distribution.values()]
    axes[0].bar(labels, gaps, color=(COLORS[0], COLORS[2], COLORS[1]))
    axes[0].axhline(10, color=COLORS[3], linestyle="--", linewidth=1)
    axes[0].axhline(-10, color=COLORS[3], linestyle="--", linewidth=1)
    axes[0].set_ylabel("Tensor vs semantic gap (%)")
    axes[0].set_title("Control distribution parity")
    effects = parity["treatment_effect"]
    names = ("Stay", "Quality view")
    semantic = [value["semantic_true_relative_itt"] * 100 for value in effects.values()]
    tensor = [value["tensor_relative_lift"] * 100 for value in effects.values()]
    x = np.arange(len(names))
    axes[1].bar(x - 0.18, semantic, 0.36, label="Semantic", color=COLORS[0])
    axes[1].bar(x + 0.18, tensor, 0.36, label="Tensor 1M", color=COLORS[2])
    axes[1].set_xticks(x, names)
    axes[1].axhline(0, color="#475569", linewidth=1)
    axes[1].set_ylabel("Relative treatment effect (%)")
    axes[1].set_title("Effect-direction parity")
    axes[1].legend(frameon=False)
    lt_metrics = report["unified_lt_exchange"]
    component_names = (
        "lt_stay_per_user",
        "lt_active_days_per_user",
        "accepted_platform_commercialization_per_user",
    )
    component_labels = ("Stay", "Active", "Commerce")
    component_effects = [
        lt_metrics["components"][name]["treatment_mean"]
        - lt_metrics["components"][name]["control_mean"]
        for name in component_names
    ]
    total = lt_metrics["total"]
    lt_values = component_effects + [total["treatment_mean"] - total["control_mean"]]
    axes[2].bar((*component_labels, "Total"), lt_values,
                color=(COLORS[0], COLORS[2], COLORS[1], COLORS[4]))
    axes[2].axhline(0, color="#475569", linewidth=1)
    axes[2].set_ylabel("Exchanged LT lift / user")
    axes[2].set_title("Unified LT gate: CI lower bound >= 0")
    performance = (report["control"]["performance"], report["treatment"]["performance"])
    throughput = [value["requests_per_second"] / 1_000_000 for value in performance]
    axes[3].bar((0, 1), throughput, color=(COLORS[0], COLORS[4]))
    axes[3].set_xticks((0, 1), ("LR control", "Guarded XGB"))
    axes[3].set_ylabel("Million requests / second")
    axes[3].set_title("RTX 4090 tensor throughput")
    for axis in axes:
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle("Published artifact: semantic-to-tensor migration", y=1.02)
    figure.tight_layout()
    _save(figure, "tensor-migration.svg")


def feature_lr_launches() -> None:
    training = _load("2026-08-23-feature-lr-hash-split-training-gpu.json")
    launches = _load("2026-08-23-feature-lr-hash-split-1m-gpu.json")
    stateful = _load("2026-08-23-feature-lr-stateful-500.json")
    groups = (
        "basic__realtime__local_context",
        "basic__realtime__local_context__duration",
        "basic__realtime__local_context__identity_hash",
        "basic__realtime__local_context__category_hash",
    )
    labels = ("Active", "+Duration", "+Identity", "+Category")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    auc = [training["offline"][name]["auc"] for name in groups]
    axes[0].bar(labels, auc, color=COLORS[:5])
    axes[0].set_ylim(0.738, 0.747)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("Rejected bundle split offline")
    launch_labels = ("Duration", "Identity", "Category")
    lt_lifts = [
        launch["ab"]["lt_value_per_user"]["relative_lift"] * 100
        for launch in launches["launches"]
    ]
    decisions = [launch["decision"] for launch in launches["launches"]]
    colors = [
        COLORS[2] if value.startswith("pass")
        else COLORS[3] if value.startswith("reject")
        else COLORS[1]
        for value in decisions
    ]
    bars = axes[1].bar(launch_labels, lt_lifts, color=colors)
    axes[1].bar_label(bars, fmt="%.3f%%", padding=3)
    axes[1].axhline(0, color="#475569", linewidth=1)
    axes[1].set_ylabel("Unified LT relative lift (%)")
    axes[1].set_title("Last-accepted-control Launch Reviews")
    attribution = stateful["joiner"]["request_candidate_dataset"][
        "stage_attribution"
    ]
    stage_names = ("recall_miss", "coarse_miss", "fine_rank_miss", "served_oracle")
    stage_labels = ("Recall", "Coarse", "Fine", "Served")
    total = attribution["requests"]
    stage_rates = [attribution[name] / total * 100 for name in stage_names]
    axes[2].bar(stage_labels, stage_rates, color=COLORS[:4])
    axes[2].set_ylabel("Requests (%)")
    axes[2].set_title("Request-level failure attribution")
    for axis in axes:
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=15)
    figure.suptitle("Small LR launches: isolate one feature contract at a time", y=1.02)
    figure.tight_layout()
    _save(figure, "feature-lr-launches.svg")


def main() -> None:
    _setup()
    model_quality()
    training_loss()
    model_ab_impact()
    cascade_and_local()
    model_scale()
    tensor_migration()
    feature_lr_launches()


if __name__ == "__main__":
    main()
