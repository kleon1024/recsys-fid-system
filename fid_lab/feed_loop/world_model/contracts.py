"""Single authority for neural world-model tasks and acceptance semantics."""

from __future__ import annotations

from dataclasses import dataclass


WORLD_MODEL_VERSION = "neural-scm-v4.2-research"

WORLD_LABEL_NAMES = (
    "play", "play_3s", "stay_seconds", "play_completion_ratio",
    "complete_play", "long_view", "quality_long_view", "like",
    "negative_feedback", "anchor_click", "poi_detail", "poi_favorite",
    "conversion", "returned_next_session", "accepted_commercialization",
    "session_exit", "comment", "share", "follow",
    "content_click",
    "source_long_view",
)
WORLD_LABEL_COUNT = len(WORLD_LABEL_NAMES)


@dataclass(frozen=True)
class BinaryAction:
    name: str
    label_index: int
    requires: str | None = None


BINARY_ACTIONS = (
    BinaryAction("play", 0),
    BinaryAction("play_3s", 1, "play"),
    BinaryAction("complete_play", 4, "play"),
    BinaryAction("long_view", 5, "play"),
    BinaryAction("quality_long_view", 6, "play"),
    BinaryAction("like", 7, "play"),
    BinaryAction("negative_feedback", 8),
    BinaryAction("anchor_click", 9),
    BinaryAction("poi_detail", 10, "anchor_click"),
    BinaryAction("poi_favorite", 11, "poi_detail"),
    BinaryAction("conversion", 12, "poi_detail"),
    BinaryAction("session_exit", 15),
    BinaryAction("returned_next_session", 13, "session_exit"),
    BinaryAction("comment", 16, "play"),
    BinaryAction("share", 17, "play"),
    BinaryAction("follow", 18, "play"),
    BinaryAction("content_click", 19, "play"),
    BinaryAction("source_long_view", 20, "play"),
)

STOCHASTIC_ACTIONS = tuple(
    action for action in BINARY_ACTIONS
    if action.name not in {
        "play_3s", "complete_play", "long_view", "quality_long_view"
    }
)

STAY_LABEL_INDEX = 2
COMPLETION_LABEL_INDEX = 3
SEQUENCE_EVENT_INDICES = (5, 6, 7, 8, 9, 12)
STRUCTURAL_INTERVENTION_NAMES = (
    "recent_interest_signal",
    "content_quality_prior",
    "negative_history_signal",
)


@dataclass(frozen=True)
class WorldModelConfig:
    feature_dim: int = 28
    sequence_dim: int = 8
    sequence_length: int = 24
    width: int = 96
    latent_dim: int = 24
    attention_heads: int = 4
    stay_mixture_components: int = 3
    ensemble_members: int = 3
    batch_size: int = 2_048
    epochs: int = 8
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    stay_loss_weight: float = 2.0
    utility_loss_weight: float = 1.0
    max_ips_weight: float = 20.0
    randomized_adaptation_epochs: int = 2
    randomized_adaptation_learning_rate: float = 1e-4
    randomized_pairwise_epochs: int = 2
    structural_adaptation_epochs: int = 16
    structural_adaptation_learning_rate: float = 2e-4
    structural_mean_loss_weight: float = 10.0
    seed: int = 20260823

    def __post_init__(self) -> None:
        if self.width % self.attention_heads:
            raise ValueError("world-model width must be divisible by attention heads")
        if self.ensemble_members < 2:
            raise ValueError("world-model uncertainty requires at least two members")
        if self.stay_mixture_components < 2:
            raise ValueError("censored stay model requires at least two mixture components")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("world-model training sizes must be positive")
        if self.stay_loss_weight <= 0.0:
            raise ValueError("stay likelihood weight must be positive")
        if self.utility_loss_weight <= 0.0:
            raise ValueError("utility loss weight must be positive")
        if self.randomized_adaptation_epochs < 0:
            raise ValueError("randomized adaptation epochs cannot be negative")
        if self.randomized_adaptation_learning_rate <= 0.0:
            raise ValueError("randomized adaptation learning rate must be positive")
        if self.randomized_pairwise_epochs < 0:
            raise ValueError("randomized pairwise epochs cannot be negative")
        if self.structural_adaptation_epochs < 0:
            raise ValueError("structural adaptation epochs cannot be negative")
        if self.structural_adaptation_learning_rate <= 0.0:
            raise ValueError("structural adaptation learning rate must be positive")
        if self.structural_mean_loss_weight <= 0.0:
            raise ValueError("structural mean loss weight must be positive")


ACCEPTANCE_THRESHOLDS = {
    "mean_binary_ece": 0.035,
    "joint_correlation_mae": 0.080,
    "stay_median_relative_error": 0.100,
    "stay_p90_relative_error": 0.150,
    "sequence_lag1_mae": 0.120,
    "maximum_ensemble_probability_std": 0.080,
    "intervention_sign_accuracy": 0.80,
    "intervention_normalized_mae": 0.50,
    "policy_kendall_tau": 2.0 / 3.0,
}
