"""One launch contract across model, feature, strategy, and system changes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..feed_loop.scale.tensor_engine import TensorPolicy


class LaunchCategory(str, Enum):
    MODEL = "model"
    FEATURE = "feature"
    STRATEGY = "strategy"
    ARCHITECTURE = "architecture"
    REALTIME = "realtime"
    BUG_FIX = "bug_fix"
    CHAIN_DIAGNOSIS = "chain_diagnosis"
    PRODUCT = "product"
    BUSINESS_VALUE = "business_value"
    LONG_TERM_VALUE = "long_term_value"


@dataclass(frozen=True)
class PolicyLaunchSpec:
    launch_id: str
    category: LaunchCategory
    title: str
    hypothesis: str
    change: str
    owner: str
    control: TensorPolicy
    treatment: TensorPolicy
    primary_metric: str
    training_mode: str
    product_dependency: str = "none"
    short_term_value: str = "stay and long-view behavior"
    long_term_value: str = "LT container, quality view, negative feedback, and return"
