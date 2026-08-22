"""Million-user tensor simulation and experiment power."""

from .small_effect_ab import run_small_effect_ab
from .tensor_engine import run_tensor_feed

__all__ = ["run_small_effect_ab", "run_tensor_feed"]
