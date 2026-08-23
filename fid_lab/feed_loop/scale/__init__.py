"""Million-user tensor simulation and experiment power."""

__all__ = ["run_small_effect_ab", "run_tensor_feed"]


def __getattr__(name):
    if name == "run_small_effect_ab":
        from .small_effect_ab import run_small_effect_ab

        return run_small_effect_ab
    if name == "run_tensor_feed":
        from .tensor_engine import run_tensor_feed

        return run_tensor_feed
    raise AttributeError(name)
