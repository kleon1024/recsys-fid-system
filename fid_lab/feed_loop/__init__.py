"""Main short-video Feed model and streaming-launch loop.

Package import stays lightweight: the trajectory benchmark owns optional GPU
and Gymnasium dependencies, while the power check needs only NumPy.
"""

__all__ = ["run_feed_model_ladder", "run_small_effect_ab"]


def __getattr__(name):
    if name == "run_feed_model_ladder":
        from .models import run_feed_model_ladder

        return run_feed_model_ladder
    if name == "run_small_effect_ab":
        from .scale import run_small_effect_ab

        return run_small_effect_ab
    raise AttributeError(name)
