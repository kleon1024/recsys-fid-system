"""Equal-request model-capacity benchmark for the V4 learned world."""


def run_model_capacity_benchmark(*args, **kwargs):
    from .runner import run_model_capacity_benchmark as run

    return run(*args, **kwargs)

__all__ = ["run_model_capacity_benchmark"]
