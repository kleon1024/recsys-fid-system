"""Stateful request/session/cross-session recommendation simulation."""

__all__ = ["run_closed_loop_experiment"]


def __getattr__(name):
    if name == "run_closed_loop_experiment":
        from .experiment import run_closed_loop_experiment

        return run_closed_loop_experiment
    raise AttributeError(name)
