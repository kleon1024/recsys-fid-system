"""Unified training, shadow, A/B, gate, and review contracts."""


def __getattr__(name: str):
    if name in __all__:
        from .policy import policy_launches, run_policy_launch, run_policy_launch_suite

        return {
            "policy_launches": policy_launches,
            "run_policy_launch": run_policy_launch,
            "run_policy_launch_suite": run_policy_launch_suite,
        }[name]
    raise AttributeError(name)

__all__ = ["policy_launches", "run_policy_launch", "run_policy_launch_suite"]
