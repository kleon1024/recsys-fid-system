from fid_lab.simulation.digital_twin.experiments.ranking.posting_launch import (
    _decision,
)


def metric(low=-0.1, high=0.1):
    return {
        "control_mean": 1.0,
        "treatment_mean": 1.1,
        "absolute_delta": 0.1,
        "relative_delta": 0.1,
        "ci95_low": low,
        "ci95_high": high,
        "standard_error": 0.05,
        "mde80_absolute": 0.14,
        "mde80_relative": 0.14,
    }


def test_creator_launch_requires_publish_lift_without_failure_harm():
    metrics = {
        name: metric() for name in (
            "click", "create", "publish", "publish_failed",
            "early_qualified_long_view",
        )
    }
    sample = {
        "control_triggered_creators": 200,
        "treatment_triggered_creators": 200,
    }
    assert _decision(metrics, sample, 100)[0] == "hold"
    metrics["publish"] = metric(0.01, 0.2)
    assert _decision(metrics, sample, 100)[0] == "promote"
    metrics["publish_failed"] = metric(0.01, 0.2)
    assert _decision(metrics, sample, 100)[0] == "reject"
