from __future__ import annotations

import unittest

import torch

from fid_lab.simulation.experimentation.contracts import (
    Experiment,
    ExperimentLayer,
    Variant,
)
from fid_lab.simulation.twin.contracts import TwinConfig, TwinPolicy
from fid_lab.simulation.twin.experimentation.orthogonal import (
    AssignmentUnit,
    TwinExperimentPlan,
    TwinLayerBinding,
    run_orthogonal_world,
)
from fid_lab.simulation.twin.kernel import DigitalTwinKernel


def config() -> TwinConfig:
    return TwinConfig(
        users=256,
        catalog_items=1_200,
        creators=120,
        topics=8,
        countries=4,
        preperiod_steps=2,
        measurement_steps=3,
        steps_per_day=2,
        history_length=8,
        route_candidates=4,
        routes=6,
        coarse_keep=12,
        fine_keep=5,
        audit_users=16,
        training_trace_users=32,
        batch_users=128,
        device="cpu",
    )


def layer(name: str, salt: str, parameter: str, value: float):
    return ExperimentLayer(
        name,
        salt,
        (Experiment(
            f"{name}-experiment",
            (Variant("treatment", 0.5, {parameter: value}),),
        ),),
    )


class TwinOrthogonalExperimentTest(unittest.TestCase):
    def plan(self):
        return TwinExperimentPlan((
            TwinLayerBinding(
                layer("fine", "fine-salt", "realtime_weight", 0.4),
                AssignmentUnit.USER,
            ),
            TwinLayerBinding(
                layer("mix", "mix-salt", "local_value_weight", 0.1),
                AssignmentUnit.REGION_TIME,
                block_steps=2,
            ),
        ))

    def test_user_layers_are_stable_and_region_layers_switch_by_time_block(self):
        kernel = DigitalTwinKernel(config())
        users = kernel.initialize().users[0]
        plan = self.plan()
        zero = plan.assign(users, 0)
        one = plan.assign(users, 1)
        two = plan.assign(users, 2)
        self.assertTrue(torch.equal(zero.assignments[0], two.assignments[0]))
        self.assertTrue(torch.equal(zero.assignments[1], one.assignments[1]))
        self.assertTrue((zero.assignments[1] != two.assignments[1]).any())

    def test_layers_cannot_own_the_same_parameter(self):
        duplicate = TwinExperimentPlan
        with self.assertRaisesRegex(ValueError, "owned by both"):
            duplicate((
                TwinLayerBinding(layer("a", "a", "realtime_weight", 0.3)),
                TwinLayerBinding(layer("b", "b", "realtime_weight", 0.4)),
            ))

    def test_orthogonal_cells_share_one_world_and_close_every_trace(self):
        kernel = DigitalTwinKernel(config())
        run = run_orthogonal_world(
            kernel,
            kernel.initialize(),
            TwinPolicy(name="orthogonal-control"),
            self.plan(),
            steps=3,
        )
        self.assertEqual(sum(run.request_counts.values()), config().users * 3)
        self.assertGreaterEqual(len(run.traces), 3)
        for trace in run.traces.values():
            self.assertTrue(all(trace.validate().values()))
        self.assertEqual(run.snapshot.step, 3)


if __name__ == "__main__":
    unittest.main()
