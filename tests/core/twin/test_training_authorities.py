from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from fid_lab.simulation.twin.contracts import TwinConfig, TwinPolicy
from fid_lab.simulation.twin.kernel import DigitalTwinKernel
from fid_lab.simulation.twin.experimentation.robustness import (
    run_heldout_environment_gate,
)
from fid_lab.simulation.twin.serving.models import ServingStack
from fid_lab.simulation.twin.serving.trace import RequestTrace
from fid_lab.simulation.twin.platform.fids import (
    TWIN_FID_FIELDS,
    TwinFidEncoder,
)
from fid_lab.simulation.twin.training import (
    ContinuousLearningConfig,
    ModelRegistry,
    ModelStatus,
    join_training_authorities,
    materialize_events,
    train_fine_ranker,
    run_continuous_learning,
)
from fid_lab.simulation.twin.training.ranker import (
    RankerArtifact,
    scoring_context_from_examples,
)


def training_config() -> TwinConfig:
    return TwinConfig(
        users=384,
        catalog_items=1_800,
        creators=180,
        topics=8,
        countries=4,
        preperiod_steps=2,
        measurement_steps=4,
        steps_per_day=2,
        history_length=8,
        route_candidates=4,
        routes=6,
        coarse_keep=12,
        fine_keep=5,
        audit_users=128,
        batch_users=192,
        device="cpu",
    )


class TwinTrainingAuthorityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = training_config()
        cls.policy = TwinPolicy(name="training-logging-policy-v1")
        kernel = DigitalTwinKernel(cls.config)
        cls.twin_run = kernel.run(kernel.initialize(), cls.policy, 6)

    def events(self, watermark_step: int):
        return materialize_events(
            self.twin_run.trace,
            world_version=self.config.version,
            served_policy=self.policy.name,
            experiment_cell="logging",
            watermark_step=watermark_step,
        )

    def test_behavior_selects_from_slate_instead_of_copying_rank_top_one(self):
        values = self.twin_run.trace.tensors()
        multi_item = values["exposed_item_ids"][:, 1] >= 0
        differs = values["selected_item"][multi_item] != (
            values["exposed_item_ids"][multi_item, 0]
        )
        self.assertTrue(differs.any())

    def test_point_in_time_gate_allows_legal_repeat_but_rejects_future_state(self):
        first = {
            name: torch.tensor(value.tolist())
            for name, value in self.twin_run.trace.rows[0].items()
        }
        trace = RequestTrace(rows=[first])
        first["history_item"][0, 0] = first["selected_item"][0]
        first["history_step"][0, 0] = first["step"][0] - 1
        self.assertTrue(trace.validate()["point_in_time_history"])
        first["history_step"][0, 0] = first["step"][0] + 1
        self.assertFalse(trace.validate()["point_in_time_history"])
        first["history_step"][0, 0] = first["step"][0] - 1

    def test_materializer_builds_three_distinct_request_level_authorities(self):
        events = self.events(watermark_step=100)
        authorities = join_training_authorities(events)
        self.assertGreater(events.requests, 0)
        self.assertGreater(len(authorities.recall.request_id), 0)
        self.assertEqual(len(authorities.coarse.request_id), events.requests)
        self.assertEqual(len(authorities.fine.request_id), events.requests)
        self.assertEqual(
            authorities.coarse.item_ids.shape[1],
            self.config.routes * self.config.route_candidates,
        )
        self.assertEqual(
            authorities.coarse.features.shape[:2],
            authorities.coarse.item_ids.shape,
        )
        self.assertTrue(
            authorities.coarse.exposed[authorities.coarse.relevance > 0].all()
        )
        self.assertTrue(
            authorities.fine.selected.sum(dim=1).eq(1).all()
        )
        self.assertEqual(
            authorities.fine.sparse_fids.shape[-1], len(TWIN_FID_FIELDS)
        )
        valid = authorities.fine.item_ids >= 0
        self.assertTrue(
            (authorities.fine.sparse_buckets[valid] > 0).all()
        )
        self.assertTrue(
            (authorities.fine.sparse_buckets[~valid] == 0).all()
        )
        expected_fids, expected_buckets = TwinFidEncoder().encode_candidates(
            user_id=authorities.fine.user_id,
            item_id=authorities.fine.item_ids,
            item_kind=authorities.fine.item_kinds,
            surface=authorities.fine.surface,
            route=authorities.fine.route,
            step=authorities.fine.step,
        )
        self.assertTrue(torch.equal(
            authorities.fine.sparse_fids, expected_fids
        ))
        self.assertTrue(torch.equal(
            authorities.fine.sparse_buckets, expected_buckets
        ))
        self.assertTrue((
            (authorities.fine.history_steps < authorities.fine.step[:, None])
            | (authorities.fine.history_steps < 0)
        ).all())
        self.assertTrue(
            authorities.recall.sampling_probability.eq(1.0).all()
        )

    def test_trace_reservoir_is_bounded_deterministic_and_probability_aware(self):
        limit = 17
        first = self.twin_run.trace.sampled(limit, salt=91)
        second = self.twin_run.trace.sampled(limit, salt=91)
        self.assertEqual(first.manifest()["requests"], limit)
        self.assertTrue(torch.equal(
            first.tensors()["request_id"], second.tensors()["request_id"]
        ))
        original_probability = self.twin_run.trace.tensors()[
            "request_sampling_probability"
        ].max()
        sampled_probability = first.tensors()[
            "request_sampling_probability"
        ].max()
        self.assertLess(float(sampled_probability), float(original_probability))

    def test_exploration_logs_nonzero_known_exposure_propensity(self):
        policy = TwinPolicy(name="exploration-log", exploration_rate=0.25)
        kernel = DigitalTwinKernel(self.config)
        explored = kernel.run(kernel.initialize(), policy, 2)
        events = materialize_events(
            explored.trace,
            world_version=self.config.version,
            served_policy=policy.name,
            experiment_cell="exploration",
            watermark_step=100,
        )
        valid = events.exposed_item_ids >= 0
        propensity = events.exposed_propensity[valid]
        self.assertTrue((propensity > 0).all())
        self.assertTrue((propensity < 1).any())
        fine = join_training_authorities(events).fine
        self.assertTrue((fine.examination_propensity[fine.item_ids >= 0] > 0).all())

    def test_label_maturity_masks_future_outcomes_without_writing_negatives(self):
        early = join_training_authorities(self.events(watermark_step=5)).fine
        mature = join_training_authorities(self.events(watermark_step=100)).fine
        self.assertLess(int(early.label_mask.sum()), int(mature.label_mask.sum()))
        payment_index = 13
        self.assertFalse(early.label_mask[:, :, payment_index].any())
        self.assertTrue(mature.label_mask[:, :, payment_index].any())
        self.assertTrue(torch.equal(early.labels, mature.labels))

    def test_trained_ranker_round_trips_and_registry_preserves_active_parent(self):
        authorities = join_training_authorities(self.events(watermark_step=100))
        artifact = train_fine_ranker(
            authorities.fine,
            model_id="fine-lr-v1",
            epochs=4,
            microbatch_rows=128,
            learning_rate=1e-2,
        )
        self.assertLess(
            artifact.training_report["loss_last"],
            artifact.training_report["loss_first"],
        )
        restored = RankerArtifact.from_checkpoint(
            artifact.checkpoint(), device="cpu"
        )
        sample_features = authorities.fine.features[:3]
        sample_surface = authorities.fine.surface[:3]
        sample_context = scoring_context_from_examples(
            authorities.fine, slice(0, 3)
        )
        self.assertTrue(torch.equal(
            artifact.score(sample_features, sample_surface, sample_context),
            restored.score(sample_features, sample_surface, sample_context),
        ))
        registry = ModelRegistry()
        first = registry.register("fine", artifact)
        registry.shadow(first.version)
        registry.promote(first.version)
        second_artifact = RankerArtifact.from_checkpoint(
            artifact.checkpoint(), device="cpu"
        )
        second_artifact.model_id = "fine-lr-v2"
        second = registry.register("fine", second_artifact)
        registry.reject(second.version)
        self.assertEqual(registry.active("fine").version, first.version)
        self.assertEqual(first.status, ModelStatus.ACTIVE)

    def test_mature_ranker_ladder_trains_and_replays_same_contract(self):
        authorities = join_training_authorities(self.events(watermark_step=100))
        sample_features = authorities.fine.features[:3]
        sample_surface = authorities.fine.surface[:3]
        sample_context = scoring_context_from_examples(
            authorities.fine, slice(0, 3)
        )
        for architecture in ("wide_deep", "deepfm", "dcnv2", "mmoe"):
            artifact = train_fine_ranker(
                authorities.fine,
                model_id=f"fine-{architecture}-contract",
                architecture=architecture,
                epochs=1,
                microbatch_rows=128,
            )
            restored = RankerArtifact.from_checkpoint(
                artifact.checkpoint(), device="cpu"
            )
            self.assertTrue(torch.equal(
                artifact.score(
                    sample_features, sample_surface, sample_context
                ),
                restored.score(
                    sample_features, sample_surface, sample_context
                ),
            ), architecture)
            self.assertGreater(
                artifact.training_report["parameters"], 0
            )
            split = artifact.training_report["chronological_split"]
            if split["mode"] == "whole_step":
                self.assertLess(
                    split["train_max_step"], split["test_min_step"]
                )
            else:
                self.assertLessEqual(
                    split["train_max_step"], split["test_min_step"]
                )
            offline = artifact.training_report["offline"]
            self.assertGreater(offline["requests"], 0)
            evaluated = [
                value for value in offline["tasks"].values()
                if value.get("auc") is not None
            ]
            self.assertGreater(len(evaluated), 0)
            self.assertIn("user_gauc", evaluated[0])

    def test_sparse_ranker_fails_closed_without_logged_fid_context(self):
        authorities = join_training_authorities(self.events(watermark_step=100))
        artifact = train_fine_ranker(
            authorities.fine,
            model_id="fine-deepfm-context-gate",
            architecture="deepfm",
            epochs=1,
            microbatch_rows=128,
        )
        with self.assertRaisesRegex(ValueError, "requires scoring context"):
            artifact.score(
                authorities.fine.features[:2], authorities.fine.surface[:2]
            )

    def test_learned_model_uses_same_serving_stack_boundary_as_rules(self):
        authorities = join_training_authorities(self.events(watermark_step=100))
        artifact = train_fine_ranker(
            authorities.fine, model_id="fine-lr-serving", epochs=1
        )
        stack = ServingStack(strategy=self.policy, fine_model=artifact)
        kernel = DigitalTwinKernel(self.config)
        learned_run = kernel.run(kernel.initialize(), stack, 1)
        self.assertGreater(learned_run.trace.manifest()["requests"], 0)
        self.assertIn("fine=fine-lr-serving", stack.name)

    def test_continuous_loop_trains_on_prior_mixed_ab_traffic(self):
        config = training_config()
        config = replace(
            config, audit_users=16, training_trace_users=64
        )
        report = run_continuous_learning(
            config,
            ContinuousLearningConfig(
                iterations=2,
                logging_steps=2,
                sample_lookback_steps=32,
                architectures=("lr", "mlp"),
                train_epochs=1,
                microbatch_rows=128,
            ),
        )
        first, second = report["iterations"]
        self.assertLess(first["world_step_before"], first["world_step_after"])
        self.assertEqual(second["world_step_before"], first["world_step_after"])
        self.assertGreaterEqual(second["samples"]["logging_policy_versions"], 2)
        self.assertGreaterEqual(second["samples"]["experiment_cells"], 3)
        self.assertEqual(len(report["registry"]["models"]), 2)
        status_by_version = {
            row["version"]: row["status"]
            for row in report["registry"]["models"]
        }
        for iteration in report["iterations"]:
            if iteration["decision"] in {"hold", "reject"}:
                expected = (
                    "shadow" if iteration["decision"] == "hold"
                    else "rejected"
                )
                self.assertEqual(
                    status_by_version[iteration["registry_version"]], expected
                )

    def test_fixed_artifact_is_replayed_in_heldout_hidden_world(self):
        report = run_heldout_environment_gate(
            self.config,
            architecture="lr",
            heldout_environment_seeds=(self.config.environment_seed + 1009,),
            source_steps=6,
            maximum_training_requests=128,
            model_weight=0.10,
        )
        self.assertEqual(len(report["artifact_fingerprint"]), 64)
        self.assertEqual(
            report["evaluations"][0]["environment_seed"],
            self.config.environment_seed + 1009,
        )
        self.assertNotEqual(
            report["source_environment_seed"],
            report["evaluations"][0]["environment_seed"],
        )
        self.assertIn(report["aggregate_decision"], {"pass", "hold", "reject"})


if __name__ == "__main__":
    unittest.main()
