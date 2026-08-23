from __future__ import annotations

import unittest

import numpy as np

from fid_lab.feed_loop.streaming.online_learning import ParameterServerFeedPolicy
from fid_lab.training.consistency import ChainConsistencyAuditor
from fid_lab.training.contracts import (
    ActionEvent,
    ChainManifest,
    ImpressionEvent,
    PredictionRecord,
    TrainingExample,
)
from fid_lab.training.evaluation import auc, compare
from fid_lab.training.joiner import ExampleJoiner
from fid_lab.training.parameter_server import VersionedParameterServer
from fid_lab.training.trainer import OnlineMultiTaskTrainer


TEST_TASKS = ("click", "like", "long_view")


def impression(event_time: int = 100) -> ImpressionEvent:
    return ImpressionEvent(
        request_id="request-1",
        user_id=1,
        item_id=7,
        event_time=event_time,
        position=2,
        propensity=0.25,
        feature_fids=tuple(range(9)),
        feature_buckets=tuple(range(9)),
        schema_version="schema-v1",
        served_model_version=0,
    )


class JoinerTest(unittest.TestCase):
    def test_waits_for_window_and_deduplicates_delayed_actions(self) -> None:
        shown = impression()
        valid = ActionEvent("a-1", "request-1", 7, "click", 120, 130)
        outside = ActionEvent("a-2", "request-1", 7, "like", 500, 505)
        joiner = ExampleJoiner()
        immature = joiner.build([shown], [valid], watermark=300)
        self.assertEqual(len(immature.examples), 0)
        self.assertEqual(immature.immature_impressions, 1)
        mature = joiner.build([shown], [valid, valid, outside], watermark=600)
        self.assertEqual(len(mature.examples), 1)
        self.assertEqual(mature.duplicate_actions, 1)
        self.assertEqual(mature.ignored_actions, 1)
        self.assertEqual(mature.examples[0].labels["click"], 1.0)
        self.assertEqual(mature.examples[0].labels["like"], 0.0)
        self.assertEqual(mature.examples[0].sample_weight, 4.0)


class ParameterServerTest(unittest.TestCase):
    def test_vectorized_online_features_match_scalar_hash_contract(self) -> None:
        server = VersionedParameterServer(feature_dim=128, tasks=TEST_TASKS)
        trainer = OnlineMultiTaskTrainer(server)
        policy = ParameterServerFeedPolicy(trainer)
        features = np.asarray([[0.0, 0.5, -0.5], [1.0, -1.0, 0.2]])
        vectors = policy._vectors(features)
        expected = np.zeros_like(vectors)
        buckets = np.clip(np.rint((features + 1.0) * 7.5), 0, 15).astype(int)
        for row in range(len(features)):
            for field, bucket in enumerate(buckets[row]):
                expected[row, (field * 131 + bucket) % expected.shape[1]] += 1.0
        np.testing.assert_array_equal(vectors, expected)

    def test_updates_are_idempotent_and_stale_gradients_fail_closed(self) -> None:
        server = VersionedParameterServer(feature_dim=8, max_staleness=0)
        weight_gradient = np.ones((len(TEST_TASKS), 8))
        bias_gradient = np.ones(len(TEST_TASKS))
        applied = server.apply("u-1", 0, weight_gradient, bias_gradient, 0.1)
        duplicate = server.apply("u-1", 0, weight_gradient, bias_gradient, 0.1)
        stale = server.apply("u-2", 0, weight_gradient, bias_gradient, 0.1)
        self.assertTrue(applied.applied)
        self.assertEqual(duplicate.reason, "duplicate_update")
        self.assertEqual(stale.reason, "stale_gradient")
        self.assertEqual(server.snapshot().version, 1)

    def test_online_trainer_improves_separable_click_auc(self) -> None:
        examples = [
            TrainingExample(
                example_id=str(index),
                user_id=index % 10,
                item_id=index,
                impression_time=index,
                feature_fids=(index % 2,),
                feature_buckets=(index % 2,),
                labels={"click": float(index % 2), "like": 0.0, "long_view": 0.0},
                sample_weight=1.0,
                schema_version="schema-v1",
            )
            for index in range(200)
        ]
        server = VersionedParameterServer(feature_dim=16)
        trainer = OnlineMultiTaskTrainer(server, learning_rate=0.2)
        before = trainer.predict(examples)[:, 0]
        for epoch in range(10):
            trainer.train_microbatch(examples, f"epoch-{epoch}")
        after = trainer.predict(examples)[:, 0]
        labels = [example.labels["click"] for example in examples]
        self.assertEqual(float(auc([PredictionRecord(0, y, s, 0) for y, s in zip(labels, before)])), 0.5)
        self.assertGreater(float(auc([PredictionRecord(0, y, s, 1) for y, s in zip(labels, after)])), 0.95)


class EvaluationAndConsistencyTest(unittest.TestCase):
    def test_offline_online_auc_gap_and_slice_are_reported(self) -> None:
        offline = [
            PredictionRecord(index % 2, label, score, 1, "new" if index < 2 else "old")
            for index, (label, score) in enumerate([(0, 0.1), (1, 0.9), (0, 0.4), (1, 0.6)])
        ]
        online = [
            PredictionRecord(record.user_id, record.label, 1.0 - record.score, 2, record.slice_name)
            for record in offline
        ]
        report = compare(offline, online)
        self.assertEqual(report.offline.auc, 1.0)
        self.assertEqual(report.online.auc, 0.0)
        self.assertEqual(report.auc_gap, -1.0)
        self.assertIn("new", report.slice_auc)

    def test_manifest_or_feature_mismatch_fails_consistency(self) -> None:
        expected = ChainManifest("schema-v1", "v2", "joiner-v1", 4, "index-v4")
        served = ChainManifest("schema-v2", "v2", "joiner-v1", 4, "index-v4")
        example = TrainingExample(
            "e", 1, 2, 3, (10, 11), (1, 2), {task: 0.0 for task in TEST_TASKS}, 1.0, "schema-v1"
        )
        self.assertEqual(expected.tasks, TEST_TASKS)
        report = ChainConsistencyAuditor().audit(
            expected, served, example, (10, 99), np.array([0.2]), np.array([0.2])
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.checks["schema_version"])
        self.assertFalse(report.checks["feature_replay"])


if __name__ == "__main__":
    unittest.main()
