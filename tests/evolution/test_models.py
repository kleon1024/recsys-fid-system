from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from fid_lab.evolution.models.deepctr_adapter import (
    DeepCTRModelAdapter,
    build_feature_bundle,
    supported_deepctr_models,
)
from fid_lab.evolution.models.generative import (
    AutoregressiveSemanticDecoder,
    LearnedGenerativeRetriever,
    SemanticVocabulary,
    SessionGenerator,
)
from fid_lab.evolution.evaluation.retrieval_benchmark import run_retrieval_benchmark
from fid_lab.evolution.models.retrieval import RetrievalSnapshot, TwoTowerRetriever
from fid_lab.generative.semantic_ids import SemanticIdIndex
from fid_lab.feed_loop.models.artifact import (
    canonical_feature_schema_hash,
    feature_schema_hash,
    publish_policy,
)
from fid_lab.feed_loop.models.deep_policy import DENSE_INDICES, SPARSE_SPECS
from fid_lab.feed_loop.world_model.benchmark.contracts import capacity_gates
from fid_lab.feed_loop.world_model.external.kuairand.dataset import _history
from fid_lab.feed_loop.world_model.external.kuairand.models import (
    KuaiSequenceTransformer,
    KuaiWideDeep,
)
from fid_lab.feed_loop.world_model.external.kuairand.kernel import (
    KuaiBehaviorKernel,
    SlateResponse,
)
from fid_lab.feed_loop.world_model.external.replay import _guard_mask
from fid_lab.simulation.environment import FEATURE_NAMES
from fid_lab.simulation.policies import fit_logistic_policy


class MatureModelAdapterTest(unittest.TestCase):
    def test_external_history_is_point_in_time_and_user_bounded(self) -> None:
        logs = pd.DataFrame({"user_id": [1, 1, 1, 2]})
        items = np.asarray([11, 12, 13, 21], dtype=np.int64)
        feedback = np.zeros((4, 7), dtype=np.uint8)
        feedback[0, 1] = 1
        history_items, history_feedback = _history(
            logs, items, feedback, length=3
        )
        self.assertEqual(history_items[2].tolist(), [0, 11, 12])
        self.assertEqual(history_feedback[2, 1, 1], 1)
        self.assertEqual(history_items[3].tolist(), [0, 0, 0])

    def test_external_behavior_models_share_multitask_contract(self) -> None:
        sparse = torch.tensor([[1, 2, 3, 1, 1, 1, 1]] * 4)
        dense = torch.rand(4, 11)
        history_items = torch.tensor([[0, 0, 2, 3]] * 4)
        history_feedback = torch.zeros(4, 4, 7)
        for model in (
            KuaiWideDeep((10, 10, 10, 5, 5, 5, 5), 11),
            KuaiSequenceTransformer((10, 10, 10, 5, 5, 5, 5), 11, 4),
        ):
            output = model(sparse, dense, history_items, history_feedback)
            self.assertEqual(output.shape, (4, 8))
            self.assertTrue(torch.isfinite(output).all())

    def test_external_kernel_updates_only_selected_point_in_time_history(self) -> None:
        history_items = torch.tensor([[0, 2, 3], [0, 4, 5]])
        history_feedback = torch.zeros(2, 3, 7)
        actions = torch.tensor([
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
        ])
        next_items, next_feedback = KuaiBehaviorKernel.advance_history(
            history_items, history_feedback, torch.tensor([7, 8]), actions
        )
        self.assertEqual(next_items.tolist(), [[2, 3, 7], [4, 5, 8]])
        self.assertEqual(next_feedback[:, -1].tolist(), actions.tolist())
        response = SlateResponse(torch.rand(2, 3, 7), torch.rand(2, 3))
        self.assertEqual(response.probabilities.shape, (2, 3, 7))

    def test_external_treatment_guard_rejects_predicted_hate_regression(self) -> None:
        base = SlateResponse(torch.zeros(1, 2, 7), torch.full((1, 2), 0.5))
        candidate_probability = torch.zeros(1, 2, 7)
        candidate_probability[0, 1, 6] = 0.9
        candidate = SlateResponse(candidate_probability, torch.full((1, 2), 0.5))
        eligible = _guard_mask(base, candidate, torch.tensor([0]))
        self.assertEqual(eligible.tolist(), [[True, False]])

    def test_v4_capacity_gate_rejects_an_unused_sequence(self) -> None:
        context = {
            "permuted_sequence": {"relative_to_baseline_std": 0.01},
            "selected_only_slate": {"relative_to_baseline_std": 0.20},
        }
        models = {
            name: {"auc": auc, "request": {"oracle_regret": regret}}
            for name, auc, regret in (
                ("logistic_regression", 0.58, 0.01),
                ("xgboost", 0.59, 0.005),
                ("wide_deep", 0.57, 0.02),
                ("deepfm", 0.57, 0.02),
                ("dcnv2", 0.58, 0.01),
                ("din_request", 0.58, 0.01),
                ("slate_transformer", 0.58, 0.01),
            )
        }
        gates = capacity_gates(context, models)
        self.assertFalse(gates["sequence_context_material"])
        self.assertTrue(gates["slate_context_material"])
        self.assertFalse(gates["request_model_auc_gain"])

    def test_feed_deep_models_cover_every_canonical_feature_once(self) -> None:
        sparse_indices = tuple(index for _, index, _ in SPARSE_SPECS)
        covered = (*sparse_indices, *DENSE_INDICES)
        self.assertEqual(len(covered), len(FEATURE_NAMES))
        self.assertEqual(set(covered), set(range(len(FEATURE_NAMES))))
        self.assertEqual(len(covered), len(set(covered)))

    def test_published_policy_binds_schema_signal_and_exact_replay(self) -> None:
        rng = np.random.default_rng(71)
        features = rng.normal(size=(128, 24)).astype(np.float32)
        labels = (features[:, 0] * features[:, 1] > 0).astype(np.float32)
        policy = fit_logistic_policy(
            "artifact_lr",
            features,
            labels,
            tuple(range(features.shape[1])),
            seed=71,
        )
        published, replay_delta = publish_policy(
            policy, features, "heterogeneous-nonlinear-v2"
        )
        self.assertLess(replay_delta, 1e-12)
        self.assertTrue(published.artifact_manifest["artifact_id"].startswith("sha256:"))
        self.assertEqual(
            published.artifact_manifest["signal_version"],
            "heterogeneous-nonlinear-v2",
        )
        self.assertEqual(len(published.artifact_manifest["feature_schema_sha256"]), 64)
        self.assertEqual(
            published.artifact_manifest["feature_schema_sha256"],
            canonical_feature_schema_hash(),
        )
        self.assertNotEqual(canonical_feature_schema_hash(), feature_schema_hash())
        self.assertEqual(
            len(published.artifact_manifest["model_input_schema_sha256"]), 64
        )
        self.assertEqual(published.artifact_manifest["serving_device"], "cpu")

    def test_deepctr_adapter_uses_supported_model_zoo(self) -> None:
        self.assertEqual(
            supported_deepctr_models(),
            ("wide_deep", "deepfm", "dcnv2", "din", "mmoe", "ple"),
        )
        rng = np.random.default_rng(8)
        sparse = rng.integers(64, size=(128, 6))
        dense = rng.normal(size=(128, 10)).astype(np.float32)
        labels = rng.integers(2, size=128).astype(np.float32)
        bundle = build_feature_bundle(sparse, dense)
        model = DeepCTRModelAdapter("wide_deep", bundle)
        model.fit(bundle.inputs, labels, batch_size=64)
        self.assertEqual(model.predict(bundle.inputs).shape, (128, 1))
        self.assertGreater(model.parameters, 0)

    def test_faiss_and_tower_retrieval_share_candidate_budget(self) -> None:
        report = run_retrieval_benchmark(items=200, queries=80, top_k=10)
        self.assertEqual(report["top_k"], 10)
        self.assertIn("exact_content", report["models"])
        self.assertIn("two_tower", report["models"])
        self.assertTrue(report["split_contract"]["query_disjoint"])
        self.assertTrue(report["split_contract"]["frozen_item_corpus"])
        self.assertEqual(
            report["negative_sampling"]["source_fractions"],
            {"in_batch": 0.6, "hard": 0.25, "random": 0.15},
        )

    def test_retrieval_snapshot_round_trip_matches_model(self) -> None:
        torch.manual_seed(11)
        model = TwoTowerRetriever(6, 6, representation_dim=4)
        items = torch.randn(20, 6)
        query = torch.randn(6)
        snapshot = model.export_snapshot(items, "retrieval-test-v1")
        expected = (
            model.encode_item(items) @ model.encode_query(query[None]).squeeze(0)
        ).detach().numpy()
        np.testing.assert_allclose(snapshot.scores(query.numpy()), expected, atol=1e-6)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.npz"
            snapshot.save(path)
            loaded = RetrievalSnapshot.load(path)
        self.assertEqual(loaded.version, snapshot.version)
        np.testing.assert_allclose(loaded.scores(query.numpy()), expected, atol=1e-6)


class GenerativeModelTest(unittest.TestCase):
    def test_learned_decoder_is_prefix_constrained_and_session_safe(self) -> None:
        rng = np.random.default_rng(9)
        item_ids = np.arange(24)
        embeddings = rng.normal(size=(24, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = SemanticIdIndex.fit(item_ids, embeddings, levels=2, codebook_size=4)
        vocabulary = SemanticVocabulary.from_index(index)
        model = AutoregressiveSemanticDecoder(8, vocabulary, hidden=16)
        retriever = LearnedGenerativeRetriever(index, model)
        query = torch.from_numpy(embeddings[0])
        generated = retriever.retrieve(query, limit=5, beam_size=10)
        self.assertTrue(generated)
        self.assertTrue(all(value.semantic_id in index.codes.values() for value in generated))
        session = SessionGenerator(
            retriever,
            {int(item): int(item % 5) for item in item_ids},
            {int(item): int(item % 3) for item in item_ids},
        ).generate(query, size=5, author_cap=2, category_cap=3)
        authors = [value.item_id % 5 for value in session]
        self.assertLessEqual(max(authors.count(author) for author in set(authors)), 2)


if __name__ == "__main__":
    unittest.main()
