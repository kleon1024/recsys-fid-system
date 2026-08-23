from __future__ import annotations

import json
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
from fid_lab.evolution.models.esmm import ESMM
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
from fid_lab.feed_loop.world_model.external.kuairand.data.sequence import build_history
from fid_lab.feed_loop.world_model.external.kuairand.modeling.architectures import (
    KuaiSequenceMMoE,
    KuaiSequenceTransformer,
    KuaiWideDeep,
)
from fid_lab.feed_loop.world_model.external.kuairand.retrieval.model import (
    KuaiMultiInterestRetriever,
    KuaiTwoTowerRetriever,
)
from fid_lab.feed_loop.world_model.external.kuairand.retrieval.review import (
    build_retrieval_review,
)
from fid_lab.feed_loop.world_model.external.kuairand.kernel import (
    KuaiBehaviorKernel,
    SlateResponse,
)
from fid_lab.feed_loop.world_model.external.kuairand.launch.contracts import (
    PolicySpec,
    assert_artifact_compatible,
    stream_sha256,
)
from fid_lab.feed_loop.world_model.external.kuairand.launch.pipeline import (
    LaunchStage,
    LaunchState,
)
from fid_lab.feed_loop.world_model.external.kuairand.data.randomized import (
    validate_sparse,
)
from fid_lab.feed_loop.world_model.external.replay import treatment_guard_mask
from fid_lab.feed_loop.world_model.external.ope import policy_value_gates
from fid_lab.simulation.environment import FEATURE_NAMES
from fid_lab.simulation.policies import fit_logistic_policy


EXPECTED_HASH_VOCABULARIES = (1_002, 262_145, 262_145, 8_193, 4, 33, 1_025)


class MatureModelAdapterTest(unittest.TestCase):
    def test_ope_uses_one_primary_metric_and_bounded_guardrails(self) -> None:
        metrics = {
            "stay_norm": {"confidence_interval_95": [0.0003, 0.0007]},
            "long_view": {"confidence_interval_95": [-0.0004, 0.0002]},
            "hate": {"confidence_interval_95": [-0.00003, 0.00006]},
            "click": {"confidence_interval_95": [-0.0006, 0.0001]},
            "like": {"confidence_interval_95": [-0.0001, 0.0001]},
        }
        diagnostics = {
            "control": {"effective_sample_fraction": 0.65},
            "treatment": {"effective_sample_fraction": 0.64},
        }
        self.assertTrue(all(policy_value_gates(metrics, diagnostics).values()))
        metrics["hate"]["confidence_interval_95"][1] = 0.001
        self.assertFalse(policy_value_gates(metrics, diagnostics)["hate_guardrail"])

    def test_external_launch_state_is_ordered_and_hold_is_terminal(self) -> None:
        state = LaunchState().record(LaunchStage.OFFLINE_CAPACITY, True)
        state = state.record(LaunchStage.RANDOMIZED_CALIBRATION, True)
        held = state.record(LaunchStage.RANDOMIZED_OPE, False)
        self.assertEqual(held.active_authority, "v3")
        self.assertEqual(held.decision, "hold_randomized_ope")
        with self.assertRaisesRegex(ValueError, "terminal"):
            held.record(LaunchStage.STATEFUL_SHADOW, True)

    def test_external_refactor_preserves_published_golden_evidence(self) -> None:
        root = Path(__file__).resolve().parents[2]
        golden = json.loads(
            (root / "reports/refactor/2026-08-24-external-world-model-golden.json")
            .read_text()
        )
        for relative, record in golden["files"].items():
            self.assertEqual(stream_sha256(root / relative), record["sha256"])
            if "decision" in record:
                payload = json.loads((root / relative).read_text())
                self.assertEqual(payload["decision"], record["decision"])

    def test_external_artifact_identity_fails_closed_before_scoring(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"rows": 1}, root / "train.pt")
            torch.save({"items": 1}, root / "random_item_catalog.pt")
            manifest = {
                "schema": "test-v1",
                "sequence_length": 4,
                "feedback_names": ["click"],
                "sparse_names": ["user_id"],
                "sparse_vocabularies": [8],
                "dense_names": ["hour"],
                "catalog_sha256": stream_sha256(root / "random_item_catalog.pt"),
                "splits": {
                    "train": {
                        "sha256": stream_sha256(root / "train.pt"),
                    }
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            artifact = root / "model.pt"
            torch.save({"dataset_manifest": manifest}, artifact)
            self.assertEqual(
                assert_artifact_compatible(root, (artifact,))["schema"], "test-v1"
            )
            stale = dict(manifest)
            stale["catalog_sha256"] = "0" * 64
            torch.save({"dataset_manifest": stale}, artifact)
            with self.assertRaisesRegex(ValueError, "catalog_sha256"):
                assert_artifact_compatible(root, (artifact,))

    def test_external_policy_contract_rejects_invalid_exploration(self) -> None:
        self.assertEqual(PolicySpec().utility_mode, "raw_probability")
        with self.assertRaisesRegex(ValueError, "uniform_mixture"):
            PolicySpec(uniform_mixture=1.01)

    def test_esmm_trains_conversion_in_impression_space_and_preserves_funnel(self) -> None:
        model = ESMM(6, width=8)
        features = torch.rand(5, 6)
        click = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
        conversion = torch.tensor([1, 0, 0, 1, 0], dtype=torch.float32)
        output = model(features)
        self.assertTrue(torch.all(output.pctcvr <= output.pctr))
        loss = model.entire_space_loss(features, click, conversion)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "must imply click"):
            model.entire_space_loss(features, click, torch.ones_like(click))

    def test_external_sparse_contract_rejects_embedding_index_at_vocabulary(self) -> None:
        sparse = np.zeros((1, len(EXPECTED_HASH_VOCABULARIES)), dtype=np.int64)
        sparse[0, 5] = EXPECTED_HASH_VOCABULARIES[5]
        with self.assertRaisesRegex(ValueError, "upload_type"):
            validate_sparse(sparse)

    def test_external_history_is_point_in_time_and_user_bounded(self) -> None:
        logs = pd.DataFrame({"user_id": [1, 1, 1, 2]})
        items = np.asarray([11, 12, 13, 21], dtype=np.int64)
        feedback = np.zeros((4, 7), dtype=np.uint8)
        feedback[0, 1] = 1
        history_items, history_feedback = build_history(
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
            KuaiSequenceMMoE((10, 10, 10, 5, 5, 5, 5), 11, 4, width=16),
        ):
            output = model(sparse, dense, history_items, history_feedback)
            self.assertEqual(output.shape, (4, 8))
            self.assertTrue(torch.isfinite(output).all())

    def test_external_transformer_cached_slate_matches_pointwise_scores(self) -> None:
        torch.manual_seed(17)
        model = KuaiSequenceTransformer((10, 10, 10, 5, 5, 5, 5), 11, 4)
        model.eval()
        sparse = torch.tensor([[[1, 2, 3, 1, 1, 1, 1]] * 3] * 2)
        dense = torch.rand(2, 3, 11)
        history_items = torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6]])
        history_feedback = torch.zeros(2, 4, 7)
        cached = model.score_slate(
            sparse, dense, history_items, history_feedback
        )
        pointwise = model(
            sparse.reshape(-1, 7), dense.reshape(-1, 11),
            history_items[:, None].expand(-1, 3, -1).reshape(-1, 4),
            history_feedback[:, None].expand(-1, 3, -1, -1).reshape(-1, 4, 7),
        ).reshape(2, 3, 8)
        torch.testing.assert_close(cached, pointwise)

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
        eligible = treatment_guard_mask(base, candidate, torch.tensor([0]))
        self.assertEqual(eligible.tolist(), [[True, False]])

    def test_external_treatment_guard_uses_nonnegative_maximum_regressions(self) -> None:
        base_probability = torch.zeros(1, 3, 7)
        base_probability[0, :, 6] = 0.001
        base = SlateResponse(base_probability, torch.full((1, 3), 0.5))
        candidate_probability = base_probability.clone()
        candidate_probability[0, 1, 6] += 0.0004
        candidate_probability[0, 2, 6] += 0.0006
        candidate = SlateResponse(candidate_probability, torch.full((1, 3), 0.5))
        eligible = treatment_guard_mask(base, candidate, torch.tensor([0]))
        self.assertEqual(eligible.tolist(), [[True, True, False]])

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

    def test_external_retrieval_towers_are_ann_compatible(self) -> None:
        vocabularies = (16, 32, 24, 12, 4, 8, 10)
        sparse = torch.randint(1, 4, (5, 7))
        dense = torch.rand(5, 11)
        history_items = torch.randint(0, 16, (5, 6))
        history_feedback = torch.randint(0, 2, (5, 6, 7))
        for model, expected in (
            (KuaiTwoTowerRetriever(vocabularies, width=16), (5, 16)),
            (KuaiMultiInterestRetriever(
                vocabularies, width=16, interests=3
            ), (5, 3, 16)),
        ):
            query = model.encode_query(
                sparse, dense, history_items, history_feedback
            )
            items = model.encode_item(sparse, dense)
            self.assertEqual(tuple(query.shape), expected)
            self.assertEqual(tuple(items.shape), (5, 16))

    def test_external_retrieval_review_freezes_sample_and_retains_control(self) -> None:
        root = Path(__file__).resolve().parents[2]
        paths = [
            root / f"reports/launches/2026-08-24-feed-retrieval-{seed}.json"
            for seed in (20260824, 20260825, 20260826)
        ]
        review = build_retrieval_review(paths)
        self.assertEqual(review["active_retrieval_control"], "popular")
        self.assertEqual(review["decision"], "retain_popular_control")
        self.assertEqual(
            [row["decision"] for row in review["launches"]],
            [
                "hold_no_ranking_delta",
                "reject_unstable_or_regressive",
                "reject_unstable_or_regressive",
            ],
        )


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
