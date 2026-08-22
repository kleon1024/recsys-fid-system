from __future__ import annotations

import unittest

import numpy as np
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
from fid_lab.generative.semantic_ids import SemanticIdIndex


class MatureModelAdapterTest(unittest.TestCase):
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
