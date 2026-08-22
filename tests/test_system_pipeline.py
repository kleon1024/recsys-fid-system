from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from fid_lab.online.catalog import ItemCatalog, make_catalog, make_request
from fid_lab.online.config import PolicyConfig, RecallConfig, ValueTreeConfig
from fid_lab.online.domain import Candidate, Item
from fid_lab.online.pipeline import RecommendationPipeline
from fid_lab.online.stages.policy import ConstrainedPolicyOptimizer
from fid_lab.online.stages.ranking import ValueTree
from fid_lab.online.stages.retrieval import LocalVikingIndex, RecallHit, RecallMerger


RECALL_RULE = RecallConfig(
    route_weights={"viking": 1.0, "popular": 0.35, "fresh": 0.25},
    reciprocal_rank_constant=20.0,
)
VALUE_RULE = ValueTreeConfig(
    engagement_weights={"p_click": 0.45, "p_like": 0.25, "p_long_view": 0.30},
    ecosystem_weights={"quality": 0.75, "freshness": 0.25},
    root_weights={"engagement": 0.72, "ecosystem": 0.28},
)
POLICY_RULE = PolicyConfig(
    min_fresh=3,
    max_per_creator=2,
    max_per_category=7,
    exploration_bonus=0.025,
)
EXPECTED_TYPE_CAPS = {"organic": 20, "live": 3, "ad": 2}


class RecallTest(unittest.TestCase):
    def test_viking_returns_nearest_vector_first(self) -> None:
        items = [
            Item(1, "organic", "a", 1, np.array([1.0, 0.0]), 0.1, 0.8, 1.0, frozenset({"SG"})),
            Item(2, "organic", "b", 2, np.array([0.0, 1.0]), 0.1, 0.8, 1.0, frozenset({"SG"})),
        ]
        catalog = ItemCatalog(items)
        request = make_request(make_catalog(size=30, dimension=2), user_id=2)
        request = request.__class__(**{**request.__dict__, "user_embedding": np.array([0.9, 0.1])})
        self.assertEqual(LocalVikingIndex(catalog).recall(request, 1)[0].item_id, 1)

    def test_recall_merge_deduplicates_and_preserves_sources(self) -> None:
        merger = RecallMerger(RECALL_RULE)
        result = merger.merge(
            {
                "viking": [RecallHit(7, 0.9, "viking", "vector")],
                "popular": [RecallHit(7, 0.8, "popular", "trend")],
                "fresh": [RecallHit(8, 0.7, "fresh", "new")],
            },
            10,
        )
        self.assertEqual([candidate.item_id for candidate in result].count(7), 1)
        self.assertEqual(set(result[0].recall_scores), {"viking", "popular"})


class RankingTest(unittest.TestCase):
    def test_value_tree_matches_independent_written_formula(self) -> None:
        predictions = {
            "p_click": 0.5,
            "p_like": 0.4,
            "p_long_view": 0.6,
            "quality": 0.8,
            "freshness": 0.2,
        }
        score, nodes = ValueTree(VALUE_RULE).evaluate(predictions)
        engagement = 0.45 * 0.5 + 0.25 * 0.4 + 0.30 * 0.6
        ecosystem = 0.75 * 0.8 + 0.25 * 0.2
        expected = 0.72 * engagement + 0.28 * ecosystem
        self.assertAlmostEqual(nodes["engagement"], engagement)
        self.assertAlmostEqual(score, expected)

    def test_policy_enforces_creator_and_category_caps(self) -> None:
        catalog = make_catalog(size=100)
        candidates = [Candidate(item.item_id, rule_score=1.0 - item.item_id / 1000) for item in catalog.items]
        selected = ConstrainedPolicyOptimizer(
            catalog,
            POLICY_RULE,
            36.0,
        ).select(candidates, 30)
        creators = Counter(catalog.get(value.item_id).creator_id for value in selected)
        categories = Counter(catalog.get(value.item_id).category for value in selected)
        self.assertLessEqual(max(creators.values()), 2)
        self.assertLessEqual(max(categories.values()), 7)


class EndToEndTest(unittest.TestCase):
    def test_complete_chain_is_deterministic_and_auditable(self) -> None:
        catalog = make_catalog()
        pipeline = RecommendationPipeline(catalog)
        request = make_request(catalog)
        first = pipeline.recommend(request)
        second = pipeline.recommend(request)
        self.assertEqual([item.item_id for item in first.items], [item.item_id for item in second.items])
        self.assertEqual(len(first.items), request.size)
        self.assertEqual(
            [trace.stage for trace in first.traces],
            [
                "recall", "recall_merge", "eligibility", "feature_join", "coarse_rank",
                "fine_rank_value_tree", "ranking_rules", "copp_policy", "mixed_rank",
            ],
        )
        for candidate in first.items:
            item = catalog.get(candidate.item_id)
            self.assertTrue(item.is_safe and item.is_active)
            self.assertNotIn(item.item_id, request.seen_item_ids)
            self.assertEqual(len(candidate.feature_fids), 9)
        type_counts = Counter(catalog.get(value.item_id).content_type for value in first.items)
        for item_type, count in type_counts.items():
            self.assertLessEqual(count, EXPECTED_TYPE_CAPS[item_type])
        creators = Counter(catalog.get(value.item_id).creator_id for value in first.items)
        categories = Counter(catalog.get(value.item_id).category for value in first.items)
        self.assertLessEqual(max(creators.values()), 2)
        self.assertLessEqual(max(categories.values()), 7)
        self.assertGreaterEqual(
            sum(catalog.get(value.item_id).age_hours <= 36.0 for value in first.items), 3
        )
        category_sequence = [catalog.get(value.item_id).category for value in first.items]
        self.assertFalse(
            any(
                category_sequence[index] == category_sequence[index + 1] == category_sequence[index + 2]
                for index in range(len(category_sequence) - 2)
            )
        )
        self.assertEqual(first.artifact_versions["vector_index"], pipeline.viking.version)


if __name__ == "__main__":
    unittest.main()
