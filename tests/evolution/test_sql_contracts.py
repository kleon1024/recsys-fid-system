from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ClickHouseContractTest(unittest.TestCase):
    def test_queries_cover_joiner_attribution_funnel_and_replay(self) -> None:
        sql = "\n".join(
            path.read_text()
            for path in sorted((ROOT / "sql" / "clickhouse").glob("*.sql"))
        )
        for contract in (
            "recommendation_decision_log",
            "commerce_event_log",
            "pixel_event_log",
            "fractional_label",
            "teacher_topk_preservation",
            "feature_replay_log",
            "recommendation_candidate_decision_log",
            "recommendation_mature_label_log",
            "recall_miss",
            "mix_rank_miss",
            "v4_request_log",
            "v4_route_candidate_log",
            "v4_candidate_decision_log",
            "v4_mature_label_log",
            "v4_checkpoint_log",
        ):
            self.assertIn(contract, sql)


if __name__ == "__main__":
    unittest.main()
