from __future__ import annotations

import unittest

import torch

from fid_lab.feed_loop.scale.artifact.features import build_tensor_features
from fid_lab.feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    candidate_batch,
    prepare_run,
    run_tensor_feed,
)
from fid_lab.feed_loop.scale.tensor_runtime.contracts import (
    EXTERNAL_MIXTURE_FEED_VERSION,
    LEGACY_LOCAL_SIGNAL_VERSION,
    LOCAL_NEURAL_SIGNAL_VERSION,
)
from fid_lab.feed_loop.scale.tensor_runtime.state import (
    advance_state,
    new_user_state,
)
from fid_lab.feed_loop.tensor_cascade import select_candidate
from fid_lab.feed_loop.tensor_policies import PERSONALIZED


def _config(users=64):
    return TensorFeedConfig(
        users=users,
        steps=2,
        candidates=8,
        route_candidates=4,
        merged_candidates=20,
        audit_candidates=8,
        catalog_items=512,
        catalog_creators=256,
        batch_users=users,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        device="cpu",
    )


class ExternalFeedDGPTest(unittest.TestCase):
    def test_feed_and_local_response_authorities_are_independent(self):
        legacy = _config(16)
        combined = TensorFeedConfig(
            **{
                **legacy.__dict__,
                "local_signal_version": LOCAL_NEURAL_SIGNAL_VERSION,
            }
        )
        self.assertEqual(legacy.signal_version, EXTERNAL_MIXTURE_FEED_VERSION)
        self.assertEqual(
            legacy.local_signal_version, LEGACY_LOCAL_SIGNAL_VERSION
        )
        self.assertEqual(combined.signal_version, EXTERNAL_MIXTURE_FEED_VERSION)
        self.assertEqual(
            combined.local_signal_version, LOCAL_NEURAL_SIGNAL_VERSION
        )

    def test_external_world_is_required_by_runtime(self):
        with self.assertRaisesRegex(ValueError, "evidence-bound behavior world"):
            run_tensor_feed(_config(16), PERSONALIZED)

    def test_hidden_mixture_is_not_part_of_serving_feature_vector(self):
        config = _config()
        device, generator, catalog = prepare_run(config, None, 0, None)
        user_ids = torch.arange(config.users)
        state = new_user_state(
            config, PERSONALIZED, generator, device, user_ids
        )
        candidates = candidate_batch(
            config, generator, device, state, catalog, 0, PERSONALIZED
        )
        before = build_tensor_features(config, user_ids, state, candidates, 0)
        state["hidden_mixture"] = torch.remainder(
            state["hidden_mixture"] + 1, 4
        )
        state["hidden_novelty"] = 1.0 - state["hidden_novelty"]
        state["hidden_patience"] = 1.0 - state["hidden_patience"]
        after = build_tensor_features(config, user_ids, state, candidates, 0)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(set(state["hidden_mixture"].tolist()), {0, 1, 2, 3})

    def test_selected_feedback_advances_only_the_last_history_slot(self):
        config = _config(32)
        device, generator, catalog = prepare_run(config, None, 0, None)
        user_ids = torch.arange(config.users)
        state = new_user_state(
            config, PERSONALIZED, generator, device, user_ids
        )
        candidates = candidate_batch(
            config, generator, device, state, catalog, 0, PERSONALIZED
        )
        selected = select_candidate(
            PERSONALIZED, user_ids, state, candidates, device, 0, config
        )
        feedback = torch.zeros(config.users, 7, dtype=torch.bool)
        feedback[:, 0] = True
        false = torch.zeros(config.users, dtype=torch.bool)
        values = {
            "long_view": false,
            "like": false,
            "negative": false,
            "comment": false,
            "share": false,
            "follow": false,
            "anchor": false,
            "ad_selected": false,
            "live_selected": false,
            "history_item": selected["item_ids"] + 1,
            "history_feedback": feedback,
        }
        advance_state(
            config, PERSONALIZED, generator, state, selected, values, 0
        )
        self.assertTrue((state["behavior_history_items"][:, :-1] == 0).all())
        self.assertTrue(torch.equal(
            state["behavior_history_items"][:, -1], selected["item_ids"] + 1
        ))
        self.assertTrue(torch.equal(
            state["behavior_history_feedback"][:, -1], feedback.to(torch.uint8)
        ))


if __name__ == "__main__":
    unittest.main()
