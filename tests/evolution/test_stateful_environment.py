import unittest

import numpy as np

from fid_lab.simulation.contracts import SimulationConfig
from fid_lab.simulation.environment import StatefulFeedEnv, build_catalog
from fid_lab.simulation.experimentation import FeedParameters
from fid_lab.simulation.policies import ParameterizedPolicy, PopularPolicy
from fid_lab.simulation.population import run_population


class StatefulEnvironmentTest(unittest.TestCase):
    def test_latent_preference_is_not_an_online_feature(self):
        config = SimulationConfig(users=2, items=300, candidates=10)
        environment = StatefulFeedEnv(config, build_catalog(config))
        observation, _ = environment.reset(options={"user_id": 7})
        item_id = int(environment.candidates[0])
        probability_before = environment._behavior_probabilities(
            observation[0], item_id
        )["long_view"]

        environment.interest = np.roll(environment.interest, 3)
        environment.satisfaction = 0.8
        environment.fatigue = 0.7
        environment.trust = 0.1
        environment.commerce_propensity = 0.9
        observation_after = environment._observation()
        probability_after = environment._behavior_probabilities(
            observation_after[0], item_id
        )["long_view"]

        np.testing.assert_allclose(observation, observation_after)
        self.assertNotEqual(probability_before, probability_after)

    def test_feed_and_local_routes_feed_a_bounded_coarse_pool(self):
        config = SimulationConfig(users=2, items=500, candidates=20)
        environment = StatefulFeedEnv(config, build_catalog(config))
        environment.reset(options={"user_id": 9})
        routes = set(environment.candidate_provider._routes(environment))

        self.assertEqual(environment.coarse_count, 20)
        self.assertGreater(environment.recall_count, environment.coarse_count)
        self.assertEqual(
            routes,
            {
                "ann",
                "graph",
                "geo",
                "fresh",
                "long_tail",
                "popular",
                "post_search",
                "retarget",
            },
        )

    def test_resolved_parameters_drive_recall_and_are_logged(self):
        config = SimulationConfig(users=1, items=500, candidates=10)
        parameters = FeedParameters(
            enabled_routes=("ann", "popular"),
            recall_budget=60,
            coarse_budget=10,
            fine_model="popular_baseline",
        )
        catalog = build_catalog(config)
        environment = StatefulFeedEnv(config, catalog, parameters)
        environment.reset(options={"user_id": 5})
        self.assertEqual(set(environment.candidate_provider._routes(environment)), {"ann", "popular"})
        trajectory = run_population(
            config, catalog, PopularPolicy(), [5], parameters=parameters
        )[0]
        self.assertEqual(trajectory.rows[0].parameter_snapshot["recall_budget"], 60)
        self.assertEqual(
            trajectory.rows[0].parameter_snapshot["enabled_routes"],
            ("ann", "popular"),
        )

    def test_trained_retrieval_fails_closed_without_matching_snapshot(self):
        config = SimulationConfig(users=1, items=300, candidates=10)
        parameters = FeedParameters(
            recall_model="two_tower_trained_v2",
            coarse_budget=10,
        )
        environment = StatefulFeedEnv(config, build_catalog(config), parameters)
        with self.assertRaisesRegex(ValueError, "requires a retrieval snapshot"):
            environment.reset(options={"user_id": 5})

    def test_fine_model_binding_fails_closed_and_value_parameters_change_score(self):
        policy = PopularPolicy()
        features = np.zeros((2, 18), dtype=np.float32)
        features[:, 0] = (0.2, 0.8)
        features[:, 1] = (0.7, 0.4)
        features[:, 3] = (0.3, 0.3)
        parameters = FeedParameters(
            fine_model=policy.name,
            hlt_weight=2.0,
            diversity_strength=0.1,
        )
        configured = ParameterizedPolicy(policy, parameters)
        self.assertFalse(np.allclose(configured.score(features), policy.score(features)))
        with self.assertRaises(ValueError):
            ParameterizedPolicy(policy, FeedParameters(fine_model="wrong-model"))


if __name__ == "__main__":
    unittest.main()
