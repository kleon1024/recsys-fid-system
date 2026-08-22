import unittest

import numpy as np

from fid_lab.simulation.contracts import SimulationConfig
from fid_lab.simulation.environment import StatefulFeedEnv, build_catalog


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

    def test_six_routes_feed_a_bounded_coarse_pool(self):
        config = SimulationConfig(users=2, items=500, candidates=20)
        environment = StatefulFeedEnv(config, build_catalog(config))
        environment.reset(options={"user_id": 9})
        routes = set(environment.candidate_provider._routes(environment))

        self.assertEqual(environment.coarse_count, 20)
        self.assertGreater(environment.recall_count, environment.coarse_count)
        self.assertEqual(routes, {"ann", "graph", "geo", "fresh", "long_tail", "popular"})


if __name__ == "__main__":
    unittest.main()
