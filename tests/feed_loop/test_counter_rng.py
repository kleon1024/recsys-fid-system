"""Counter random streams must remain valid at industrial ID scale."""

import unittest

import torch

from fid_lab.simulation.randomness import uniform, uniform_for_items


class CounterRngTest(unittest.TestCase):
    def test_uniform_is_half_open_for_one_million_entity_ids(self):
        entity_ids = torch.arange(1_000_000)
        values = uniform(entity_ids, 31, 227, 20260824)
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLess(float(values.max()), 1.0)

    def test_item_uniform_is_half_open_after_wide_counter_mixing(self):
        entity_ids = torch.arange(100_000)
        item_ids = torch.arange(32).expand(100_000, -1)
        values = uniform_for_items(
            entity_ids, item_ids, 17, 229, 20260824
        )
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLess(float(values.max()), 1.0)

    def test_event_streams_are_not_shifted_copies(self):
        entity_ids = torch.arange(100_000)
        item_ids = torch.remainder(entity_ids * 17 + 5, 32_768)
        first = uniform_for_items(
            entity_ids, item_ids, 0, 201, 20260824
        )
        second = uniform_for_items(
            entity_ids, item_ids, 0, 203, 20260824
        )
        correlation = torch.corrcoef(torch.stack((first, second)))[0, 1]
        self.assertLess(abs(float(correlation)), 0.02)


if __name__ == "__main__":
    unittest.main()
