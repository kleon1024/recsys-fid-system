from __future__ import annotations

import unittest

import torch

from fid_lab.feed_loop.scale.tensor_runtime.local_response import (
    LOCAL_NEURAL_SIGNAL_VERSION,
    _neural_logits,
    sample_local_response,
)


class LocalNeuralDGPTest(unittest.TestCase):
    def _inputs(self, rows=2_000):
        generator = torch.Generator().manual_seed(17)
        user_ids = torch.arange(rows)
        floats = [torch.rand(rows, generator=generator) for _ in range(7)]
        return (
            user_ids,
            floats[0] * 2.0 - 1.0,
            *floats[1:],
            torch.remainder(user_ids, 3),
        )

    def test_v4_is_deterministic_and_preserves_the_behavior_funnel(self):
        values = self._inputs()
        arguments = (
            values[0], 3, 20260824, LOCAL_NEURAL_SIGNAL_VERSION,
            torch.ones(len(values[0]), dtype=torch.bool), values[1],
            torch.ones(len(values[0])), *values[2:],
        )
        first = sample_local_response(*arguments)
        second = sample_local_response(*arguments)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        anchor, detail, favorite, paid, pixel = first
        self.assertFalse((detail & ~anchor).any())
        self.assertFalse((favorite & ~detail).any())
        self.assertFalse(((paid | pixel) & ~detail).any())

    def test_hidden_teacher_contains_non_additive_interactions(self):
        values = list(self._inputs(64))
        baseline = _neural_logits(*values)
        affinity = values.copy()
        affinity[1] = (affinity[1] + 0.2).clamp(-1.0, 1.0)
        search = values.copy()
        search[6] = (search[6] + 0.2).clamp(0.0, 1.0)
        both = affinity.copy()
        both[6] = search[6]
        interaction = (
            _neural_logits(*both) - _neural_logits(*affinity)
            - _neural_logits(*search) + baseline
        )
        self.assertGreater(float(interaction.abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
