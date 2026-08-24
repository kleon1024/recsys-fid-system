from __future__ import annotations

import torch

from fid_lab.scoring import request_standardize


def test_request_standardization_is_positive_affine_invariant():
    values = torch.tensor([[1.0, 3.0, 2.0], [8.0, -2.0, 4.0]])
    transformed = 7.5 * values + 31.0
    torch.testing.assert_close(
        request_standardize(values), request_standardize(transformed)
    )


def test_request_standardization_is_finite_for_single_candidate():
    result = request_standardize(torch.tensor([[4.0], [-2.0]]))
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))
