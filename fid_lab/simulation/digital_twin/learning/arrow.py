"""Arrow-to-tensor conversions shared by persisted learning authorities."""

from __future__ import annotations

import torch


def list_column_to_tensor(column, dtype: torch.dtype) -> torch.Tensor:
    values = column.combine_chunks()
    width = len(values[0]) if len(values) else 0
    flat = values.values.to_numpy(zero_copy_only=False)
    return torch.as_tensor(flat.copy(), dtype=dtype).reshape(len(values), width)
