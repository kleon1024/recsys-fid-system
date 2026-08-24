import torch

from fid_lab.feed_loop.scale.model_ladder.v4.contracts import TASKS
from fid_lab.feed_loop.scale.model_ladder.v4.networks import (
    MMoERanker,
    PLERanker,
    SingleTaskDIN,
    SingleTaskTransformer,
)
from fid_lab.feed_loop.scale.model_ladder.v4.training import request_loss


def _batch(rows=8, candidates=12, sequence_length=16):
    labels = torch.zeros(rows, 21)
    labels[:, 1] = torch.randint(0, 2, (rows,)).float()
    labels[:, 2] = torch.rand(rows) * 120.0
    labels[:, 3] = torch.rand(rows)
    for index in (5, 6, 7, 8, 9, 12, 13):
        labels[:, index] = torch.randint(0, 2, (rows,)).float()
    return {
        "candidates": torch.rand(rows, candidates, 28),
        "sequence": torch.rand(rows, sequence_length, 8),
        "exposed_index": torch.arange(rows) % candidates,
        "labels": labels,
        "masks": torch.ones(rows, 21, dtype=torch.bool),
        "weights": torch.ones(rows),
    }


def test_v4_request_rankers_share_the_request_contract():
    batch = _batch()
    models = (
        SingleTaskDIN(), SingleTaskTransformer(),
        MMoERanker(len(TASKS)), PLERanker(len(TASKS)),
    )
    for model in models:
        output = model(batch["candidates"], batch["sequence"])
        assert output.shape == (
            len(batch["labels"]), batch["candidates"].shape[1], model.task_count
        )
        loss = request_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
