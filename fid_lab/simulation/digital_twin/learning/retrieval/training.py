"""Train retrieval models from the persisted v4 RecallExample authority."""

from __future__ import annotations

from time import perf_counter

import torch

from ...samples import corrected_sampled_softmax_loss
from .contracts import (
    DEFAULT_RETRIEVAL_FEATURE_CONTRACT,
    RetrievalCorpus,
    RetrievalModelConfig,
    RetrievalQueryBatch,
)
from .model import ObservableRetrievalModel, retrieval_scores


def _batch_loss(
    model: ObservableRetrievalModel,
    batch: RetrievalQueryBatch,
    device: torch.device,
) -> torch.Tensor:
    query = model.encode_queries(batch)
    positive_state = model.encode_items(batch.positive_item_id.to(device))
    positive = retrieval_scores(query, positive_state[:, None]).squeeze(1)
    negative_id = batch.negative_item_id.to(device)
    negative_state = model.encode_items(negative_id.reshape(-1)).reshape(
        len(batch.request_id), negative_id.shape[1], -1,
    )
    negative = retrieval_scores(query, negative_state)
    loss = corrected_sampled_softmax_loss(
        positive / model.config.temperature,
        negative / model.config.temperature,
        batch.negative_expected_count.to(device),
        batch.negative_loss_mask.to(device),
        reduction="none",
    )
    weight = (1.0 + torch.log1p(batch.positive_strength.to(device))).clamp_max(4.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_retrieval_model(
    batch: RetrievalQueryBatch,
    corpus: RetrievalCorpus,
    config: RetrievalModelConfig,
    *,
    device: str | torch.device,
) -> tuple[ObservableRetrievalModel, dict[str, object]]:
    target = torch.device(device)
    torch.manual_seed(config.seed)
    model = ObservableRetrievalModel(corpus, config).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    started = perf_counter()
    for epoch in range(config.epochs):
        order = torch.randperm(len(batch.request_id), generator=generator)
        losses = []
        model.train()
        for start in range(0, len(order), config.batch_size):
            selected = order[start:start + config.batch_size]
            loss = _batch_loss(model, batch.select(selected), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({
            "epoch": epoch + 1,
            "loss": sum(losses) / max(len(losses), 1),
        })
    model.eval()
    return model, {
        "schema": "v4-retrieval-training-report-v1",
        "architecture": config.architecture,
        "rows": len(batch.request_id),
        "negative_width": batch.negative_item_id.shape[1],
        "event_watermark": batch.event_watermark,
        "partition_content_hashes": list(batch.partition_content_hashes),
        "feature_manifest_hash": batch.feature_manifest_hash,
        "retrieval_feature_contract_hash": (
            DEFAULT_RETRIEVAL_FEATURE_CONTRACT.manifest_hash
        ),
        "corpus_sha256": corpus.content_sha256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": config.epochs,
        "history": history,
        "seconds": perf_counter() - started,
        "device": str(target),
        "seed": config.seed,
        "sampled_softmax_expected_count_correction": True,
    }
