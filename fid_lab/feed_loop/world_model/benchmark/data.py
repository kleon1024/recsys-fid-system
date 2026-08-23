"""V4 targets and model views derived from one request-level authority."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ....evolution.models.deepctr_adapter import build_feature_bundle
from ...models.deep_policy import DENSE_INDICES, SPARSE_SPECS
from ..contracts import BINARY_ACTIONS
from ..data import WorldModelSplit


LONG_VIEW_COLUMN = next(
    index for index, action in enumerate(BINARY_ACTIONS)
    if action.name == "long_view"
)


@dataclass(frozen=True)
class V4Target:
    labels: np.ndarray
    oracle_probability: np.ndarray


def materialize_target(ensemble, split: WorldModelSplit, device, seed: int) -> V4Target:
    probabilities = []
    for start in range(0, len(split), 4_096):
        indices = torch.arange(start, min(start + 4_096, len(split)))
        prediction = ensemble.predict(split.batch(indices, device))
        probabilities.append(
            prediction["probability_mean"][:, LONG_VIEW_COLUMN].cpu()
        )
    oracle = torch.cat(probabilities).numpy().astype(np.float32)
    labels = (
        np.random.default_rng(seed).random(len(oracle)) < oracle
    ).astype(np.float32)
    return V4Target(labels, oracle)


def deepctr_bundle(features: np.ndarray):
    sparse_values = []
    bucket_sizes = []
    for _, index, vocabulary in SPARSE_SPECS:
        sparse_values.append(
            np.rint(features[:, index] * (vocabulary - 1)).astype(np.int64)
        )
        bucket_sizes.append(vocabulary)
    sparse = np.stack(sparse_values, axis=1)
    dense = features[:, DENSE_INDICES].astype(np.float32)
    return build_feature_bundle(sparse, dense, bucket_sizes=tuple(bucket_sizes))


@torch.inference_mode()
def candidate_oracle(ensemble, split, device, rows: int, request_batch: int = 128):
    count = min(rows, len(split))
    candidates = split.slate_features.shape[1]
    output = []
    for start in range(0, count, request_batch):
        stop = min(start + request_batch, count)
        indices = torch.arange(start, stop)
        slate = split.slate_features[indices].to(device)
        sequence = split.sequence[indices].to(device)
        lifecycle = split.lifecycle[indices].to(device)
        region = split.region[indices].to(device)
        repeated = {
            "selected_features": slate.reshape(-1, slate.shape[-1]),
            "slate_features": slate[:, None].expand(
                -1, candidates, -1, -1
            ).reshape(-1, candidates, slate.shape[-1]),
            "sequence": sequence[:, None].expand(
                -1, candidates, -1, -1
            ).reshape(-1, sequence.shape[1], sequence.shape[2]),
            "lifecycle": lifecycle[:, None].expand(-1, candidates).reshape(-1),
            "region": region[:, None].expand(-1, candidates).reshape(-1),
            "labels": torch.zeros(
                (stop - start) * candidates, 16, device=device
            ),
        }
        probability = ensemble.predict(repeated)["probability_mean"]
        output.append(
            probability[:, LONG_VIEW_COLUMN].reshape(stop - start, candidates).cpu()
        )
    return torch.cat(output).numpy().astype(np.float32)


def information_ceiling(labels: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    return {
        "auc": float(roc_auc_score(labels, oracle)),
        "log_loss": float(log_loss(labels, oracle)),
        "entropy_bits": float(
            np.mean(-oracle * np.log2(oracle) - (1.0 - oracle) * np.log2(1.0 - oracle))
        ),
        "random_auc_standard_error": float(
            math.sqrt(1.0 / max(len(labels), 1))
        ),
    }


@torch.inference_mode()
def context_ablation(ensemble, split, device, rows=50_000, seed=20260823):
    count = min(rows, len(split))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    values = {"baseline": [], "permuted_sequence": [], "selected_only_slate": []}
    for start in range(0, count, 2_048):
        indices = torch.arange(start, min(start + 2_048, count))
        batch = split.batch(indices, device)
        sequence_indices = permutation[indices]
        sequence_batch = {
            **batch,
            "sequence": split.sequence[sequence_indices].to(device),
        }
        selected_batch = {
            **batch,
            "slate_features": batch["selected_features"][:, None].expand_as(
                batch["slate_features"]
            ),
        }
        for name, candidate_batch in (
            ("baseline", batch),
            ("permuted_sequence", sequence_batch),
            ("selected_only_slate", selected_batch),
        ):
            probability = ensemble.predict(candidate_batch)["probability_mean"]
            values[name].append(probability[:, LONG_VIEW_COLUMN].cpu())
    baseline = torch.cat(values["baseline"])
    report = {"rows": count, "baseline_probability_std": float(baseline.std())}
    for name in ("permuted_sequence", "selected_only_slate"):
        delta = (torch.cat(values[name]) - baseline).abs()
        report[name] = {
            "mean_absolute_probability_delta": float(delta.mean()),
            "p95_absolute_probability_delta": float(delta.quantile(0.95)),
            "relative_to_baseline_std": float(
                delta.mean() / baseline.std().clamp_min(1e-8)
            ),
        }
    return report
