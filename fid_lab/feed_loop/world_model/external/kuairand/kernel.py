"""Request/slate adapter for the externally validated behavior kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .models import KuaiSequenceTransformer, KuaiWideDeep


@dataclass(frozen=True)
class SlateResponse:
    probabilities: torch.Tensor
    stay_norm: torch.Tensor


class KuaiBehaviorKernel:
    def __init__(self, model, device) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @classmethod
    def load(cls, artifact: Path, device="cuda:0"):
        payload = torch.load(artifact, map_location=device, weights_only=False)
        manifest = payload["dataset_manifest"]
        vocabularies = tuple(manifest["sparse_vocabularies"])
        dense_dim = len(manifest["dense_names"])
        if payload["model_name"] == "sequence_transformer":
            model = KuaiSequenceTransformer(
                vocabularies, dense_dim, manifest["sequence_length"]
            )
        elif payload["model_name"] == "wide_deep":
            model = KuaiWideDeep(vocabularies, dense_dim)
        else:
            raise ValueError(f"unsupported external kernel: {payload['model_name']}")
        model.load_state_dict(payload["state_dict"])
        return cls(model, device)

    @torch.inference_mode()
    def score_slate(self, request_sparse, request_dense, candidate_sparse,
                    candidate_dense, history_items, history_feedback):
        batch, candidates, fields = candidate_sparse.shape
        sparse = candidate_sparse.clone()
        sparse[:, :, 0] = request_sparse[:, None, 0]
        dense = candidate_dense.clone()
        dense[:, :, 1:3] = request_dense[:, None, 1:3]
        dense[:, :, 4:] = request_dense[:, None, 4:]
        logits = self.model(
            sparse.reshape(-1, fields).to(self.device),
            dense.reshape(-1, dense.shape[-1]).to(self.device),
            history_items[:, None].expand(-1, candidates, -1).reshape(
                -1, history_items.shape[1]
            ).to(self.device),
            history_feedback[:, None].expand(-1, candidates, -1, -1).reshape(
                -1, history_feedback.shape[1], history_feedback.shape[2]
            ).to(self.device),
        ).reshape(batch, candidates, -1)
        return SlateResponse(
            probabilities=torch.sigmoid(logits[:, :, :7]),
            stay_norm=torch.sigmoid(logits[:, :, 7]),
        )

    def sample_selected(self, response: SlateResponse, choice, seed):
        choice = choice.to(self.device)
        rows = torch.arange(len(choice), device=self.device)
        probability = response.probabilities[rows, choice]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        actions = torch.rand(
            probability.shape, generator=generator, device=self.device
        ) < probability
        stay = response.stay_norm[rows, choice]
        return actions, stay

    @staticmethod
    def advance_history(history_items, history_feedback, selected_items, actions):
        next_items = torch.roll(history_items, shifts=-1, dims=1)
        next_feedback = torch.roll(history_feedback, shifts=-1, dims=1)
        next_items[:, -1] = selected_items
        next_feedback[:, -1] = actions.to(next_feedback.dtype)
        return next_items, next_feedback
