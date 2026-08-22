"""Versioned frame/text fusion outside the online ranking request."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class MediaAsset:
    video_id: int
    frame_embeddings: np.ndarray
    text_embedding: np.ndarray
    source_timestamp: int


@dataclass(frozen=True)
class MediaFeature:
    video_id: int
    content_embedding: np.ndarray
    frame_attention: np.ndarray
    source_timestamp: int
    encoder_version: str
    content_hash: str


class MediaFeatureMaterializer:
    """Small deterministic adapter over upstream frame and text embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        version: str = "media-encoder-v1",
        seed: int = 101,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.version = version
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(
            0.0, 1.0 / input_dim**0.5, size=(input_dim * 2, output_dim)
        ).astype(np.float32)

    def materialize(self, asset: MediaAsset) -> MediaFeature:
        if asset.frame_embeddings.ndim != 2:
            raise ValueError("frame embeddings must have shape [frames, dimension]")
        if asset.frame_embeddings.shape[1] != self.input_dim:
            raise ValueError("frame embedding dimension does not match encoder")
        if asset.text_embedding.shape != (self.input_dim,):
            raise ValueError("text embedding dimension does not match encoder")
        logits = asset.frame_embeddings @ asset.text_embedding
        attention = np.exp(logits - logits.max())
        attention /= attention.sum()
        video = (attention[:, None] * asset.frame_embeddings).sum(axis=0)
        fused = np.concatenate([video, asset.text_embedding]) @ self.projection
        fused /= max(float(np.linalg.norm(fused)), 1e-8)
        payload = asset.frame_embeddings.tobytes() + asset.text_embedding.tobytes()
        return MediaFeature(
            video_id=asset.video_id,
            content_embedding=fused.astype(np.float32),
            frame_attention=attention.astype(np.float32),
            source_timestamp=asset.source_timestamp,
            encoder_version=self.version,
            content_hash=hashlib.sha256(payload).hexdigest(),
        )
