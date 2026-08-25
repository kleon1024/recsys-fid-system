"""Content-bound retrieval checkpoint and FAISS serving index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from time import perf_counter

import numpy as np
import torch

from ...contracts import PlatformRequestBatch, Surface
from ...platform.projection import PlatformProjectionState
from ...platform.route_contracts import surface_eligibility
from ..contracts import ArtifactCompatibility
from .contracts import (
    RetrievalCorpus,
    RetrievalModelConfig,
    tensor_content_hash,
)
from .model import ObservableRetrievalModel


@dataclass(frozen=True)
class RetrievalArtifact:
    model: ObservableRetrievalModel
    config: RetrievalModelConfig
    feature_manifest_hash: str
    retrieval_feature_contract_hash: str
    corpus_sha256: str
    training_report: dict[str, object]

    @property
    def model_name(self) -> str:
        return f"v4-{self.config.architecture}-retriever"

    @property
    def state_sha256(self) -> str:
        return tensor_content_hash({
            name: value for name, value in self.model.state_dict().items()
        })

    @property
    def index_version(self) -> str:
        payload = json.dumps({
            "schema": "v4-retrieval-index-v1",
            "architecture": self.config.architecture,
            "config_hash": self.config.config_hash,
            "corpus_sha256": self.corpus_sha256,
            "state_sha256": self.state_sha256,
            "ann_policy": "faiss-ivf-flat-v1",
        }, sort_keys=True, separators=(",", ":"))
        return f"retrieval-{sha256(payload.encode('utf-8')).hexdigest()[:24]}"

    def validate_compatibility(self, expected: ArtifactCompatibility) -> None:
        if self.feature_manifest_hash != expected.feature_manifest_hash:
            raise ValueError("retrieval artifact feature manifest differs")
        if self.retrieval_feature_contract_hash != expected.stage_contract_hash:
            raise ValueError("retrieval artifact stage feature contract differs")
        if self.corpus_sha256 != expected.corpus_sha256:
            raise ValueError("retrieval artifact corpus differs")
        if self.index_version != expected.index_version:
            raise ValueError("retrieval artifact index version differs")

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema": "v4-retrieval-artifact-v1",
            "config": asdict(self.config),
            "feature_manifest_hash": self.feature_manifest_hash,
            "retrieval_feature_contract_hash": self.retrieval_feature_contract_hash,
            "corpus_sha256": self.corpus_sha256,
            "state_sha256": self.state_sha256,
            "index_version": self.index_version,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "training_report": self.training_report,
        }

    @classmethod
    def from_checkpoint(
        cls,
        value: dict[str, object],
        corpus: RetrievalCorpus,
    ) -> RetrievalArtifact:
        if value.get("schema") != "v4-retrieval-artifact-v1":
            raise ValueError("retrieval checkpoint schema is unsupported")
        if str(value["corpus_sha256"]) != corpus.content_sha256:
            raise ValueError("retrieval checkpoint corpus differs")
        config = RetrievalModelConfig(**value["config"])
        model = ObservableRetrievalModel(corpus, config)
        model.load_state_dict(value["state_dict"])
        model.eval()
        result = cls(
            model=model,
            config=config,
            feature_manifest_hash=str(value["feature_manifest_hash"]),
            retrieval_feature_contract_hash=str(
                value["retrieval_feature_contract_hash"]
            ),
            corpus_sha256=str(value["corpus_sha256"]),
            training_report=dict(value["training_report"]),
        )
        if result.state_sha256 != value["state_sha256"]:
            raise ValueError("retrieval checkpoint state hash differs")
        if result.index_version != value["index_version"]:
            raise ValueError("retrieval checkpoint index version differs")
        return result


class RetrievalANNIndex:
    def __init__(
        self,
        artifact: RetrievalArtifact,
        corpus: RetrievalCorpus,
        *,
        device: str | torch.device,
        item_batch_size: int = 65_536,
    ) -> None:
        import faiss

        if artifact.corpus_sha256 != corpus.content_sha256:
            raise ValueError("retrieval index corpus differs from model")
        self.artifact = artifact
        self.corpus = corpus
        self.device = torch.device(device)
        self.backend = "faiss-flat"
        self.nlist = 1
        self.nprobe = 1
        self.artifact.model.to(self.device).eval()
        eligible = corpus.active & surface_eligibility(
            int(Surface.FEED), corpus.content_kind,
        )
        self.item_ids = corpus.item_id[eligible]
        if not len(self.item_ids):
            raise ValueError("retrieval index has no eligible Feed items")
        started = perf_counter()
        states = []
        with torch.inference_mode():
            for start in range(0, len(self.item_ids), item_batch_size):
                item = self.item_ids[start:start + item_batch_size].to(self.device)
                states.append(self.artifact.model.encode_items(item).cpu())
        self.item_embeddings = torch.cat(states).numpy().astype(np.float32)
        dimension = self.item_embeddings.shape[1]
        faiss.omp_set_num_threads(int(os.environ.get("FID_RETRIEVAL_FAISS_THREADS", "8")))
        if len(self.item_ids) < 10_000:
            base = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap2(base)
        else:
            self.backend = "faiss-ivf-flat"
            self.nlist = min(2_048, max(64, int(np.sqrt(len(self.item_ids)))))
            self.nprobe = min(32, self.nlist)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT,
            )
            self.index.cp.seed = 2_026_082_5
            self.index.cp.niter = 15
            training_rows = min(
                len(self.item_embeddings), max(20_000, self.nlist * 40),
            )
            location = np.linspace(
                0, len(self.item_embeddings) - 1, training_rows, dtype=np.int64,
            )
            self.index.train(self.item_embeddings[location])
            self.index.nprobe = self.nprobe
        self.index.add_with_ids(
            self.item_embeddings,
            self.item_ids.numpy().astype(np.int64),
        )
        self.build_seconds = perf_counter() - started

    def search(self, query: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if top_k <= 0:
            raise ValueError("retrieval search Top-K must be positive")
        interests = query.shape[1] if query.ndim == 3 else 1
        flat = query.reshape(-1, query.shape[-1]).detach().cpu().numpy().astype(np.float32)
        scores, items = self.index.search(flat, min(top_k, len(self.item_ids)))
        if interests == 1:
            return torch.from_numpy(items), torch.from_numpy(scores)
        items = items.reshape(-1, interests, items.shape[1])
        scores = scores.reshape(-1, interests, scores.shape[1])
        merged_item = np.full((len(items), top_k), -1, dtype=np.int64)
        merged_score = np.full((len(items), top_k), -np.inf, dtype=np.float32)
        for row in range(len(items)):
            best: dict[int, float] = {}
            for item, score in zip(items[row].reshape(-1), scores[row].reshape(-1)):
                if item >= 0:
                    best[int(item)] = max(best.get(int(item), -np.inf), float(score))
            ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
            for column, (item, score) in enumerate(ranked):
                merged_item[row, column] = item
                merged_score[row, column] = score
        return torch.from_numpy(merged_item), torch.from_numpy(merged_score)


class LearnedRetrievalAdapter:
    def __init__(
        self,
        index: RetrievalANNIndex,
        serving_version_id: int,
    ) -> None:
        self.index = index
        self.serving_version_id = serving_version_id

    @property
    def index_version(self) -> str:
        return self.index.artifact.index_version

    @torch.inference_mode()
    def retrieve(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.index.artifact.model
        device = next(model.parameters()).device
        user = requests.user_id
        query = model.encode_query_tensors(
            user_id=user.to(device),
            surface=requests.surface.to(device),
            event_time=requests.event_time.to(device),
            query_topic=requests.query_topic.to(device),
            country=state.user_country[user].to(device),
            region=state.user_region[user].to(device),
            user_counts=state.user_event_counts[user].to(device),
            surface_counts=state.user_surface_counts[user].to(device),
            history_item_id=state.user_history_item[user].to(device),
        )
        item, score = self.index.search(query, top_k)
        return item.to(requests.user_id.device), score.to(requests.user_id.device)
