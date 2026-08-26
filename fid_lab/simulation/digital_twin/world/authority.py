"""One replaceable response authority behind the private world boundary."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Protocol

import torch

from ....feed_loop.world_model.ensemble import WorldModelEnsemble
from ....feed_loop.world_model.training import load_world_ensemble
from ....feed_loop.world_model.validation.support import (
    SUPPORT_PROFILE_SCHEMA,
    request_support_mask,
)
from ...randomness.counter import uniform_for_items
from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    EventType,
    RenderedSlateBatch,
    Surface,
)
from .behavior import (
    ResponseTensors,
    materialize_response_events,
    response_events,
)
from .experience import EXPERIENCE_DYNAMICS_VERSION
from .neural_features import (
    NEURAL_FEATURE_VERSION,
    V4_FEATURE_CONTRACT,
    V4_FEATURE_COVERAGE,
    build_neural_scm_batch,
    request_keyed_structural_noise,
)
from .state import UserWorldSnapshot

DEFAULT_NEURAL_INFERENCE_BATCH_SIZE = 4_096


class ResponseAuthority(Protocol):
    version: str

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch: ...

    def manifest(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class FactualResponseArtifact:
    """Content identity required to load one factual Feed response model."""

    artifact_dir: str
    manifest_sha256: str
    member_index: int
    inference_batch_size: int = DEFAULT_NEURAL_INFERENCE_BATCH_SIZE
    maximum_support_fallback_rate: float = 0.03

    def __post_init__(self) -> None:
        if not self.artifact_dir:
            raise ValueError("factual response artifact directory is required")
        if len(self.manifest_sha256) != 64:
            raise ValueError("factual response manifest requires a SHA-256")
        if self.member_index < 0:
            raise ValueError("factual response member index cannot be negative")
        if self.inference_batch_size <= 0:
            raise ValueError("factual response batch size must be positive")
        if not 0.0 <= self.maximum_support_fallback_rate < 1.0:
            raise ValueError("support fallback rate must be in [0, 1)")


class FormulaResponseAuthority:
    """Deterministic invariant oracle; not the accepted P2 behavioral world."""

    version = "formula-invariant-oracle-v1"

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        return response_events(snapshot, catalog, slate, seed)

    def manifest(self) -> dict[str, object]:
        return {"kind": "formula_oracle", "version": self.version}


class BehavioralSCMResponseAuthority:
    """Nonlinear hidden-state response authority for the factual user world."""

    version = f"behavioral-scm-v2:{EXPERIENCE_DYNAMICS_VERSION}"

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        return response_events(snapshot, catalog, slate, seed)

    def manifest(self) -> dict[str, object]:
        return {"kind": "legacy_formula_alias", "version": self.version}


ACTION_EVENT_MAP = {
    "play": EventType.PLAY,
    "play_3s": EventType.PLAY_3S,
    "long_view": EventType.LONG_VIEW,
    "complete_play": EventType.COMPLETE,
    "like": EventType.LIKE,
    "comment": EventType.COMMENT,
    "share": EventType.SHARE,
    "follow": EventType.FOLLOW,
    "negative_feedback": EventType.NEGATIVE,
    "poi_detail": EventType.DETAIL,
    "poi_favorite": EventType.FAVORITE,
}
class NeuralFeedResponseAuthority:
    """Use one ensemble member as a plausible Feed world; delegate other surfaces."""

    def __init__(
        self,
        ensemble: WorldModelEnsemble,
        *,
        member_index: int,
        artifact_sha256: str,
        feature_contract_sha256: str,
        feature_coverage: dict[str, str],
        support_profile: dict,
        inference_batch_size: int = DEFAULT_NEURAL_INFERENCE_BATCH_SIZE,
        weights_sha256: str = "",
        maximum_support_fallback_rate: float = 0.03,
    ) -> None:
        if not 0 <= member_index < len(ensemble.members):
            raise ValueError("world-model member index is out of range")
        if len(artifact_sha256) != 64:
            raise ValueError("world-model authority requires a SHA-256 artifact")
        if feature_contract_sha256 != V4_FEATURE_CONTRACT["sha256"]:
            raise ValueError(
                "world-model feature contract does not match v4 serving semantics"
            )
        missing = [
            index for index, status in V4_FEATURE_COVERAGE.items()
            if status == "native_v4"
            and feature_coverage.get(index) not in {"native_v4", "multi_source"}
        ]
        if missing:
            raise ValueError(
                f"world-model feature coverage is incomplete at indices {missing}"
            )
        if support_profile.get("schema") != SUPPORT_PROFILE_SCHEMA:
            raise ValueError("world-model authority requires a support profile")
        if inference_batch_size <= 0:
            raise ValueError("neural inference batch size must be positive")
        if weights_sha256 and len(weights_sha256) != 64:
            raise ValueError("world-model weights require a SHA-256")
        if not 0.0 <= maximum_support_fallback_rate < 1.0:
            raise ValueError("support fallback rate must be in [0, 1)")
        self.ensemble = ensemble
        self.member_index = member_index
        self.artifact_sha256 = artifact_sha256
        self.feature_contract_sha256 = feature_contract_sha256
        self.feature_coverage = dict(feature_coverage)
        self.support_profile = dict(support_profile)
        self.inference_batch_size = inference_batch_size
        self.weights_sha256 = weights_sha256
        self.maximum_support_fallback_rate = maximum_support_fallback_rate
        self.supported_feed_requests = 0
        self.support_fallback_requests = 0
        self.non_feed_requests = 0
        self.formula = FormulaResponseAuthority()
        self.version = (
            f"neural-feed:{NEURAL_FEATURE_VERSION}:member-{member_index}:"
            f"{artifact_sha256}"
        )
        self.ensemble.members[member_index].eval()

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "neural_feed",
            "version": self.version,
            "artifact_manifest_sha256": self.artifact_sha256,
            "weights_sha256": self.weights_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "member_index": self.member_index,
            "maximum_support_fallback_rate": self.maximum_support_fallback_rate,
        }

    def stats(self) -> dict[str, object]:
        total_feed = self.supported_feed_requests + self.support_fallback_requests
        return {
            "supported_feed_requests": self.supported_feed_requests,
            "support_fallback_requests": self.support_fallback_requests,
            "support_fallback_rate": (
                self.support_fallback_requests / max(total_feed, 1)
            ),
            "non_feed_requests": self.non_feed_requests,
        }

    def restore_stats(self, value: dict[str, object]) -> None:
        self.supported_feed_requests = int(value["supported_feed_requests"])
        self.support_fallback_requests = int(value["support_fallback_requests"])
        self.non_feed_requests = int(value["non_feed_requests"])

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        feed = slate.surface == int(Surface.FEED)
        batches = []
        if feed.any():
            batches.append(self._feed_response(
                snapshot, catalog, slate.select(feed), seed,
            ))
        if (~feed).any():
            self.non_feed_requests += int((~feed).sum())
            batches.append(self.formula.respond(
                snapshot, catalog, slate.select(~feed), seed,
            ))
        return AppEventBatch.concatenate(tuple(batches))

    def _feed_response(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        batches = []
        for start in range(0, len(slate.request_id), self.inference_batch_size):
            stop = min(start + self.inference_batch_size, len(slate.request_id))
            batches.append(self._feed_response_batch(
                snapshot, catalog, slate.select(slice(start, stop)), seed,
            ))
        return AppEventBatch.concatenate(tuple(batches))

    def _feed_response_batch(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        batch = build_neural_scm_batch(snapshot, catalog, slate)
        supported = request_support_mask(
            batch["slate_features"], self.support_profile,
        )
        supported_count = int(supported.sum())
        fallback_count = int((~supported).sum())
        next_supported = self.supported_feed_requests + supported_count
        next_fallback = self.support_fallback_requests + fallback_count
        fallback_rate = next_fallback / max(next_supported + next_fallback, 1)
        if fallback_count and fallback_rate > self.maximum_support_fallback_rate:
            raise RuntimeError(
                "NeuralSCM support fallback exceeds the factual authority budget"
            )
        self.supported_feed_requests = next_supported
        self.support_fallback_requests = next_fallback
        batches = []
        if supported.any():
            batches.append(self._supported_feed_response(
                snapshot, catalog, slate.select(supported), seed,
            ))
        if (~supported).any():
            batches.append(self.formula.respond(
                snapshot, catalog, slate.select(~supported), seed,
            ))
        return AppEventBatch.concatenate(tuple(batches))

    def _supported_feed_response(
        self,
        snapshot: UserWorldSnapshot,
        catalog: PublicCatalog,
        slate: RenderedSlateBatch,
        seed: int,
    ) -> AppEventBatch:
        batch = build_neural_scm_batch(snapshot, catalog, slate)
        noise = request_keyed_structural_noise(
            slate, self.ensemble.config, seed,
        )
        with torch.inference_mode():
            sampled = self.ensemble.members[self.member_index].sample_slate(
                batch, noise.latent, noise.mixture, noise.stay, noise.actions,
            )
        play_probability = sampled["probabilities"]["play"]
        examination_probability = torch.sigmoid(
            2.1
            - 0.52 * slate.positions.float()
            + 0.55 * play_probability
            + 0.35 * snapshot.users.satisfaction[slate.user_id, None]
            - 0.45 * snapshot.users.fatigue[slate.user_id, None]
        )
        examine_draw = uniform_for_items(
            slate.request_id,
            slate.item_ids.clamp_min(0) * 37 + slate.positions.clamp_min(0),
            0,
            1_741,
            seed,
        )
        examined = slate.valid & (examine_draw < examination_probability)
        examined[:, 0] |= slate.valid[:, 0]
        actions = {
            event_type: sampled["actions"][name] & examined
            for name, event_type in ACTION_EVENT_MAP.items()
        }
        actions[EventType.CLICK] = (
            sampled["actions"]["content_click"]
            | sampled["actions"]["anchor_click"]
        ) & examined
        actions[EventType.SLIDE] = examined & ~actions[EventType.COMPLETE]
        actions[EventType.CREATE] = torch.zeros_like(examined)
        actions[EventType.PUBLISH] = torch.zeros_like(examined)
        actions[EventType.SEARCH_SUCCESS] = torch.zeros_like(examined)
        dwell_ms = (
            sampled["stay_seconds"] * 1_000.0
        ).round().long().clamp_min(0)
        response = ResponseTensors(
            examined=examined,
            affinity=torch.zeros_like(dwell_ms, dtype=torch.float),
            utility=torch.zeros_like(dwell_ms, dtype=torch.float),
            dwell_ms=dwell_ms,
            action=actions,
            session_end=sampled["actions"]["session_exit"][:, 0],
        )
        return materialize_response_events(
            response, snapshot, catalog, slate, seed,
        )


def load_factual_response_authority(
    artifact: FactualResponseArtifact,
    device: str | torch.device,
) -> NeuralFeedResponseAuthority:
    """Load exactly one promoted artifact; missing or changed bytes fail closed."""
    root = Path(artifact.artifact_dir)
    manifest_path = root / "manifest.json"
    weights_path = root / "world_model.pt"
    if not manifest_path.is_file():
        raise ValueError("factual response manifest is missing")
    actual_manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    if actual_manifest_hash != artifact.manifest_sha256:
        raise ValueError("factual response manifest hash differs from authority")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("authority_status") != "accepted_feed_authority":
        raise ValueError("factual response artifact has not been promoted")
    if not weights_path.is_file():
        raise ValueError("factual response weights are missing")
    actual_weights_hash = sha256(weights_path.read_bytes()).hexdigest()
    if actual_weights_hash != manifest.get("weights_sha256"):
        raise ValueError("factual response weights hash differs from manifest")
    ensemble = load_world_ensemble(root, str(device))
    return NeuralFeedResponseAuthority(
        ensemble,
        member_index=artifact.member_index,
        artifact_sha256=actual_manifest_hash,
        feature_contract_sha256=manifest["feature_contract_sha256"],
        feature_coverage=manifest["feature_coverage"],
        support_profile=manifest["support_profile"],
        inference_batch_size=artifact.inference_batch_size,
        weights_sha256=actual_weights_hash,
        maximum_support_fallback_rate=artifact.maximum_support_fallback_rate,
    )
