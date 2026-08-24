"""Versioned candidate, shadow, active, rejected model lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ranker import RankerArtifact


class ModelStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


@dataclass
class RegisteredModel:
    version: int
    stage: str
    artifact: RankerArtifact
    status: ModelStatus
    parent_version: int | None


class ModelRegistry:
    def __init__(self):
        self._next_version = 1
        self._models: dict[int, RegisteredModel] = {}
        self._active: dict[str, int] = {}

    def register(self, stage: str, artifact: RankerArtifact) -> RegisteredModel:
        if stage not in {"coarse", "fine"}:
            raise ValueError("learned rank model stage must be coarse or fine")
        version = self._next_version
        self._next_version += 1
        registered = RegisteredModel(
            version=version,
            stage=stage,
            artifact=artifact,
            status=ModelStatus.CANDIDATE,
            parent_version=self._active.get(stage),
        )
        self._models[version] = registered
        return registered

    def shadow(self, version: int) -> RegisteredModel:
        model = self._models[version]
        if model.status is not ModelStatus.CANDIDATE:
            raise ValueError("only a candidate model can enter shadow")
        model.status = ModelStatus.SHADOW
        return model

    def promote(self, version: int) -> RegisteredModel:
        model = self._models[version]
        if model.status not in {ModelStatus.CANDIDATE, ModelStatus.SHADOW}:
            raise ValueError("only candidate or shadow model can be promoted")
        previous = self._active.get(model.stage)
        if previous is not None:
            self._models[previous].status = ModelStatus.RETIRED
        model.status = ModelStatus.ACTIVE
        self._active[model.stage] = version
        return model

    def reject(self, version: int) -> RegisteredModel:
        model = self._models[version]
        if model.status is ModelStatus.ACTIVE:
            raise ValueError("active model requires replacement before retirement")
        model.status = ModelStatus.REJECTED
        return model

    def active(self, stage: str) -> RegisteredModel | None:
        version = self._active.get(stage)
        return self._models.get(version) if version is not None else None

    def manifest(self) -> dict[str, object]:
        return {
            "active": dict(self._active),
            "models": [
                {
                    "version": model.version,
                    "model_id": model.artifact.model_id,
                    "stage": model.stage,
                    "status": model.status.value,
                    "parent_version": model.parent_version,
                    "architecture": model.artifact.architecture,
                }
                for model in self._models.values()
            ],
        }
