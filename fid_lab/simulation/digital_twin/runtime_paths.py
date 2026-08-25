"""Persistent paths owned by one continuous factual simulation profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .profile import SimulationProfile


RUNTIME_AUTHORITY_SCHEMA = "digital-twin-runtime-authority/v1"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def request_stream(self) -> Path:
        return self.root / "request-stream"

    @property
    def event_partitions(self) -> Path:
        return self.root / "event-partitions"

    @property
    def sample_dataset(self) -> Path:
        return self.root / "sample-dataset"

    @property
    def model_registry(self) -> Path:
        return self.root / "model-registry"

    @property
    def launch_journal(self) -> Path:
        return self.root / "launch-journal"

    @property
    def authority_file(self) -> Path:
        return self.root / "runtime-authority.json"

    @classmethod
    def standard(
        cls,
        profile: SimulationProfile,
        root: Path | None = None,
    ) -> RuntimePaths:
        base = root
        if base is None:
            configured = os.environ.get("FID_LAB_RUNTIME_ROOT")
            base = (
                Path(configured).expanduser()
                if configured
                else Path.home() / ".local" / "share" / "recsys-fid-system"
            )
        return cls(base / profile.name)

    def initialize(self, profile: SimulationProfile) -> dict[str, object]:
        manifest = {
            "schema": RUNTIME_AUTHORITY_SCHEMA,
            "profile": profile.manifest(),
            "profile_hash": profile.profile_hash,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.data_paths():
            path.mkdir(parents=True, exist_ok=True)
        if self.authority_file.exists():
            current = json.loads(self.authority_file.read_text())
            if current != manifest:
                raise ValueError("runtime root belongs to another simulation profile")
            return current
        with NamedTemporaryFile(
            mode="w", dir=self.root, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, self.authority_file)
        return manifest

    def data_paths(self) -> tuple[Path, ...]:
        return (
            self.checkpoints,
            self.request_stream,
            self.event_partitions,
            self.sample_dataset,
            self.model_registry,
            self.launch_journal,
        )

    def manifest(self) -> dict[str, object]:
        if not self.authority_file.is_file():
            raise ValueError("runtime authority has not been initialized")
        value = json.loads(self.authority_file.read_text())
        if value.get("schema") != RUNTIME_AUTHORITY_SCHEMA:
            raise ValueError("runtime authority schema is unsupported")
        return value
