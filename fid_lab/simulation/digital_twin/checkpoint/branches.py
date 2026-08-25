"""Durable branch heads over immutable world checkpoints.

Only ``main`` is factual and may feed later training. Diagnostic branches can
advance from an earlier factual checkpoint, but they can never become training
authority or overwrite factual history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from filelock import FileLock

from .store import WorldCheckpointStore


WORLD_BRANCH_REGISTRY_SCHEMA = "world-branch-registry/v1"
DIAGNOSTIC_BRANCH_KINDS = frozenset({"shadow", "replay", "counterfactual"})
_BRANCH_NAME = re.compile(r"^[a-z][a-z0-9._/-]{0,127}$")


@dataclass(frozen=True)
class WorldBranchRef:
    name: str
    kind: str
    status: str
    base_checkpoint_id: str
    head_checkpoint_id: str
    source_branch: str
    created_at_logical_time: int
    purpose: str
    training_authority: bool
    status_reason: str = ""


class WorldBranchRegistry:
    """Compare-and-swap control plane for world checkpoint branches."""

    def __init__(self, store: WorldCheckpointStore):
        self.store = store
        self.path = store.root / "world-branches.json"
        self.lock_path = store.root / "world-branches.lock"

    def initialize_main(self, checkpoint_id: str) -> WorldBranchRef:
        checkpoint = self.store.get_ref(checkpoint_id)
        with self._locked():
            state = self._read()
            existing = state["branches"].get("main")
            if existing is not None:
                branch = WorldBranchRef(**existing)
                if branch.head_checkpoint_id != checkpoint_id:
                    raise ValueError("main already points at another checkpoint")
                return branch
            branch = WorldBranchRef(
                name="main",
                kind="factual",
                status="active",
                base_checkpoint_id=checkpoint_id,
                head_checkpoint_id=checkpoint_id,
                source_branch="",
                created_at_logical_time=checkpoint.logical_time,
                purpose="single factual ecosystem history",
                training_authority=True,
            )
            state["branches"][branch.name] = asdict(branch)
            self._commit(state)
            return branch

    def get(self, name: str) -> WorldBranchRef:
        self._validate_name(name)
        with self._locked():
            value = self._read()["branches"].get(name)
        if value is None:
            raise KeyError(f"unknown world branch: {name}")
        return WorldBranchRef(**value)

    def list(self) -> tuple[WorldBranchRef, ...]:
        with self._locked():
            branches = self._read()["branches"]
        return tuple(
            WorldBranchRef(**branches[name]) for name in sorted(branches)
        )

    def fork(
        self,
        source_branch: str,
        new_branch: str,
        *,
        kind: str,
        purpose: str,
        from_checkpoint_id: str | None = None,
    ) -> WorldBranchRef:
        self._validate_name(source_branch)
        self._validate_name(new_branch)
        if kind not in DIAGNOSTIC_BRANCH_KINDS:
            raise ValueError("fork kind must be shadow, replay or counterfactual")
        if not purpose.strip():
            raise ValueError("diagnostic branch requires a purpose")
        with self._locked():
            state = self._read()
            if new_branch in state["branches"]:
                raise ValueError(f"world branch already exists: {new_branch}")
            source_value = state["branches"].get(source_branch)
            if source_value is None:
                raise KeyError(f"unknown source branch: {source_branch}")
            source = WorldBranchRef(**source_value)
            checkpoint_id = from_checkpoint_id or source.head_checkpoint_id
            if not self.store.is_ancestor(
                checkpoint_id, source.head_checkpoint_id,
            ):
                raise ValueError("fork checkpoint is not on the source branch")
            checkpoint = self.store.get_ref(checkpoint_id)
            branch = WorldBranchRef(
                name=new_branch,
                kind=kind,
                status="active",
                base_checkpoint_id=checkpoint_id,
                head_checkpoint_id=checkpoint_id,
                source_branch=source_branch,
                created_at_logical_time=checkpoint.logical_time,
                purpose=purpose.strip(),
                training_authority=False,
            )
            state["branches"][branch.name] = asdict(branch)
            self._commit(state)
            return branch

    def advance(
        self,
        name: str,
        checkpoint_id: str,
        *,
        expected_head_checkpoint_id: str,
    ) -> WorldBranchRef:
        self._validate_name(name)
        checkpoint = self.store.get_ref(checkpoint_id)
        with self._locked():
            state = self._read()
            value = state["branches"].get(name)
            if value is None:
                raise KeyError(f"unknown world branch: {name}")
            current = WorldBranchRef(**value)
            if current.status != "active":
                raise ValueError(f"cannot advance {current.status} branch")
            if current.head_checkpoint_id != expected_head_checkpoint_id:
                raise ValueError("world branch head changed concurrently")
            if checkpoint.parent_checkpoint_id != current.head_checkpoint_id:
                raise ValueError("new checkpoint is not a direct branch child")
            updated = WorldBranchRef(
                **{
                    **asdict(current),
                    "head_checkpoint_id": checkpoint_id,
                },
            )
            state["branches"][name] = asdict(updated)
            self._commit(state)
            return updated

    def close(self, name: str, *, reason: str) -> WorldBranchRef:
        return self._set_status(name, "closed", reason)

    def invalidate(self, name: str, *, reason: str) -> WorldBranchRef:
        if name == "main":
            raise ValueError("main cannot be invalidated; create a corrected descendant")
        return self._set_status(name, "invalid", reason)

    def _set_status(
        self, name: str, status: str, reason: str,
    ) -> WorldBranchRef:
        self._validate_name(name)
        if not reason.strip():
            raise ValueError("branch status change requires a reason")
        with self._locked():
            state = self._read()
            value = state["branches"].get(name)
            if value is None:
                raise KeyError(f"unknown world branch: {name}")
            current = WorldBranchRef(**value)
            updated = WorldBranchRef(**{
                **asdict(current),
                "status": status,
                "status_reason": reason.strip(),
            })
            state["branches"][name] = asdict(updated)
            self._commit(state)
            return updated

    def _locked(self) -> FileLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.lock_path))

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "schema": WORLD_BRANCH_REGISTRY_SCHEMA,
                "revision": 0,
                "branches": {},
            }
        state = json.loads(self.path.read_text())
        if state.get("schema") != WORLD_BRANCH_REGISTRY_SCHEMA:
            raise ValueError("world branch registry schema is unsupported")
        if not isinstance(state.get("branches"), dict):
            raise ValueError("world branch registry is corrupted")
        return state

    def _commit(self, state: dict[str, object]) -> None:
        state["revision"] = int(state["revision"]) + 1
        payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
        with NamedTemporaryFile(
            mode="w", dir=self.path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, self.path)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _BRANCH_NAME.fullmatch(name):
            raise ValueError(f"invalid world branch name: {name!r}")
