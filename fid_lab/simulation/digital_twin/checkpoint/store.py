"""Content-addressed save, restore and fork boundary for the full twin state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping

import scipy.sparse
import torch

from fid_lab.launches.release_resources import file_sha256

from ..catalog import PublicCatalog
from ..contracts import AppEventBatch
from ..engine import AtomicSimulationKernel, ExperimentPlan
from ..experiments.layered import LayeredExperimentPlan, PolicyLayer
from ..platform.ranking import CascadePolicy
from ..platform.runtime import ReferenceRecommendationPlatform
from ..world.runtime import UserEcosystemWorld


WORLD_CHECKPOINT_SCHEMA = "digital-twin-world-checkpoint-v2"


@dataclass(frozen=True)
class WorldCheckpointRef:
    checkpoint_id: str
    logical_time: int
    parent_checkpoint_id: str
    state_object: str
    event_objects: tuple[str, ...]
    catalog_sha256: str
    runtime_sha256: str
    world_code_sha256: str
    platform_code_sha256: str
    experiment_sha256: str
    event_cursor_sha256: str = ""


@dataclass(frozen=True)
class RestoredWorldCheckpoint:
    ref: WorldCheckpointRef
    experiment: ExperimentPlan | LayeredExperimentPlan
    learning_cursors: dict[str, object]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _tensor_payload(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _dataclass_tensor_state(value: object) -> dict[str, torch.Tensor]:
    return {
        field.name: _tensor_payload(getattr(value, field.name))
        for field in fields(value)
    }


def _restore_tensor_state(
    target: object,
    state: Mapping[str, torch.Tensor],
    *,
    allow_additive_fields: bool = False,
) -> None:
    expected = {field.name for field in fields(target)}
    actual = set(state)
    valid = actual.issubset(expected) if allow_additive_fields else actual == expected
    if not valid:
        raise ValueError("checkpoint tensor state schema differs from runtime")
    for name, saved in state.items():
        current = getattr(target, name)
        if current.shape != saved.shape or current.dtype != saved.dtype:
            raise ValueError(f"checkpoint tensor {name} is incompatible")
        current.copy_(saved.to(current.device))


def _event_state(events: AppEventBatch) -> dict[str, torch.Tensor]:
    return _dataclass_tensor_state(events)


def _restore_events(
    value: Mapping[str, torch.Tensor], device: torch.device,
) -> AppEventBatch:
    expected = {field.name for field in fields(AppEventBatch)}
    if set(value) != expected:
        raise ValueError("checkpoint event schema differs from runtime")
    return AppEventBatch(**{
        name: tensor.to(device) for name, tensor in value.items()
    })


def _catalog_hash(catalog: PublicCatalog) -> str:
    digest = sha256()
    for field in fields(catalog):
        tensor = getattr(catalog, field.name).detach().cpu().contiguous()
        digest.update(field.name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_closure_hash(files: tuple[Path, ...], root: Path) -> str:
    digest = sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _world_source_hash() -> str:
    digital_twin = Path(__file__).resolve().parents[1]
    simulation = digital_twin.parent
    files = tuple((digital_twin / "world").rglob("*.py")) + (
        digital_twin / "catalog.py",
        digital_twin / "contracts.py",
    ) + tuple((simulation / "randomness").rglob("*.py"))
    return _source_closure_hash(files, simulation)


def _platform_source_hash() -> str:
    digital_twin = Path(__file__).resolve().parents[1]
    files = tuple((digital_twin / "platform").rglob("*.py")) + (
        digital_twin / "engine.py",
        digital_twin / "event_log.py",
    )
    return _source_closure_hash(files, digital_twin)


def _runtime_manifest(
    world: UserEcosystemWorld,
    platform: ReferenceRecommendationPlatform,
) -> dict[str, object]:
    return {
        "world_config": asdict(world.config),
        "world_manifest": world.manifest(),
        "platform_config": asdict(platform.config),
        "retrieval_config": asdict(platform.retriever.config),
        "ranking_config": asdict(platform.ranker.config),
        "feature_manifest_hash": platform.ranker.features.manifest.manifest_hash,
        "fine_scorer_versions": sorted(platform.ranker._fine_scorers),
        "learned_retriever": (
            None
            if platform.retriever.learned_retriever is None
            else {
                "serving_version_id": (
                    platform.retriever.learned_retriever.serving_version_id
                ),
                "index_version": platform.retriever.learned_retriever.index_version,
            }
        ),
    }


def _is_additive_contract(previous: object, current: object) -> bool:
    """Allow only new mapping keys; every historical value stays identical."""
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        return all(
            key in current and _is_additive_contract(value, current[key])
            for key, value in previous.items()
        )
    return previous == current


def _is_approved_contract_migration(
    previous: object,
    current: object,
    approved: Mapping[str, tuple[object, object]],
    path: str = "",
) -> bool:
    """Allow additive keys plus exact, preregistered value transitions."""
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        return all(
            key in current and _is_approved_contract_migration(
                value,
                current[key],
                approved,
                f"{path}.{key}" if path else str(key),
            )
            for key, value in previous.items()
        )
    if previous == current:
        return True
    return approved.get(path) == (previous, current)


def _policy_state(policy: CascadePolicy) -> dict[str, object]:
    return asdict(policy)


def _experiment_state(
    plan: ExperimentPlan | LayeredExperimentPlan,
) -> dict[str, object]:
    if isinstance(plan, ExperimentPlan):
        return {
            "kind": "ramped",
            "policies": {
                str(cell): _policy_state(policy)
                for cell, policy in plan.policies.items()
            },
            "experiment_seed": plan.experiment_seed,
            "control_fraction": plan.control_fraction,
            "treatment_fraction": plan.treatment_fraction,
            "analysis_cells": list(plan.analysis_cells),
            "default_cell": plan.default_cell,
            "assignment_unit": plan.assignment_unit,
            "eligible_surfaces": (
                None
                if plan.eligible_surfaces is None
                else list(plan.eligible_surfaces)
            ),
        }
    if isinstance(plan, LayeredExperimentPlan):
        return {
            "kind": "layered",
            "base_policy": _policy_state(plan.base_policy),
            "layers": [asdict(layer) for layer in plan.layers],
            "assignment_unit": plan.assignment_unit,
        }
    raise TypeError("checkpoint supports only factual digital-twin experiment plans")


def _restore_experiment(
    state: Mapping[str, object],
) -> ExperimentPlan | LayeredExperimentPlan:
    kind = state.get("kind")
    if kind == "ramped":
        policies = {
            int(cell): CascadePolicy(**value)
            for cell, value in state["policies"].items()
        }
        eligible = state["eligible_surfaces"]
        return ExperimentPlan(
            policies=policies,
            experiment_seed=int(state["experiment_seed"]),
            control_fraction=float(state["control_fraction"]),
            treatment_fraction=float(state["treatment_fraction"]),
            analysis_cells=tuple(state["analysis_cells"]),
            default_cell=int(state["default_cell"]),
            assignment_unit=str(state["assignment_unit"]),
            eligible_surfaces=None if eligible is None else tuple(eligible),
        )
    if kind == "layered":
        return LayeredExperimentPlan(
            base_policy=CascadePolicy(**state["base_policy"]),
            layers=tuple(PolicyLayer(**value) for value in state["layers"]),
            assignment_unit=str(state["assignment_unit"]),
        )
    raise ValueError("checkpoint experiment schema is unsupported")


def _world_state(world: UserEcosystemWorld) -> dict[str, object]:
    state = {
        "users": _dataclass_tensor_state(world.users),
        "supply": _dataclass_tensor_state(world.supply.state),
        "trend": {
            "strength": _tensor_payload(world.trends.state.strength),
            "momentum": _tensor_payload(world.trends.state.momentum),
            "last_time": world.trends.state.last_time,
        },
        "growth": {
            "campaign_intensity": _tensor_payload(
                world.growth.state.campaign_intensity,
            ),
            "referral_pressure": _tensor_payload(
                world.growth.state.referral_pressure,
            ),
            "creator_pressure": _tensor_payload(
                world.growth.state.creator_pressure,
            ),
            "last_time": world.growth.state.last_time,
        },
        "delayed": {
            str(ingest_time): _event_state(events)
            for ingest_time, events in world.delayed._pending.items()
        },
    }
    stats = getattr(world.response_authority, "stats", None)
    if stats is not None:
        state["response_authority_stats"] = stats()
    return state


def _graph_state(platform: ReferenceRecommendationPlatform) -> dict[str, object]:
    graph = platform.retriever.graph
    matrix = graph._matrix
    return {
        "indptr": torch.from_numpy(matrix.indptr.copy()),
        "indices": torch.from_numpy(matrix.indices.copy()),
        "data": torch.from_numpy(matrix.data.copy()),
        "pending_source": [
            torch.from_numpy(value.copy()) for value in graph._pending_source
        ],
        "pending_target": [
            torch.from_numpy(value.copy()) for value in graph._pending_target
        ],
        "last_user": torch.tensor(
            list(graph._last_item_by_user), dtype=torch.long,
        ),
        "last_item": torch.tensor(
            list(graph._last_item_by_user.values()), dtype=torch.long,
        ),
        "neighbor": _tensor_payload(graph.neighbor),
        "score": _tensor_payload(graph.score),
        "version": graph.version,
    }


def _platform_state(
    platform: ReferenceRecommendationPlatform,
) -> dict[str, object]:
    retriever = platform.retriever
    return {
        "projection": _dataclass_tensor_state(platform.projection.state),
        "retriever": {
            "last_refresh": retriever._last_refresh,
            "faiss_version": retriever.faiss.version,
            "indexed_active": _tensor_payload(retriever.faiss._indexed_active),
            "graph": _graph_state(platform),
        },
    }


def _restore_world(
    world: UserEcosystemWorld,
    state: Mapping[str, object],
    *,
    allow_additive_fields: bool = False,
) -> None:
    _restore_tensor_state(
        world.users, state["users"],
        allow_additive_fields=allow_additive_fields,
    )
    _restore_tensor_state(
        world.supply.state, state["supply"],
        allow_additive_fields=allow_additive_fields,
    )
    trend = state["trend"]
    world.trends.state.strength.copy_(
        trend["strength"].to(world.trends.state.strength.device)
    )
    world.trends.state.momentum.copy_(
        trend["momentum"].to(world.trends.state.momentum.device)
    )
    world.trends.state.last_time = int(trend["last_time"])
    growth = state.get("growth")
    if growth is not None:
        world.growth.state.campaign_intensity.copy_(
            growth["campaign_intensity"].to(
                world.growth.state.campaign_intensity.device,
            )
        )
        world.growth.state.referral_pressure.copy_(
            growth["referral_pressure"].to(
                world.growth.state.referral_pressure.device,
            )
        )
        world.growth.state.creator_pressure.copy_(
            growth["creator_pressure"].to(
                world.growth.state.creator_pressure.device,
            )
        )
        world.growth.state.last_time = int(growth["last_time"])
    device = world.catalog.item_id.device
    world.delayed._pending = {
        int(ingest_time): _restore_events(events, device)
        for ingest_time, events in state["delayed"].items()
    }
    response_stats = state.get("response_authority_stats")
    restore_stats = getattr(world.response_authority, "restore_stats", None)
    if response_stats is not None and restore_stats is not None:
        restore_stats(response_stats)


def _restore_graph(
    platform: ReferenceRecommendationPlatform, state: Mapping[str, object],
) -> None:
    graph = platform.retriever.graph
    graph._matrix = scipy.sparse.csr_matrix(
        (
            state["data"].numpy(),
            state["indices"].numpy(),
            state["indptr"].numpy(),
        ),
        shape=(graph.items, graph.items),
    )
    graph._pending_source = [value.numpy() for value in state["pending_source"]]
    graph._pending_target = [value.numpy() for value in state["pending_target"]]
    graph._last_item_by_user = dict(zip(
        state.get("last_user", torch.empty(0, dtype=torch.long)).tolist(),
        state.get("last_item", torch.empty(0, dtype=torch.long)).tolist(),
    ))
    graph.neighbor.copy_(state["neighbor"].to(graph.device))
    graph.score.copy_(state["score"].to(graph.device))
    graph.version = str(state["version"])


def _restore_platform(
    platform: ReferenceRecommendationPlatform,
    state: Mapping[str, object],
    *,
    allow_additive_fields: bool = False,
) -> None:
    _restore_tensor_state(
        platform.projection.state,
        state["projection"],
        allow_additive_fields=allow_additive_fields,
    )
    retriever_state = state["retriever"]
    retriever = platform.retriever
    retriever._last_refresh = int(retriever_state["last_refresh"])
    indexed = retriever_state["indexed_active"].to(retriever.device)
    retriever.faiss._index = None
    retriever.faiss._torch_item = None
    retriever.faiss._torch_embedding = None
    retriever.faiss._indexed_active.zero_()
    retriever.faiss.sync(indexed, str(retriever_state["faiss_version"]))
    _restore_graph(platform, retriever_state["graph"])


class WorldCheckpointStore:
    """Writes immutable objects and restores them into a compatible runtime."""

    def __init__(self, root: Path):
        self.root = root
        self.objects = root / "objects"
        self.refs = root / "refs"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.refs.mkdir(parents=True, exist_ok=True)

    def _write_object(self, value: object) -> str:
        with NamedTemporaryFile(dir=self.objects, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(value, stream)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        try:
            digest = file_sha256(temporary)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        target = self.objects / f"{digest}.pt"
        if not target.exists():
            os.replace(temporary, target)
        else:
            temporary.unlink()
            if file_sha256(target) != digest:
                raise ValueError("checkpoint object content hash mismatch")
        return digest

    def _read_object(self, digest: str) -> object:
        path = self.objects / f"{digest}.pt"
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("checkpoint object is missing or corrupted")
        return torch.load(path, map_location="cpu", weights_only=False)

    def save(
        self,
        kernel: AtomicSimulationKernel,
        logical_time: int,
        experiment: ExperimentPlan | LayeredExperimentPlan,
        *,
        parent_checkpoint_id: str = "",
        learning_cursors: Mapping[str, object] | None = None,
    ) -> WorldCheckpointRef:
        world, platform = self._compatible_runtime(kernel)
        experiment_state = _experiment_state(experiment)
        runtime_manifest = _runtime_manifest(world, platform)
        state_object = self._write_object({
            "world": _world_state(world),
            "platform": _platform_state(platform),
        })
        if kernel.event_log.durable:
            event_cursor = kernel.event_log.checkpoint_cursor()
            event_objects = ()
        else:
            event_cursor = {}
            event_objects = tuple(
                self._write_object(_event_state(batch))
                for batch in kernel.event_log._batches
            )
        manifest = {
            "schema": WORLD_CHECKPOINT_SCHEMA,
            "logical_time": logical_time,
            "parent_checkpoint_id": parent_checkpoint_id,
            "state_object": state_object,
            "event_objects": list(event_objects),
            "event_cursor": event_cursor,
            "catalog_sha256": _catalog_hash(world.catalog),
            "runtime_manifest": runtime_manifest,
            "runtime_sha256": sha256(
                _canonical_json(runtime_manifest)
            ).hexdigest(),
            "world_code_sha256": _world_source_hash(),
            "platform_code_sha256": _platform_source_hash(),
            "experiment": experiment_state,
            "experiment_sha256": sha256(
                _canonical_json(experiment_state)
            ).hexdigest(),
            "learning_cursors": dict(learning_cursors or {}),
            "event_manifest": kernel.event_log.manifest(),
        }
        checkpoint_id = sha256(_canonical_json(manifest)).hexdigest()
        path = self.refs / f"{checkpoint_id}.json"
        if not path.exists():
            self._write_json_atomic(path, manifest)
        elif json.loads(path.read_text()) != manifest:
            raise ValueError("checkpoint identity collision")
        return self._ref(checkpoint_id, manifest)

    def restore(
        self,
        kernel: AtomicSimulationKernel,
        checkpoint_id: str,
        *,
        require_code_match: bool = True,
        allow_additive_runtime_migration: bool = False,
        approved_runtime_changes: Mapping[
            str, tuple[object, object]
        ] | None = None,
    ) -> RestoredWorldCheckpoint:
        world, platform = self._compatible_runtime(kernel)
        manifest = self._manifest(checkpoint_id)
        if manifest["catalog_sha256"] != _catalog_hash(world.catalog):
            raise ValueError("checkpoint catalog differs from runtime")
        runtime = _runtime_manifest(world, platform)
        runtime_matches = manifest["runtime_sha256"] == sha256(
            _canonical_json(runtime)
        ).hexdigest()
        additive = (
            allow_additive_runtime_migration
            and _is_additive_contract(manifest["runtime_manifest"], runtime)
        )
        approved = (
            approved_runtime_changes is not None
            and _is_approved_contract_migration(
                manifest["runtime_manifest"], runtime,
                approved_runtime_changes,
            )
        )
        if not runtime_matches and not (additive or approved):
            raise ValueError("checkpoint runtime contract differs")
        if (
            require_code_match
            and manifest["world_code_sha256"] != _world_source_hash()
        ):
            raise ValueError("checkpoint world code closure differs from runtime")
        state = self._read_object(manifest["state_object"])
        _restore_world(
            world,
            state["world"],
            allow_additive_fields=allow_additive_runtime_migration,
        )
        _restore_platform(
            platform,
            state["platform"],
            allow_additive_fields=allow_additive_runtime_migration,
        )
        self._restore_event_log(kernel, manifest)
        projection = platform.projection.state
        if (
            allow_additive_runtime_migration
            and int(projection.user_exposure_cursor.sum()) > 0
            and int(projection.user_feed_exposure_cursor.sum()) == 0
        ):
            platform.projection.rebuild_exposures(
                kernel.event_log.read(
                    ingested_through=int(manifest["logical_time"]),
                ),
            )
        return RestoredWorldCheckpoint(
            ref=self._ref(checkpoint_id, manifest),
            experiment=_restore_experiment(manifest["experiment"]),
            learning_cursors=dict(manifest["learning_cursors"]),
        )

    def get_ref(self, checkpoint_id: str) -> WorldCheckpointRef:
        """Return a verified immutable checkpoint reference without restoring it."""
        return self._ref(checkpoint_id, self._manifest(checkpoint_id))

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Return whether ``ancestor_id`` is on the descendant parent chain."""
        self._manifest(ancestor_id)
        current_id = descendant_id
        visited: set[str] = set()
        while current_id:
            if current_id == ancestor_id:
                return True
            if current_id in visited:
                raise ValueError("checkpoint parent lineage contains a cycle")
            visited.add(current_id)
            current_id = str(self._manifest(current_id)["parent_checkpoint_id"])
        return False

    def _restore_event_log(
        self, kernel: AtomicSimulationKernel, manifest: Mapping[str, object],
    ) -> None:
        allowed = int(manifest["event_manifest"]["allowed_lateness"])
        if kernel.event_log.allowed_lateness != allowed:
            raise ValueError("checkpoint event-log lateness differs")
        if kernel.event_log.durable:
            cursor = dict(manifest.get("event_cursor", {}))
            if cursor:
                kernel.event_log.restore_cursor(cursor)
            else:
                kernel.event_log.restore_partitions(
                    tuple(manifest.get("event_partitions", ())),
                    dict(manifest["event_manifest"]),
                )
            return
        device = kernel.world.catalog.item_id.device
        kernel.event_log._batches.clear()
        kernel.event_log._ids_by_event_time.clear()
        kernel.event_log._events = 0
        kernel.event_log._ingest_watermark = -1
        for digest in manifest["event_objects"]:
            kernel.event_log.append(
                _restore_events(self._read_object(digest), device)
            )
        if kernel.event_log.manifest() != manifest["event_manifest"]:
            raise ValueError("restored event log differs from checkpoint")

    def _manifest(self, checkpoint_id: str) -> dict[str, object]:
        path = self.refs / f"{checkpoint_id}.json"
        if not path.is_file():
            raise KeyError(f"unknown world checkpoint: {checkpoint_id}")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != WORLD_CHECKPOINT_SCHEMA:
            raise ValueError("world checkpoint schema is unsupported")
        expected = sha256(_canonical_json(manifest)).hexdigest()
        if expected != checkpoint_id:
            raise ValueError("world checkpoint manifest hash mismatch")
        return manifest

    @staticmethod
    def _ref(
        checkpoint_id: str, manifest: Mapping[str, object],
    ) -> WorldCheckpointRef:
        return WorldCheckpointRef(
            checkpoint_id=checkpoint_id,
            logical_time=int(manifest["logical_time"]),
            parent_checkpoint_id=str(manifest["parent_checkpoint_id"]),
            state_object=str(manifest["state_object"]),
            event_objects=tuple(manifest["event_objects"]),
            catalog_sha256=str(manifest["catalog_sha256"]),
            runtime_sha256=str(manifest["runtime_sha256"]),
            world_code_sha256=str(manifest["world_code_sha256"]),
            platform_code_sha256=str(manifest["platform_code_sha256"]),
            experiment_sha256=str(manifest["experiment_sha256"]),
            event_cursor_sha256=sha256(_canonical_json(
                manifest.get("event_cursor", {}),
            )).hexdigest() if manifest.get("event_cursor") else "",
        )

    @staticmethod
    def _compatible_runtime(
        kernel: AtomicSimulationKernel,
    ) -> tuple[UserEcosystemWorld, ReferenceRecommendationPlatform]:
        if not isinstance(kernel.world, UserEcosystemWorld):
            raise TypeError("world checkpoint requires UserEcosystemWorld")
        if not isinstance(kernel.platform, ReferenceRecommendationPlatform):
            raise TypeError("world checkpoint requires reference platform")
        return kernel.world, kernel.platform

    @staticmethod
    def _write_json_atomic(path: Path, value: object) -> None:
        payload = _canonical_json(value)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
