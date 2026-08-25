"""One sequential factual Launch Review transaction for the main Feed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..contracts import AppEventBatch, ContentKind, EventType, Surface
from ..engine import ExperimentPlan
from ..evaluation.experiment import FactualABAccumulator, aa_decision
from ..evaluation.request import RequestWindowAccumulator
from ..learning.request_stream import FactualRequestStream
from ..learning.request_stream import FactualRequestPartitionRef
from ..learning.contracts import ServingCompatibility, learning_source_hash
from ..learning.registry import PersistentModelRegistry
from ..platform import CascadePolicy
from ..profile import STANDARD_FEED_PROFILE, SimulationProfile
from ..runtime_paths import RuntimePaths
from .retrieval_ladder import RetrievalLadderConfig, _build_kernel


LAUNCH_SPEC_SCHEMA = "factual-feed-launch-spec/v1"
LAUNCH_REVIEW_SCHEMA = "factual-feed-launch-review/v1"


@dataclass(frozen=True)
class FeedLaunchSpec:
    launch_id: str
    kind: str
    hypothesis: str
    isolated_change: str
    primary_metric: str
    treatment_changes: dict[str, object]
    experiment_seed: int
    control_fraction: float = 0.2
    treatment_fraction: float = 0.2
    minimum_triggered_users: int = 5_000
    minimum_ticks: int = 96
    maximum_ticks: int = 192
    mde_relative: float = 0.005
    predecessor_launch_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"aa", "baseline", "policy"}:
            raise ValueError("unsupported Feed launch kind")
        if not self.launch_id or not self.hypothesis or not self.isolated_change:
            raise ValueError("launch identity and hypothesis are required")
        if not 0 < self.control_fraction < 1:
            raise ValueError("control traffic is invalid")
        if not 0 < self.treatment_fraction < 1:
            raise ValueError("treatment traffic is invalid")
        if self.control_fraction + self.treatment_fraction > 1:
            raise ValueError("experiment traffic exceeds one")
        if not 0 < self.minimum_ticks <= self.maximum_ticks:
            raise ValueError("launch tick window is invalid")
        if self.minimum_triggered_users <= 1 or self.mde_relative <= 0:
            raise ValueError("launch power gate is invalid")

    def manifest(self) -> dict[str, object]:
        return {"schema": LAUNCH_SPEC_SCHEMA, **asdict(self)}

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    @classmethod
    def load(cls, path: Path) -> FeedLaunchSpec:
        value = json.loads(path.read_text())
        if value.pop("schema", None) != LAUNCH_SPEC_SCHEMA:
            raise ValueError("Feed launch spec schema is unsupported")
        return cls(**value)


def canonical_random_policy(profile: SimulationProfile) -> CascadePolicy:
    return CascadePolicy(
        name="feed-random-dedup-v1",
        recall_version_id=0,
        coarse_version_id=0,
        fine_version_id=0,
        mix_version_id=0,
        enabled_routes=("random",),
        feed_exposure_dedup_ticks=30 * profile.ticks_per_day,
        feed_session_dedup=True,
    )


def _baseline_plan(
    policy: CascadePolicy,
    spec: FeedLaunchSpec,
) -> ExperimentPlan:
    return ExperimentPlan.ramped_user_ab(
        active_policy=policy,
        treatment_policy=policy,
        experiment_seed=spec.experiment_seed + 10_000,
        control_fraction=spec.control_fraction,
        treatment_fraction=spec.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )


def _runtime_config(profile: SimulationProfile, device: str) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        users=profile.users,
        items=profile.items,
        ticks_per_day=profile.ticks_per_day,
        seed=profile.seed,
        device=device,
    )


def initialize_canonical_runtime(
    paths: RuntimePaths,
    profile: SimulationProfile = STANDARD_FEED_PROFILE,
    *,
    device: str = "cuda",
) -> str:
    paths.initialize(profile)
    _, kernel = _build_kernel(_runtime_config(profile, device))
    store = WorldCheckpointStore(paths.checkpoints)
    branches = WorldBranchRegistry(store)
    try:
        current = branches.get("main")
    except KeyError:
        policy = canonical_random_policy(profile)
        initialization = FeedLaunchSpec(
            launch_id="runtime-initialization",
            kind="baseline",
            hypothesis="create the single factual world",
            isolated_change="initialize persistent state without user events",
            primary_metric="dwell_seconds",
            treatment_changes={},
            experiment_seed=profile.seed + 401,
            minimum_triggered_users=2,
            minimum_ticks=1,
            maximum_ticks=1,
        )
        ref = store.save(
            kernel,
            logical_time=-1,
            experiment=_baseline_plan(policy, initialization),
            learning_cursors={
                "launch_ladder": {
                    "reviewed": [],
                    "passed": [],
                    "next_launch_id": "F-AA-00",
                    "last_review": None,
                },
            },
        )
        branches.initialize_main(ref.checkpoint_id)
        return ref.checkpoint_id
    store.get_ref(current.head_checkpoint_id)
    return current.head_checkpoint_id


def _active_policy(plan) -> CascadePolicy:
    if not isinstance(plan, ExperimentPlan):
        raise ValueError("sequential Feed launch requires a simple experiment plan")
    return plan.policies[-1]


def _resolved_policies(
    active: CascadePolicy,
    spec: FeedLaunchSpec,
) -> tuple[CascadePolicy, CascadePolicy]:
    treatment = replace(
        active,
        name=f"{active.name}-{spec.launch_id.lower()}",
        **spec.treatment_changes,
    )
    return active, treatment


def _serving_compatibility(kernel) -> ServingCompatibility:
    platform = kernel.platform
    manifest = platform.ranker.features.manifest
    return ServingCompatibility(
        feature_manifest_hash=manifest.manifest_hash,
        feature_version=manifest.schema_version,
        fid_version=f"fid-{manifest.fid_version}",
        catalog_version=platform.config.catalog_version,
        index_version=platform.retriever.index_version,
        code_sha256=learning_source_hash(),
    )


def _install_policy_artifacts(
    kernel,
    paths: RuntimePaths,
    policies: tuple[CascadePolicy, ...],
) -> None:
    registry = PersistentModelRegistry(paths.model_registry)
    compatibility = _serving_compatibility(kernel)
    coarse_versions = {
        policy.coarse_version_id for policy in policies
        if policy.coarse_version_id > 0
    }
    fine_versions = {
        policy.fine_version_id for policy in policies
        if policy.fine_version_id > 0
    }
    artifacts = {}
    for version in coarse_versions | fine_versions:
        _, artifact = registry.load_version_for_serving(
            version, compatibility,
        )
        if not callable(getattr(artifact, "score", None)):
            raise ValueError("ranking policy references a non-ranking artifact")
        artifacts[version] = artifact
    for version in coarse_versions:
        kernel.platform.install_coarse_scorer(version, artifacts[version])
    for version in fine_versions:
        kernel.platform.install_fine_scorer(version, artifacts[version])
    for policy in policies:
        kernel.platform.validate_policy_artifacts(policy)


def _source_revision(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("factual launch requires a clean Git checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _feed_impression_keys(
    events: AppEventBatch,
    item_universe: int,
) -> torch.Tensor:
    selected = (
        events.event(EventType.IMPRESSION)
        & (events.surface == int(Surface.FEED))
        & (events.content_kind == int(ContentKind.SHORT_VIDEO))
        & (events.user_id >= 0)
        & (events.item_id >= 0)
    )
    return (
        events.user_id[selected] * item_universe + events.item_id[selected]
    ).detach().cpu()


def _feed_repeat_report(keys: tuple[torch.Tensor, ...]) -> dict[str, object]:
    if not keys:
        return {"impressions": 0, "repeated_impressions": 0, "repeat_rate": 0.0}
    key = torch.cat(keys)
    unique = int(torch.unique(key).numel())
    impressions = len(key)
    repeated = impressions - unique
    return {
        "impressions": impressions,
        "repeated_impressions": repeated,
        "repeat_rate": repeated / impressions,
    }


def _policy_decision(
    spec: FeedLaunchSpec,
    ab: dict[str, object],
    repeats: dict[str, object],
) -> dict[str, str]:
    enough = min(ab["control_users"], ab["treatment_users"]) >= (
        spec.minimum_triggered_users
    )
    if spec.kind == "aa":
        decision = aa_decision(ab)
        if not enough:
            return {
                "decision": "hold",
                "reason": "A/A has not reached its pre-registered user floor",
            }
        return decision
    if spec.kind == "baseline":
        passes = enough and repeats["repeat_rate"] == 0.0
        return {
            "decision": "pass" if passes else "hold",
            "reason": (
                "random baseline and exact recent dedup are observable"
                if passes else "baseline sample or duplicate gate is incomplete"
            ),
        }
    primary = ab["metrics"][spec.primary_metric]
    negative = ab["metrics"]["negative"]
    primary_estimated = primary.get("status") == "estimated"
    negative_estimated = negative.get("status") == "estimated"
    regresses = enough and (
        (
            primary_estimated
            and primary["relative_delta"] is not None
            and primary["ci95_high"] < 0.0
        )
        or (negative_estimated and negative["ci95_low"] > 0.0)
    )
    passes = (
        enough
        and primary_estimated
        and primary["relative_delta"] is not None
        and primary["relative_delta"] >= spec.mde_relative
        and primary["ci95_low"] > 0.0
        and not (negative_estimated and negative["ci95_low"] > 0.0)
    )
    return {
        "decision": "pass" if passes else "reject" if regresses else "hold",
        "reason": (
            "primary MDE and negative-feedback guardrail pass"
            if passes
            else "powered regression or negative-feedback guardrail fails"
            if regresses
            else "powered business gate is not satisfied"
        ),
    }


def _write_review(
    paths: RuntimePaths,
    review: dict[str, object],
) -> Path:
    path = paths.launch_journal / f"{review['launch_id']}.json"
    payload = json.dumps(review, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError("immutable launch review already exists with other content")
        return path
    temporary = path.with_suffix(".pending")
    temporary.write_text(payload)
    temporary.replace(path)
    return path


def _execute_window(
    kernel,
    plan: ExperimentPlan,
    stream: FactualRequestStream,
    transaction_id: str,
    start: int,
    ticks: int,
    profile: SimulationProfile,
    spec: FeedLaunchSpec,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    tuple[torch.Tensor, ...],
    tuple[FactualRequestPartitionRef, ...],
    int,
]:
    impression_keys = []
    staged = []
    requests = RequestWindowAccumulator()
    ab = FactualABAccumulator(
        profile.users,
        control_fraction=spec.control_fraction,
        treatment_fraction=spec.treatment_fraction,
        device=kernel.platform.catalog.item_id.device,
    )
    logical_time = start
    for _ in range(ticks):
        tick = kernel.step(logical_time, plan)
        staged.append(stream.stage(
            transaction_id,
            tick,
            kernel.platform.projection.snapshot(),
            kernel.world.manifest(),
        ))
        if tick.candidate_trace is not None:
            events = AppEventBatch.concatenate((
                tick.entry_events, tick.response_events,
            ))
            ab.update(tick.candidate_trace, events)
            requests.update(tick.candidate_trace)
            keys = _feed_impression_keys(events, profile.items)
            if len(keys):
                impression_keys.append(keys)
        logical_time += 1
    return (
        requests.stage(),
        requests.support(),
        ab.report(),
        tuple(impression_keys),
        tuple(staged),
        logical_time,
    )


def run_feed_launch(
    paths: RuntimePaths,
    spec: FeedLaunchSpec,
    profile: SimulationProfile = STANDARD_FEED_PROFILE,
    *,
    device: str = "cuda",
    source_revision: str | None = None,
) -> dict[str, object]:
    revision = source_revision or _source_revision(Path.cwd())
    initialize_canonical_runtime(paths, profile, device=device)
    paths.initialize(profile)
    _, kernel = _build_kernel(_runtime_config(profile, device))
    store = WorldCheckpointStore(paths.checkpoints)
    branches = WorldBranchRegistry(store)
    branch = branches.get("main")
    restored = store.restore(kernel, branch.head_checkpoint_id)
    cursor = dict(restored.learning_cursors.get("launch_ladder", {}))
    reviewed = list(cursor.get("reviewed", []))
    passed = list(cursor.get("passed", []))
    if spec.launch_id in reviewed:
        raise ValueError("launch has already completed on this factual world")
    if spec.predecessor_launch_id and (
        spec.predecessor_launch_id not in passed
    ):
        raise ValueError("launch predecessor has not passed")
    active = _active_policy(restored.experiment)
    control, treatment = _resolved_policies(active, spec)
    _install_policy_artifacts(kernel, paths, (control, treatment))
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=spec.experiment_seed,
        control_fraction=spec.control_fraction,
        treatment_fraction=spec.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    stream = FactualRequestStream(paths.request_stream / "main", branch)
    stream.reconcile_through(restored.ref.logical_time)
    transaction_id = f"{spec.launch_id}-{branch.head_checkpoint_id[:12]}"
    start = restored.ref.logical_time + 1
    try:
        stage, support, ab, impression_keys, staged, logical_time = (
            _execute_window(
                kernel,
                plan,
                stream,
                transaction_id,
                start,
                spec.maximum_ticks,
                profile,
                spec,
            )
        )
        repeats = _feed_repeat_report(impression_keys)
        decision = _policy_decision(spec, ab, repeats)
        review = {
            "schema": LAUNCH_REVIEW_SCHEMA,
            "launch_id": spec.launch_id,
            "quality_claim": "synthetic factual-world evidence only",
            "source_revision": revision,
            "spec": spec.manifest(),
            "spec_hash": spec.spec_hash,
            "profile_hash": profile.profile_hash,
            "parent_checkpoint_id": branch.head_checkpoint_id,
            "analysis_time": [start, logical_time - 1],
            "stage": stage,
            "support": support,
            "feed_repeat": repeats,
            "ab": ab,
            **decision,
        }
        next_active = treatment if decision["decision"] == "pass" and (
            spec.kind == "policy"
        ) else control
        reviewed.append(spec.launch_id)
        if decision["decision"] == "pass":
            passed.append(spec.launch_id)
        cursor = {
            "reviewed": reviewed,
            "passed": passed,
            "next_launch_id": None,
            "last_review": review,
        }
        stream.commit_staged(transaction_id, staged)
        checkpoint = store.save(
            kernel,
            logical_time - 1,
            _baseline_plan(next_active, spec),
            parent_checkpoint_id=branch.head_checkpoint_id,
            learning_cursors={
                **restored.learning_cursors,
                "launch_ladder": cursor,
                "factual_request_stream": {
                    "branch": "main",
                    "stream_sha256": stream.stream_sha256,
                    "last_collected_time": logical_time - 1,
                    "partitions": len(stream.refs(training=True)),
                },
            },
        )
        branches.advance(
            "main",
            checkpoint.checkpoint_id,
            expected_head_checkpoint_id=branch.head_checkpoint_id,
        )
    except Exception:
        stream.abort_staged(transaction_id)
        raise
    final = {**review, "result_checkpoint_id": checkpoint.checkpoint_id}
    _write_review(paths, final)
    return final
