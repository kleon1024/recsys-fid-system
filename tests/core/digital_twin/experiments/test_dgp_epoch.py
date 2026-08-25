from __future__ import annotations

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.engine import ExperimentPlan
from fid_lab.simulation.digital_twin.experiments.dgp_epoch import (
    DGPEpochMigrationConfig,
    migrate_dgp_epoch,
)
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
)
from fid_lab.simulation.digital_twin.platform import CascadePolicy
from fid_lab.simulation.digital_twin.world.authority import (
    BehavioralSCMResponseAuthority,
    FormulaResponseAuthority,
)


def test_dgp_epoch_migration_backfills_memory_and_closes_old_samples(tmp_path):
    config = RetrievalLadderConfig(
        users=256,
        items=2_400,
        device="cpu",
        ticks_per_day=8,
        checkpoint_root=str(tmp_path / "checkpoints"),
    )
    _, old_kernel = _build_kernel(config)
    old_kernel.world.response_authority = FormulaResponseAuthority()
    policy = CascadePolicy("factual-baseline", 1, 1, 1)
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=policy,
        treatment_policy=policy,
        experiment_seed=809,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    old_kernel.step(0, plan)
    store = WorldCheckpointStore(tmp_path / "checkpoints")
    old = store.save(old_kernel, 0, plan)
    WorldBranchRegistry(store).initialize_main(old.checkpoint_id)

    report = migrate_dgp_epoch(DGPEpochMigrationConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        users=256,
        items=2_400,
        device="cpu",
        ticks_per_day=8,
    ))

    assert report["parent_checkpoint_id"] == old.checkpoint_id
    assert report["historical_samples_training_eligible"] is False
    assert report["users_with_exposure_memory"] > 0
    _, current_kernel = _build_kernel(config)
    restored = store.restore(current_kernel, report["checkpoint_id"])
    assert current_kernel.world.response_authority.version == (
        BehavioralSCMResponseAuthority.version
    )
    assert restored.learning_cursors["dgp_epoch_v5"][
        "historical_request_stream_training_eligible"
    ] is False
