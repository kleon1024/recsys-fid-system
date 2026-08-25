from __future__ import annotations

import pytest

from fid_lab.simulation.digital_twin import (
    ExperimentPlan,
    WorldBranchRegistry,
    WorldCheckpointStore,
)

from .test_world_checkpoint import _system


def _child_checkpoint(store, checkpoint_id, *, experiment_seed):
    kernel, _ = _system()
    restored = store.restore(kernel, checkpoint_id, require_code_match=False)
    active = restored.experiment.policies[0]
    treatment = restored.experiment.policies[1]
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=experiment_seed,
        control_fraction=restored.experiment.control_fraction,
        treatment_fraction=restored.experiment.treatment_fraction,
    )
    logical_time = restored.ref.logical_time + 1
    kernel.step(logical_time, plan)
    return store.save(
        kernel,
        logical_time,
        plan,
        parent_checkpoint_id=checkpoint_id,
    )


def test_main_and_diagnostic_worlds_advance_without_overwriting_each_other(
    tmp_path,
):
    kernel, plan = _system()
    kernel.step(0, plan)
    store = WorldCheckpointStore(tmp_path)
    base = store.save(kernel, 0, plan)
    registry = WorldBranchRegistry(store)

    main = registry.initialize_main(base.checkpoint_id)
    shadow = registry.fork(
        "main",
        "shadow/ranker-v2",
        kind="shadow",
        purpose="diagnose ranker behavior without training contamination",
    )
    assert main.training_authority
    assert not shadow.training_authority
    assert shadow.head_checkpoint_id == main.head_checkpoint_id

    main_child = _child_checkpoint(
        store, base.checkpoint_id, experiment_seed=3_001,
    )
    shadow_child = _child_checkpoint(
        store, base.checkpoint_id, experiment_seed=3_003,
    )
    registry.advance(
        "main",
        main_child.checkpoint_id,
        expected_head_checkpoint_id=base.checkpoint_id,
    )
    registry.advance(
        "shadow/ranker-v2",
        shadow_child.checkpoint_id,
        expected_head_checkpoint_id=base.checkpoint_id,
    )

    assert registry.get("main").head_checkpoint_id == main_child.checkpoint_id
    assert (
        registry.get("shadow/ranker-v2").head_checkpoint_id
        == shadow_child.checkpoint_id
    )
    assert main_child.checkpoint_id != shadow_child.checkpoint_id
    assert store.is_ancestor(base.checkpoint_id, main_child.checkpoint_id)
    assert store.is_ancestor(base.checkpoint_id, shadow_child.checkpoint_id)


def test_branch_registry_rejects_stale_or_cross_branch_updates(tmp_path):
    kernel, plan = _system()
    kernel.step(0, plan)
    store = WorldCheckpointStore(tmp_path)
    base = store.save(kernel, 0, plan)
    registry = WorldBranchRegistry(store)
    registry.initialize_main(base.checkpoint_id)
    registry.fork(
        "main",
        "replay/feature-fix",
        kind="replay",
        purpose="reproduce a feature fix",
    )
    main_child = _child_checkpoint(
        store, base.checkpoint_id, experiment_seed=3_101,
    )
    registry.advance(
        "main",
        main_child.checkpoint_id,
        expected_head_checkpoint_id=base.checkpoint_id,
    )

    with pytest.raises(ValueError, match="changed concurrently"):
        registry.advance(
            "main",
            main_child.checkpoint_id,
            expected_head_checkpoint_id=base.checkpoint_id,
        )
    with pytest.raises(ValueError, match="direct branch child"):
        registry.advance(
            "replay/feature-fix",
            store.save(
                kernel,
                1,
                plan,
                parent_checkpoint_id=main_child.checkpoint_id,
            ).checkpoint_id,
            expected_head_checkpoint_id=base.checkpoint_id,
        )


def test_only_main_can_be_training_authority(tmp_path):
    kernel, plan = _system()
    kernel.step(0, plan)
    store = WorldCheckpointStore(tmp_path)
    base = store.save(kernel, 0, plan)
    registry = WorldBranchRegistry(store)
    registry.initialize_main(base.checkpoint_id)
    branch = registry.fork(
        "main",
        "counterfactual/value-tree",
        kind="counterfactual",
        purpose="stress-test a value tree intervention",
    )
    invalid = registry.invalidate(
        branch.name,
        reason="diagnostic configuration was invalid",
    )
    assert not invalid.training_authority
    assert invalid.status == "invalid"
    with pytest.raises(ValueError, match="main cannot be invalidated"):
        registry.invalidate("main", reason="must not rewrite factual history")
