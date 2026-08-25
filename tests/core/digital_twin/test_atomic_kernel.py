from __future__ import annotations

from dataclasses import fields

import torch

from fid_lab.simulation.digital_twin import (
    AppEventBatch,
    APP_EVENT_SCHEMA_VERSION,
    AtomicSimulationKernel,
    EventType,
    ExperimentPlan,
    ObservableEventLog,
    PlatformRequestBatch,
    RenderedSlateBatch,
)
from fid_lab.simulation.digital_twin.contracts import deterministic_event_id


def event_batch(
    event_type, request_id, user_id, item_id, logical_time, cell,
):
    rows = len(request_id)
    event_types = torch.full((rows,), int(event_type), dtype=torch.long)
    ordinal = torch.zeros(rows, dtype=torch.long)
    integer_missing = torch.full((rows,), -1, dtype=torch.long)
    return AppEventBatch(
        event_id=deterministic_event_id(
            request_id, event_types, item_id, ordinal
        ),
        schema_version=torch.full(
            (rows,), APP_EVENT_SCHEMA_VERSION, dtype=torch.long,
        ),
        event_type=event_types,
        event_time=torch.full((rows,), logical_time, dtype=torch.long),
        ingest_time=torch.full((rows,), logical_time, dtype=torch.long),
        request_id=request_id,
        user_id=user_id,
        surface=torch.remainder(user_id, 6),
        item_id=item_id,
        post_id=integer_missing.clone(),
        source_candidate_id=integer_missing.clone(),
        creator_id=integer_missing.clone(),
        merchant_id=integer_missing.clone(),
        advertiser_id=integer_missing.clone(),
        product_id=integer_missing.clone(),
        poi_id=integer_missing.clone(),
        order_id=integer_missing.clone(),
        position=ordinal,
        content_kind=integer_missing.clone(),
        topic_id=integer_missing.clone(),
        country=integer_missing.clone(),
        region=integer_missing.clone(),
        query_id=integer_missing.clone(),
        duration_ms=integer_missing.clone(),
        value=torch.ones(rows),
        logging_probability=torch.ones(rows),
        assignment_probability=torch.ones(rows),
        experiment_cell=cell,
    )


class FakeWorld:
    def __init__(self, users=6):
        self.user_id = torch.arange(users)
        self.market_budget = torch.tensor(100.0)
        self.impressions = torch.zeros(users)

    def schedule(self, logical_time):
        request_id = self.user_id * 1_000 + logical_time
        return event_batch(
            EventType.SESSION_START,
            request_id,
            self.user_id,
            torch.full_like(self.user_id, -1),
            logical_time,
            torch.full_like(self.user_id, -1),
        )

    def snapshot(self):
        return {"market_budget": self.market_budget.clone()}

    def respond(self, snapshot, slate):
        assert float(snapshot["market_budget"]) == 100.0
        return event_batch(
            EventType.IMPRESSION,
            slate.request_id,
            slate.user_id,
            slate.item_ids[:, 0],
            int(slate.event_time[0]),
            slate.ui_variant,
        )

    def commit(self, events):
        impression = events.event(EventType.IMPRESSION)
        if impression.any():
            self.market_budget -= events.value[impression].sum()
            self.impressions.scatter_add_(
                0, events.user_id[impression], events.value[impression]
            )


class FakePlatform:
    def __init__(self):
        self.ingested: list[int] = []

    def ingest(self, events):
        self.ingested.extend(int(value) for value in events.event_id)

    def snapshot(self):
        return {"ingested": tuple(self.ingested)}

    def open_requests(self, entry_events):
        session = entry_events.event(EventType.SESSION_START)
        return PlatformRequestBatch(
            request_id=entry_events.request_id[session],
            user_id=entry_events.user_id[session],
            surface=entry_events.surface[session],
            event_time=entry_events.event_time[session],
            query_topic=torch.full_like(entry_events.user_id[session], -1),
        )

    def render(
        self, snapshot, requests, policy, experiment_cell,
        assignment_probability,
    ):
        assert len(snapshot["ingested"]) == 6
        item = requests.user_id + int(policy) * 100
        return RenderedSlateBatch(
            request_id=requests.request_id,
            user_id=requests.user_id,
            surface=requests.surface,
            event_time=requests.event_time,
            item_ids=item[:, None],
            positions=torch.zeros(len(item), 1, dtype=torch.long),
            valid=torch.ones(len(item), 1, dtype=torch.bool),
            ui_variant=torch.full(
                (len(item),), experiment_cell, dtype=torch.long
            ),
            exposure_probability=torch.ones(len(item), 1),
            selection_policy_kind=torch.zeros(len(item), dtype=torch.long),
            exploration_rate=torch.zeros(len(item)),
            slate_log_probability=torch.zeros(len(item)),
            assignment_probability=assignment_probability,
        )


def run(order):
    world = FakeWorld()
    platform = FakePlatform()
    log = ObservableEventLog()
    result = AtomicSimulationKernel(world, platform, log).step(
        7,
        ExperimentPlan.ramped_user_ab(
            active_policy=10,
            treatment_policy=20,
            experiment_seed=20260824,
            control_fraction=0.5,
            treatment_fraction=0.5,
        ),
        cell_order=order,
    )
    return world, platform, log, result


def test_atomic_tick_is_invariant_to_experiment_cell_order():
    left = run((-1, 0, 1))
    right = run((1, 0, -1))
    torch.testing.assert_close(left[0].market_budget, right[0].market_budget)
    torch.testing.assert_close(left[0].impressions, right[0].impressions)
    left_events, right_events = left[2].read(), right[2].read()
    for field in fields(AppEventBatch):
        assert torch.equal(
            getattr(left_events, field.name), getattr(right_events, field.name)
        ), field.name
    assert left[3].cell_counts == right[3].cell_counts
    assert sum(left[3].cell_counts.values()) == 6


def test_boundary_payloads_exclude_model_scores_and_hidden_state():
    _, _, log, result = run((-1, 0, 1))
    slate_fields = {field.name for field in fields(RenderedSlateBatch)}
    event_fields = {field.name for field in fields(AppEventBatch)}
    assert "score" not in " ".join(slate_fields)
    assert "feature" not in " ".join(slate_fields)
    assert "latent" not in " ".join(event_fields)
    assert result.rendered_requests == 6
    assert log.manifest()["events"] == 12


def test_event_log_rejects_duplicate_delivery():
    world = FakeWorld(users=2)
    events = world.schedule(3)
    log = ObservableEventLog()
    log.append(events)
    try:
        log.append(events)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate event delivery must fail closed")


def test_event_log_duplicate_failure_is_transactional_across_event_times():
    log = ObservableEventLog()
    original = event_batch(
        EventType.IMPRESSION,
        torch.tensor([101]),
        torch.tensor([1]),
        torch.tensor([7]),
        2,
        torch.tensor([0]),
    )
    future = event_batch(
        EventType.IMPRESSION,
        torch.tensor([202]),
        torch.tensor([2]),
        torch.tensor([9]),
        3,
        torch.tensor([1]),
    )
    log.append(original)
    attempted = AppEventBatch.concatenate((future, original))
    try:
        log.append(attempted)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("cross-time duplicate delivery must fail closed")
    assert log.manifest()["events"] == 1
    log.append(future)
    assert log.manifest()["events"] == 2


def test_ramped_user_ab_leaves_unallocated_traffic_on_active_policy():
    users = torch.arange(5_000).repeat_interleave(2)
    requests = PlatformRequestBatch(
        request_id=torch.arange(len(users)),
        user_id=users,
        surface=torch.remainder(users, 2),
        event_time=torch.zeros_like(users),
        query_topic=torch.full_like(users, -1),
    )
    eligible = users.remainder(2) == 0
    active, treatment = object(), object()
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=20260824,
        control_fraction=0.05,
        treatment_fraction=0.05,
        eligible_surfaces=(0,),
    )
    assignment = plan.assign(requests)
    assert torch.equal(
        assignment.cell_by_request[0::2],
        assignment.cell_by_request[1::2],
    )
    assert (assignment.cell_by_request[~eligible] == -1).all()
    assert (assignment.probability_by_request[~eligible] == 1.0).all()
    allocated_default = eligible & (assignment.cell_by_request == -1)
    torch.testing.assert_close(
        assignment.probability_by_request[allocated_default],
        torch.full_like(
            assignment.probability_by_request[allocated_default], 0.9,
        ),
    )
    assert (
        assignment.probability_by_request[assignment.experiment_mask] == 0.05
    ).all()
    assert 350 <= int(assignment.experiment_mask.sum()) <= 650
    assert plan.policies[-1] is active
    assert plan.policies[0] is active
    assert plan.policies[1] is treatment


def test_event_identity_closes_the_former_linear_hash_collision():
    event_id = deterministic_event_id(
        torch.tensor([25, 1]),
        torch.tensor([3, 0]),
        torch.tensor([0, 548_667]),
        torch.tensor([2, 0]),
    )
    assert event_id[0] != event_id[1]
