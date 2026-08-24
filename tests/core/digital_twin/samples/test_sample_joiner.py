from __future__ import annotations

import pytest
import torch

from fid_lab.simulation.digital_twin import (
    AppEventBatch,
    EventType,
    JoinerConfig,
    ObservableProjection,
    RequestCandidateTrace,
    RequestLevelJoiner,
    TraceManifest,
    build_public_catalog,
    capture_request_context,
    make_app_events,
)


def build_catalog():
    return build_public_catalog(
        items=120,
        creators=20,
        merchants=10,
        advertisers=5,
        topics=8,
        countries=2,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=401,
        device="cpu",
    )


def build_trace():
    recall = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [11, 12, 13, 14, 15, 16],
    ])
    coarse = recall[:, :5]
    fine = coarse[:, :4]
    exposed = fine[:, :2]
    return RequestCandidateTrace(
        request_id=torch.tensor([101, 201]),
        user_id=torch.tensor([0, 1]),
        surface=torch.tensor([0, 2]),
        event_time=torch.tensor([0, 0]),
        query_topic=torch.tensor([-1, -1]),
        user_country=torch.tensor([0, 1]),
        user_region=torch.tensor([0, 3]),
        user_creator_id=torch.tensor([0, 1]),
        route_item_id=recall[:, None, :],
        route_score=torch.linspace(1.0, 0.1, 12).reshape(2, 1, 6),
        route_valid=torch.ones(2, 1, 6, dtype=torch.bool),
        route_lifecycle_id=torch.full((2, 1, 6), 2),
        recall_item_id=recall,
        recall_route_id=torch.tensor([
            [0, 1, 1, 2, 3, 4],
            [2, 2, 3, 3, 4, 5],
        ]),
        recall_score=torch.linspace(1.0, 0.1, 12).reshape(2, 6),
        recall_sampling_probability=torch.ones(2, 6),
        recall_lifecycle_id=torch.full((2, 6), 2),
        coarse_item_id=coarse,
        coarse_score=torch.linspace(0.9, 0.1, 10).reshape(2, 5),
        coarse_sampling_probability=torch.ones(2, 5),
        fine_item_id=fine,
        fine_score=torch.linspace(0.8, 0.1, 8).reshape(2, 4),
        exposed_item_id=exposed,
        exposed_position=torch.tensor([[0, 1], [0, 1]]),
        exposure_probability=torch.ones(2, 2),
        experiment_cell=torch.tensor([0, 1]),
        assignment_probability=torch.tensor([0.05, 0.05]),
        recall_version_id=torch.tensor([1, 1]),
        coarse_version_id=torch.tensor([1, 2]),
        fine_version_id=torch.tensor([1, 2]),
        mix_version_id=torch.tensor([1, 2]),
        manifest=TraceManifest(
            schema_version="request-candidate-trace-v1",
            feature_version="feature-v1",
            catalog_version="catalog-v1",
            policy_registry_version="policy-registry-v1",
            route_names=("fixture",),
        ),
    )


def observed_event(trace, catalog, request_row, position, event_type, time=0):
    request = trace.request_id[request_row:request_row + 1]
    user = trace.user_id[request_row:request_row + 1]
    surface = trace.surface[request_row:request_row + 1]
    item = trace.exposed_item_id[
        request_row:request_row + 1, position
    ]
    ordinal = torch.tensor([position])
    return make_app_events(
        event_type,
        event_time=time,
        ingest_time=time,
        request_id=request,
        user_id=user,
        surface=surface,
        item_id=item,
        position=ordinal,
        creator_id=catalog.creator_id[item],
        merchant_id=catalog.merchant_id[item],
        advertiser_id=catalog.advertiser_id[item],
        content_kind=catalog.content_kind[item],
        topic_id=catalog.topic_id[item],
        experiment_cell=trace.experiment_cell[request_row:request_row + 1],
        logging_probability=torch.ones(1),
        assignment_probability=trace.assignment_probability[
            request_row:request_row + 1
        ],
        ordinal=ordinal,
    )


def build_events(trace, catalog):
    batches = []
    for request in range(2):
        for position in range(2):
            batches.append(observed_event(
                trace, catalog, request, position, EventType.IMPRESSION,
            ))
    batches.append(observed_event(
        trace, catalog, 0, 0, EventType.PLAY,
    ))
    batches.append(observed_event(
        trace, catalog, 0, 0, EventType.PLAY_3S,
    ))
    batches.append(observed_event(
        trace, catalog, 0, 0, EventType.LONG_VIEW,
    ))
    batches.append(observed_event(
        trace, catalog, 1, 0, EventType.ORDER, time=5,
    ))
    return AppEventBatch.concatenate(batches)


def test_trace_rejects_broken_stage_closure():
    trace = build_trace()
    values = trace.__dict__.copy()
    values["exposed_item_id"] = torch.tensor([[99, 2], [11, 12]])
    with pytest.raises(ValueError, match="exposed is not a subset"):
        RequestCandidateTrace(**values)


def test_joiner_masks_delayed_labels_until_watermark_maturity():
    catalog, trace = build_catalog(), build_trace()
    projection = ObservableProjection(2, catalog, history_length=4)
    context = capture_request_context(trace, projection.snapshot())
    joiner = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=3), catalog,
    )
    events = build_events(trace, catalog)
    early = joiner.materialize(trace, context, events, event_watermark=4)
    long_view = early.fine.task_names.index("long_view")
    order = early.fine.task_names.index("order")
    assert float(early.fine.labels[0, 0, long_view]) == 1.0
    assert bool(early.fine.label_mask[0, 0, long_view])
    assert float(early.fine.labels[1, 0, order]) == 1.0
    assert not bool(early.fine.label_mask[1, 0, order])
    mature = joiner.materialize(trace, context, events, event_watermark=20)
    assert bool(mature.fine.label_mask[1, 0, order])


def test_three_authorities_preserve_observability_and_teacher_boundaries():
    catalog, trace = build_catalog(), build_trace()
    projection = ObservableProjection(2, catalog, history_length=4)
    context = capture_request_context(trace, projection.snapshot())
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=3), catalog,
    ).materialize(
        trace,
        context,
        build_events(trace, catalog),
        event_watermark=20,
    )
    assert joined.recall.positive_item_id.tolist() == [1, 11]
    assert (joined.recall.negative_sampling_probability > 0).all()
    assert set(joined.recall.negative_source.flatten().tolist()) <= {0, 1}
    assert joined.coarse.hard_label_mask[:, :2].all()
    assert not joined.coarse.hard_label_mask[:, 2:].any()
    assert joined.coarse.teacher_mask[:, :4].all()
    assert not joined.coarse.teacher_mask[:, 4:].any()
    assert torch.equal(joined.fine.context.request_id, trace.request_id)
    assert joined.manifest == trace.manifest


def test_context_capture_rejects_projection_from_the_future():
    catalog, trace = build_catalog(), build_trace()
    projection = ObservableProjection(2, catalog, history_length=4)
    later = make_app_events(
        EventType.REGISTRATION,
        event_time=5,
        ingest_time=5,
        request_id=torch.tensor([501]),
        user_id=torch.tensor([0]),
        surface=torch.tensor([-1]),
    )
    projection.ingest(later)
    with pytest.raises(ValueError, match="later than a request"):
        capture_request_context(trace, projection.snapshot())
