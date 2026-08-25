from __future__ import annotations

import pytest
import torch

from fid_lab.simulation.digital_twin import (
    AppEventBatch,
    EventType,
    JoinerConfig,
    NegativeSource,
    ObservableProjection,
    RequestCandidateTrace,
    RequestLevelJoiner,
    TraceManifest,
    build_public_catalog,
    capture_request_context,
    corrected_sampled_softmax_loss,
    make_app_events,
    negative_source_counts,
)
from fid_lab.simulation.digital_twin.samples.negative_sampling import (
    build_recall_negatives,
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
        [1, 2, 9, 4, 5, 6],
        [11, 12, 19, 14, 15, 16],
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
        coarse_input_score=torch.linspace(0.9, 0.1, 12).reshape(2, 6),
        coarse_admission_probability=torch.tensor([
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.float),
        coarse_item_id=coarse,
        coarse_selected_score=torch.linspace(0.9, 0.1, 10).reshape(2, 5),
        fine_input_score=torch.linspace(0.8, 0.1, 10).reshape(2, 5),
        fine_admission_probability=torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ], dtype=torch.float),
        fine_item_id=fine,
        fine_selected_score=torch.linspace(0.8, 0.1, 8).reshape(2, 4),
        candidate_dense_features=torch.arange(
            2 * 6 * 11, dtype=torch.float32,
        ).reshape(2, 6, 11),
        candidate_sparse_fids=(
            torch.arange(2 * 6 * 13).reshape(2, 6, 13) + 1
        ),
        candidate_sparse_buckets=torch.remainder(
            torch.arange(2 * 6 * 13).reshape(2, 6, 13), 32,
        ),
        exposed_item_id=exposed,
        exposed_position=torch.tensor([[0, 1], [0, 1]]),
        exposure_probability=torch.ones(2, 2),
        selection_policy_kind=torch.zeros(2, dtype=torch.long),
        exploration_rate=torch.zeros(2),
        slate_log_probability=torch.zeros(2),
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
            feature_manifest_hash="a" * 64,
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
    assert bool(early.fine.label_applicable[1, 0, order])
    assert not bool(early.fine.label_mature[1, 0, order])
    assert not bool(early.fine.label_mask[1, 0, order])
    assert not bool(early.fine.label_applicable[0, 0, order])
    mature = joiner.materialize(trace, context, events, event_watermark=20)
    assert bool(mature.fine.label_mature[1, 0, order])
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
    assert set(joined.recall.negative_source.flatten().tolist()) <= {0, 1, 2, 3}
    assert joined.coarse.hard_label_mask[:, :2].all()
    assert not joined.coarse.hard_label_mask[:, 2:].any()
    assert joined.coarse.teacher_mask[:, :5].all()
    assert not joined.coarse.teacher_mask[:, 5:].any()
    assert torch.equal(joined.fine.context.request_id, trace.request_id)
    assert torch.equal(
        joined.fine.label_mask,
        joined.fine.label_applicable & joined.fine.label_mature,
    )
    assert torch.allclose(
        joined.fine.joint_logging_probability,
        joined.fine.exposure_probability
        * joined.fine.assignment_probability[:, None],
    )
    assert joined.manifest == trace.manifest
    assert torch.equal(
        joined.fine.dense_features,
        trace.candidate_dense_features[:, :5],
    )
    assert joined.fine.feature_manifest_hash == "a" * 64


def test_recall_sources_carry_q_expected_count_and_false_negative_mask():
    catalog, trace = build_catalog(), build_trace()
    context = capture_request_context(
        trace, ObservableProjection(2, catalog, history_length=4).snapshot(),
    )
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=20), catalog,
    ).materialize(trace, context, build_events(trace, catalog), 20)
    recall = joined.recall
    counts = negative_source_counts(20)
    for source, count in enumerate(counts):
        assert (recall.negative_source == source).sum(dim=1).tolist() == [
            count, count,
        ]
    expected = recall.negative_sampling_probability.clone()
    for source, count in enumerate(counts):
        expected[recall.negative_source == source] *= count
    assert torch.allclose(recall.negative_expected_count, expected)
    assert recall.negative_observed[
        recall.negative_source == int(NegativeSource.EXPOSED)
    ].all()
    assert torch.allclose(
        recall.negative_log_q[recall.negative_item_id >= 0],
        recall.negative_sampling_probability[
            recall.negative_item_id >= 0
        ].log(),
    )

    all_history = catalog.item_id[None, :]
    direct = build_recall_negatives(
        request_id=torch.tensor([99]),
        positive_item_id=torch.tensor([1]),
        exposed_item_id=torch.tensor([[2, 3]]),
        exposed_negative=torch.tensor([[True, True]]),
        recall_item_id=torch.tensor([[9, 4, 5]]),
        recalled_unexposed=torch.tensor([[True, True, True]]),
        history_item_id=all_history,
        catalog=catalog,
        total=20,
        seed=31,
    )
    assert direct.false_negative_mask[direct.item_id >= 0].all()


def test_sampled_softmax_correction_matches_exhaustive_oracle():
    positive = torch.tensor([0.7, -0.2])
    negative = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
    exhaustive = torch.nn.functional.cross_entropy(
        torch.cat((positive[:, None], negative), dim=1),
        torch.zeros(2, dtype=torch.long),
    )
    corrected = corrected_sampled_softmax_loss(
        positive,
        negative,
        torch.ones_like(negative),
        torch.ones_like(negative, dtype=torch.bool),
    )
    assert torch.allclose(corrected, exhaustive)
    masked = corrected_sampled_softmax_loss(
        positive,
        negative,
        torch.ones_like(negative),
        torch.tensor([[True, False], [True, True]]),
    )
    assert not torch.allclose(masked, exhaustive)


def test_in_batch_sampling_uses_peer_frequency_without_quadratic_pool():
    positives = torch.tensor([1, 1, 2, 3])
    catalog = build_catalog()
    samples = build_recall_negatives(
        request_id=torch.tensor([10, 20, 30, 40]),
        positive_item_id=positives,
        exposed_item_id=torch.full((4, 1), -1),
        exposed_negative=torch.zeros(4, 1, dtype=torch.bool),
        recall_item_id=torch.full((4, 1), -1),
        recalled_unexposed=torch.zeros(4, 1, dtype=torch.bool),
        history_item_id=torch.full((4, 1), -1),
        catalog=catalog,
        total=20,
        seed=7,
    )
    in_batch = samples.source == int(NegativeSource.IN_BATCH)
    for row in range(len(positives)):
        peers = torch.cat((positives[:row], positives[row + 1:]))
        expected = torch.stack(tuple(
            (peers == item).float().mean()
            for item in samples.item_id[row, in_batch[row]]
        ))
        assert torch.allclose(
            samples.sampling_probability[row, in_batch[row]], expected,
        )


def test_coarse_teacher_rank_detects_real_order_conflicts():
    catalog, trace = build_catalog(), build_trace()
    values = trace.__dict__.copy()
    values["fine_input_score"] = trace.fine_input_score.clone()
    values["fine_input_score"][0] = torch.tensor([0.8, 0.7, 0.9, 0.6, 0.5])
    reordered = RequestCandidateTrace(**values)
    context = capture_request_context(
        reordered, ObservableProjection(2, catalog, history_length=4).snapshot(),
    )
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=3), catalog,
    ).materialize(reordered, context, build_events(reordered, catalog), 20)
    assert joined.coarse.teacher_rank[0, :4].tolist() == [2, 3, 1, 4]
    assert joined.coarse.conflict_mask[0, :3].all()
    assert not bool(joined.coarse.conflict_mask[0, 3])


def test_context_is_chronological_heterogeneous_and_point_in_time():
    catalog, trace = build_catalog(), build_trace()
    values = trace.__dict__.copy()
    values["event_time"] = torch.tensor([10, 10])
    later_trace = RequestCandidateTrace(**values)
    projection = ObservableProjection(2, catalog, history_length=4)
    history = AppEventBatch.concatenate((
        observed_event(trace, catalog, 0, 0, EventType.PLAY, time=2),
        observed_event(trace, catalog, 0, 0, EventType.LIKE, time=4),
        observed_event(trace, catalog, 0, 1, EventType.DWELL, time=6),
    ))
    projection.ingest(history)
    context = capture_request_context(later_trace, projection.snapshot())
    valid = context.history_item_id[0] >= 0
    assert context.history_event_time[0, valid].tolist() == [2, 4, 6]
    assert context.history_event_type[0, valid].tolist() == [
        int(EventType.PLAY), int(EventType.LIKE), int(EventType.DWELL),
    ]
    assert (context.history_event_time[0, valid] <= 10).all()


def test_late_event_is_inserted_by_event_time_without_dropping_newer_history():
    catalog, trace = build_catalog(), build_trace()
    values = trace.__dict__.copy()
    values["event_time"] = torch.tensor([40, 40])
    request = RequestCandidateTrace(**values)
    projection = ObservableProjection(2, catalog, history_length=4)
    first = AppEventBatch.concatenate((
        observed_event(trace, catalog, 0, 0, EventType.PLAY, time=10),
        observed_event(trace, catalog, 0, 1, EventType.LIKE, time=20),
    ))
    projection.ingest(first)
    late = observed_event(trace, catalog, 0, 0, EventType.DWELL, time=15)
    late_values = late.__dict__.copy()
    late_values["ingest_time"] = torch.full_like(late.ingest_time, 30)
    projection.ingest(AppEventBatch(**late_values))
    context = capture_request_context(request, projection.snapshot())
    valid = context.history_item_id[0] >= 0
    assert context.history_event_time[0, valid].tolist() == [10, 15, 20]
    assert context.history_ingest_time[0, valid].tolist() == [10, 30, 20]


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
