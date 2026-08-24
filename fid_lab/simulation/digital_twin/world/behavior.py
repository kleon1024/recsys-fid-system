"""Hidden examination and action SCM for heterogeneous app surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...randomness.counter import uniform_for_item_channels
from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    ContentKind,
    EventType,
    RenderedSlateBatch,
    Surface,
    make_app_events,
)
from .state import UserWorldSnapshot


@dataclass(frozen=True)
class ResponseTensors:
    examined: torch.Tensor
    affinity: torch.Tensor
    utility: torch.Tensor
    dwell_ms: torch.Tensor
    action: dict[EventType, torch.Tensor]


def _event_draws(
    slate: RenderedSlateBatch, channels: int, seed: int,
) -> torch.Tensor:
    key = slate.request_id * 1_103_515_245 + slate.user_id
    channel = torch.arange(channels, device=key.device)[None, None, :]
    item = slate.item_ids.clamp_min(0)[:, :, None].expand(-1, -1, channels)
    return uniform_for_item_channels(
        key,
        item,
        channel.expand_as(item),
        0,
        1_201,
        seed,
    )


def _surface_mask(slate: RenderedSlateBatch, surface: Surface) -> torch.Tensor:
    return (slate.surface == int(surface))[:, None]


def _examination_probability(
    snapshot: UserWorldSnapshot,
    slate: RenderedSlateBatch,
) -> torch.Tensor:
    users = snapshot.users
    row = slate.user_id
    position = slate.positions.float()
    feed = _surface_mask(slate, Surface.FEED)
    live = _surface_mask(slate, Surface.LIVE)
    list_ui = ~(feed | live)
    continuation = torch.sigmoid(
        1.3
        + 1.15 * users.satisfaction[row, None]
        - 1.7 * users.fatigue[row, None]
        - 0.42 * position
        + 0.18 * users.response_style[row, 0, None]
    )
    list_scan = torch.sigmoid(
        1.7
        - 0.58 * position
        + 0.55 * users.activity[row, None]
        + 0.20 * users.response_style[row, 1, None]
    )
    probability = torch.where(feed | live, continuation, list_scan)
    probability = torch.where(
        (feed | live) & (position == 0), torch.ones_like(probability), probability,
    )
    return torch.where(list_ui | feed | live, probability, probability)


def _latent_utility(
    snapshot: UserWorldSnapshot,
    catalog: PublicCatalog,
    slate: RenderedSlateBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    users = snapshot.users
    truth = snapshot.catalog_truth
    row = slate.user_id
    item = slate.item_ids.clamp_min(0)
    semantic = truth.semantic_embedding[item]
    long_affinity = torch.einsum(
        "bkd,bd->bk", semantic, users.long_interest[row],
    )
    short_affinity = torch.einsum(
        "bkd,bd->bk", semantic, users.short_interest[row],
    )
    affinity = (
        (0.58 + 0.18 * users.habit[row, None]) * long_affinity
        + (0.64 - 0.22 * users.habit[row, None]) * short_affinity
    )
    local_match = (
        (catalog.country[item] == users.country[row, None]).float()
        + 0.8 * (catalog.region[item] == users.region[row, None]).float()
    )
    affordability = torch.exp(-torch.abs(
        torch.log1p(catalog.price[item])
        - 4.0 * users.spending_power[row, None]
    ))
    style = users.response_style[row]
    interaction = torch.sin(
        1.7 * affinity + 0.35 * style[:, 2, None]
    ) * torch.tanh(2.2 * truth.quality[item] - 1.0)
    utility = (
        (1.05 + 0.12 * style[:, 3, None]) * affinity
        + (0.42 + 0.10 * style[:, 4, None]) * truth.quality[item]
        - (0.65 + 0.12 * style[:, 5, None]) * truth.risk[item]
        + 0.24 * users.novelty[row, None]
        * (1.0 - catalog.quality_prior[item])
        + 0.16 * interaction
    )
    local = _surface_mask(slate, Surface.LOCAL)
    commerce = _surface_mask(slate, Surface.COMMERCE)
    posting = _surface_mask(slate, Surface.POSTING)
    utility += local * 0.34 * local_match
    utility += commerce * 0.30 * affordability
    utility += posting * 0.22 * local_match
    return affinity, utility


def sample_response_tensors(
    snapshot: UserWorldSnapshot,
    catalog: PublicCatalog,
    slate: RenderedSlateBatch,
    seed: int,
) -> ResponseTensors:
    users = snapshot.users
    row = slate.user_id
    item = slate.item_ids.clamp_min(0)
    valid = slate.valid
    draws = _event_draws(slate, 24, seed)
    examined = valid & (draws[:, :, 0] < _examination_probability(snapshot, slate))
    affinity, utility = _latent_utility(snapshot, catalog, slate)
    style = users.response_style[row]
    quality = snapshot.catalog_truth.quality[item]
    risk = snapshot.catalog_truth.risk[item]
    feed = _surface_mask(slate, Surface.FEED)
    live = _surface_mask(slate, Surface.LIVE)
    search = _surface_mask(slate, Surface.SEARCH)
    commerce = _surface_mask(slate, Surface.COMMERCE)
    local = _surface_mask(slate, Surface.LOCAL)
    posting = _surface_mask(slate, Surface.POSTING)
    play_probability = torch.sigmoid(
        -0.35 + 1.20 * utility + 0.35 * style[:, 0, None]
    )
    play = examined & (feed | live) & (draws[:, :, 1] < play_probability)
    play_3s = play & (draws[:, :, 2] < torch.sigmoid(
        -0.05 + 1.05 * utility + 0.85 * quality - 0.7 * risk,
    ))
    long_view = play_3s & (draws[:, :, 3] < torch.sigmoid(
        -0.9 + 1.18 * utility + 1.1 * quality
        - 0.75 * users.fatigue[row, None],
    ))
    duration = catalog.duration_seconds[item].clamp_min(1.0)
    complete = long_view & (draws[:, :, 4] < torch.sigmoid(
        0.8 + 1.0 * utility - 0.018 * duration,
    ))
    click_probability = torch.sigmoid(
        -1.25 + 1.38 * utility + 0.26 * style[:, 1, None]
    )
    click = examined & (search | commerce | local | posting) & (
        draws[:, :, 5] < click_probability
    )
    detail = click & ~posting & (draws[:, :, 6] < torch.sigmoid(
        -0.15 + 0.85 * utility + 0.3 * quality,
    ))
    create = click & posting & (draws[:, :, 7] < torch.sigmoid(
        -0.7 + 0.95 * utility + 0.75 * users.habit[row, None],
    ))
    action = {
        EventType.PLAY: play,
        EventType.PLAY_3S: play_3s,
        EventType.LONG_VIEW: long_view,
        EventType.COMPLETE: complete,
        EventType.CLICK: click,
        EventType.DETAIL: detail,
        EventType.LIKE: play & (draws[:, :, 8] < torch.sigmoid(-2.5 + utility)),
        EventType.COMMENT: play & (draws[:, :, 9] < torch.sigmoid(-3.5 + utility)),
        EventType.SHARE: play & (draws[:, :, 10] < torch.sigmoid(-3.25 + utility)),
        EventType.FOLLOW: examined & (play | live) & (
            draws[:, :, 11] < torch.sigmoid(-3.0 + utility)
        ),
        EventType.NEGATIVE: examined & (draws[:, :, 12] < torch.sigmoid(
            -3.2 - 0.9 * utility + 1.4 * risk,
        )),
        EventType.FAVORITE: detail & (local | search) & (
            draws[:, :, 13] < torch.sigmoid(-2.1 + utility)
        ),
        EventType.ADD_CART: detail & commerce & (
            draws[:, :, 14] < torch.sigmoid(-2.0 + utility)
        ),
        EventType.ORDER: detail & (commerce | local) & (
            draws[:, :, 15] < torch.sigmoid(-3.0 + 1.1 * utility)
        ),
        EventType.CREATE: create,
        EventType.PUBLISH: create & (
            draws[:, :, 16] < torch.sigmoid(-0.45 + 0.85 * utility)
        ),
    }
    action[EventType.PAYMENT] = action[EventType.ORDER] & (
        draws[:, :, 17] < torch.sigmoid(1.0 + 0.3 * utility)
    )
    action[EventType.SLIDE] = examined & feed & ~complete
    base_seconds = torch.exp(
        1.0 + 0.55 * utility + 0.50 * quality
        + 0.14 * style[:, 6, None]
    )
    engaged = play | click | create
    dwell_ms = torch.where(
        engaged,
        (1_000.0 * base_seconds).clamp(250.0, 300_000.0),
        torch.zeros_like(base_seconds),
    ).long()
    return ResponseTensors(examined, affinity, utility, dwell_ms, action)


def _events_for_mask(
    event_type: EventType,
    mask: torch.Tensor,
    slate: RenderedSlateBatch,
    catalog: PublicCatalog,
    *,
    duration_ms: torch.Tensor | None = None,
) -> AppEventBatch:
    row, position = torch.where(mask)
    if not len(row):
        return AppEventBatch.empty(slate.request_id.device)
    item = slate.item_ids[row, position]
    kind = catalog.content_kind[item]
    product = torch.where(
        kind == int(ContentKind.PRODUCT), item, torch.full_like(item, -1),
    )
    poi = torch.where(
        kind == int(ContentKind.POI), item, torch.full_like(item, -1),
    )
    order_id = (
        slate.request_id[row] * 10_000 + position
        if event_type in {EventType.ORDER, EventType.PAYMENT}
        else torch.full_like(item, -1)
    )
    return make_app_events(
        event_type,
        event_time=slate.event_time[row],
        request_id=slate.request_id[row],
        user_id=slate.user_id[row],
        surface=slate.surface[row],
        item_id=item,
        position=position,
        experiment_cell=slate.ui_variant[row],
        content_kind=kind,
        topic_id=catalog.topic_id[item],
        country=catalog.country[item],
        region=catalog.region[item],
        creator_id=catalog.creator_id[item],
        product_id=product,
        poi_id=poi,
        order_id=order_id,
        duration_ms=(
            None if duration_ms is None else duration_ms[row, position]
        ),
        logging_probability=slate.exposure_probability[row, position],
        assignment_probability=slate.assignment_probability[row],
        ordinal=position,
    )


def response_events(
    snapshot: UserWorldSnapshot,
    catalog: PublicCatalog,
    slate: RenderedSlateBatch,
    seed: int,
) -> AppEventBatch:
    sampled = sample_response_tensors(snapshot, catalog, slate, seed)
    batches = [
        _events_for_mask(EventType.IMPRESSION, slate.valid, slate, catalog),
        _events_for_mask(EventType.EXAMINE, sampled.examined, slate, catalog),
    ]
    for event_type, mask in sampled.action.items():
        batches.append(_events_for_mask(event_type, mask, slate, catalog))
    engaged = (
        sampled.action[EventType.PLAY]
        | sampled.action[EventType.CLICK]
        | sampled.action[EventType.CREATE]
    )
    batches.append(_events_for_mask(
        EventType.DWELL,
        engaged,
        slate,
        catalog,
        duration_ms=sampled.dwell_ms,
    ))
    positive = torch.stack((
        sampled.action[EventType.LONG_VIEW],
        sampled.action[EventType.LIKE],
        sampled.action[EventType.SHARE],
        sampled.action[EventType.CLICK],
        sampled.action[EventType.ORDER],
        sampled.action[EventType.PUBLISH],
    )).any(dim=0)
    request_value = (
        sampled.dwell_ms.float().sum(dim=1) / 1_000.0
        + 8.0 * positive.float().sum(dim=1)
    )
    users = snapshot.users
    leave_probability = torch.sigmoid(
        -2.4
        + 1.8 * users.fatigue[slate.user_id]
        + 0.10 * users.session_depth[slate.user_id].float()
        - 0.014 * request_value
        - 0.8 * users.satisfaction[slate.user_id]
    )
    leave_draw = _event_draws(slate, 24, seed)[:, 0, 23]
    leave = leave_draw < leave_probability
    batches.append(make_app_events(
        EventType.SESSION_END,
        event_time=slate.event_time[leave],
        request_id=slate.request_id[leave],
        user_id=slate.user_id[leave],
        surface=slate.surface[leave],
        experiment_cell=slate.ui_variant[leave],
        assignment_probability=slate.assignment_probability[leave],
    ))
    return AppEventBatch.concatenate(batches)
