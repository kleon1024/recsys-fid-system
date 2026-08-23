"""Position-biased search cascade and closed/open-loop Local outcomes."""

from __future__ import annotations

import torch

from ..contracts import LOCAL_SEARCH_TASKS
from .samples import LocalSearchExamples
from .world import deterministic_gumbel, deterministic_uniform, hidden_utility


def _cascade(world, poi, utility, clicked):
    requests, catalog, config = world.requests, world.catalog, world.config
    detail_probability = torch.sigmoid(
        -0.75 + 0.90 * utility + 0.45 * requests.urgency
    )
    detail = clicked & (
        deterministic_uniform(requests.request_id, poi, 302, config.seed)
        < detail_probability
    )
    save_probability = torch.sigmoid(
        -2.2 + 0.85 * catalog.quality[poi] + 0.55 * utility
    )
    saved = detail & (
        deterministic_uniform(requests.request_id, poi, 303, config.seed)
        < save_probability
    )
    order_probability = torch.sigmoid(
        -3.5 + 1.0 * utility + 0.9 * catalog.availability[poi]
        + 0.65 * requests.urgency
    )
    ordered = detail & (
        deterministic_uniform(requests.request_id, poi, 304, config.seed)
        < order_probability
    )
    pixel_observable = catalog.closed_loop[poi] | (
        deterministic_uniform(requests.request_id, poi, 305, config.seed) < 0.72
    )
    return detail, saved, ordered, pixel_observable


def _labels(world, candidates, top, selected_rank, states, observable):
    clicked, detail, saved, ordered = states
    config = world.config
    labels = torch.zeros(
        config.requests, config.merged_candidates, len(LOCAL_SEARCH_TASKS),
        device=top.device,
    )
    masks = torch.zeros_like(labels)
    exposed = torch.zeros(
        config.requests, config.merged_candidates,
        dtype=torch.bool, device=top.device,
    )
    exposed.scatter_(1, top, True)
    masks[:, :, :3] = exposed[:, :, None]
    masks[:, :, 3] = exposed
    batch = torch.arange(config.requests, device=top.device)
    selected = top.gather(1, selected_rank[:, None]).squeeze(1)
    for task, state in enumerate((clicked, detail, saved, ordered)):
        labels[batch[state], selected[state], task] = 1.0
    open_loop_unobservable = clicked & ~observable
    masks[batch[open_loop_unobservable], selected[open_loop_unobservable], 3] = 0
    return labels, masks


def simulate_response(world, candidates, scores):
    config, requests, catalog = world.config, world.requests, world.catalog
    top = torch.topk(scores, config.exposed_candidates, dim=1).indices
    exposed = candidates.poi_ids.gather(1, top)
    utility = hidden_utility(world, exposed)
    position = torch.arange(config.exposed_candidates, device=scores.device).float()
    propensity = torch.exp(-0.14 * position).clamp_min(0.20)
    choice = utility - 0.10 * position[None, :] + deterministic_gumbel(
        requests.request_id[:, None], exposed, 300, config.seed
    )
    best, selected_rank = choice.max(1)
    poi = exposed.gather(1, selected_rank[:, None]).squeeze(1)
    examined = deterministic_uniform(
        requests.request_id, poi, 301, config.seed
    ) < propensity[selected_rank]
    clicked = examined & (best > requests.outside_preference)
    selected_utility = utility.gather(1, selected_rank[:, None]).squeeze(1)
    detail, saved, ordered, observable = _cascade(
        world, poi, selected_utility, clicked
    )
    labels, masks = _labels(
        world, candidates, top, selected_rank,
        (clicked, detail, saved, ordered), observable,
    )
    stay = clicked.float() * (2.0 + 3.2 * torch.sigmoid(selected_utility))
    stay += detail.float() * (3.0 + 2.0 * catalog.quality[poi])
    active_day = torch.clamp(
        0.0007 * detail.float() + 0.0012 * saved.float()
        + 0.0020 * ordered.float(), max=0.004,
    )
    examples = LocalSearchExamples(
        requests.request_id, candidates.poi_ids, candidates.route_bits, top,
        propensity[None, :].expand(config.requests, -1), labels, masks, scores,
    )
    examples.validate(config.exposed_candidates)
    return {
        "examples": examples,
        "clicked": clicked,
        "detail": detail,
        "saved": saved,
        "ordered": ordered,
        "pixel_observable": observable,
        "selected_poi": poi,
        "selected_utility": selected_utility,
        "stay_seconds": stay,
        "active_day": active_day,
        "selected_risk": catalog.risk[poi],
        "closed_loop_order": ordered & catalog.closed_loop[poi],
        "open_loop_order": ordered & ~catalog.closed_loop[poi],
    }
