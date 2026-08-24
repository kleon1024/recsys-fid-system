"""One feature contract shared by retrieval training and tensor serving."""

from __future__ import annotations

import torch


CONTEXT_FEATURE_INDICES = (6, 7, 8, 9, 24, 25, 26, 27)


def _category_means(category, values, topics):
    output = torch.zeros(len(values), topics, device=values.device)
    count = torch.zeros_like(output)
    output.scatter_add_(1, category, values)
    count.scatter_add_(1, category, torch.ones_like(values))
    return output / count.clamp_min(1.0)


def logged_query_features(payload, rows=None, topics=12):
    rows = slice(None) if rows is None else rows
    features = payload["candidate_features"][rows].float()
    category = torch.round(features[:, :, 17] * (topics - 1)).long()
    observed = _category_means(category, features[:, :, 0], topics)
    local = _category_means(category, features[:, :, 23], topics)
    context = features[:, 0][
        :, list(CONTEXT_FEATURE_INDICES)
    ]
    return torch.cat((observed, local, context), dim=1)


def live_query_features(state, category_centers):
    context = torch.stack((
        state["satisfaction"],
        state["fatigue"],
        state["trust"],
        state["commerce_propensity"],
        state["account_age_days"] / 3_650.0,
        state["historical_activity"] / 200.0,
        state["lifecycle_bucket"].float() / 3.0,
        state["region_bucket"].float() / 9.0,
    ), dim=1)
    observed = state["observed_interest"] @ category_centers.T
    local = state["local_observed_interest"] @ category_centers.T
    return torch.cat((observed, local, context), dim=1)


def category_centers(catalog, topics=12):
    centers = torch.zeros(topics, catalog.topics.shape[1], device=catalog.topics.device)
    count = torch.zeros(topics, 1, device=catalog.topics.device)
    centers.index_add_(0, catalog.category, catalog.topics)
    count.index_add_(0, catalog.category, torch.ones(
        len(catalog.category), 1, device=catalog.topics.device
    ))
    return torch.nn.functional.normalize(centers / count.clamp_min(1.0), dim=1)


def catalog_item_features(catalog):
    scalar = torch.stack((
        catalog.quality,
        catalog.freshness,
        catalog.is_poi,
        catalog.commerce,
        catalog.poi_quality,
        catalog.inventory,
        catalog.city.float() / 99.0,
        catalog.fulfillment.float() / 2.0,
        catalog.popularity,
        torch.log1p(catalog.duration_seconds) / torch.log(
            torch.tensor(181.0, device=catalog.quality.device)
        ),
        catalog.author.float() / 1023.0,
    ), dim=1)
    return torch.cat((catalog.topics, scalar), dim=1)
