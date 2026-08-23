"""Tensor candidate scoring, coarse pass-through, and fine-rank selection."""

from __future__ import annotations

import torch


def _fine_score(policy, eligible, user_ids, state, candidates):
    topics = candidates["topics"]
    observed_affinity = torch.einsum(
        "bkd,bd->bk", topics, state["observed_interest"]
    )
    local_observed_affinity = torch.einsum(
        "bkd,bd->bk", topics, state["local_observed_interest"]
    )
    quality = candidates["quality"]
    freshness = candidates["freshness"]
    score = (
        eligible * policy.affinity_weight * observed_affinity
        + (eligible * policy.quality_weight + 1.0 - eligible) * quality
        + (eligible * policy.freshness_weight + 0.15 * (1.0 - eligible))
        * freshness
        - eligible
        * policy.fatigue_match_penalty
        * state["fatigue"][:, None]
        * observed_affinity.clamp_min(0)
    )
    candidate_index = torch.arange(score.shape[1], device=score.device)[None, :]
    collision = torch.sin(user_ids[:, None] * 0.013 + candidate_index * 12.9898)
    score += eligible * policy.uid_collision_weight * collision
    local_signal = (
        0.24 * observed_affinity
        + 0.18 * candidates["commerce"]
        + 0.28 * candidates["same_city"]
        + 0.16 * candidates["poi_quality"]
        + 0.14 * candidates["inventory"]
        + policy.search_weight * candidates["search_match"]
        + policy.retarget_weight * candidates["retarget_match"]
    )
    score += eligible * policy.local_weight * candidates["is_poi"] * local_signal
    local_intent = torch.clamp(
        candidates["search_match"] + candidates["retarget_match"], 0.0, 1.0
    )
    intent_quality = (
        0.35 * local_observed_affinity
        + 0.25 * quality
        + 0.20 * candidates["poi_quality"]
        + 0.10 * candidates["commerce"]
        + 0.10 * candidates["same_city"]
    )
    score += (
        eligible
        * policy.local_intent_quality_weight
        * candidates["is_poi"]
        * local_intent
        * intent_quality
    )
    score += (
        eligible
        * policy.local_embedding_correction_weight
        * candidates["is_poi"]
        * local_intent
        * (local_observed_affinity - observed_affinity)
    )
    return score, observed_affinity


def _coarse_mask(policy, observed_affinity, candidates):
    candidate_count = observed_affinity.shape[1]
    keep = candidate_count if policy.coarse_keep <= 0 else min(
        policy.coarse_keep, candidate_count
    )
    if keep == candidate_count:
        return torch.ones_like(observed_affinity, dtype=torch.bool), keep
    coarse_score = (
        policy.coarse_affinity_weight * observed_affinity
        + policy.coarse_quality_weight * candidates["quality"]
        + policy.coarse_local_weight
        * candidates["is_poi"]
        * (
            candidates["same_city"]
            + candidates["search_match"]
            + candidates["retarget_match"]
        )
    )
    indices = torch.topk(coarse_score, keep, dim=1).indices
    mask = torch.zeros_like(coarse_score, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask, keep


def stage_diagnostics(candidates, choice, fine_choice, coarse_survives):
    batch = torch.arange(len(choice), device=choice.device)
    chosen_item = candidates["item_ids"][batch, choice]
    fine_item = candidates["item_ids"][batch, fine_choice]
    oracle = candidates["audit_oracle_item"]
    recall = candidates["audit_oracle_in_recall"]
    coarse = coarse_survives
    attribution = torch.zeros_like(choice)
    attribution = torch.where(recall, torch.ones_like(attribution), attribution)
    attribution = torch.where(coarse, torch.full_like(attribution, 2), attribution)
    attribution = torch.where(
        coarse & (fine_item == oracle), torch.full_like(attribution, 3), attribution
    )
    attribution = torch.where(
        coarse & (fine_item == oracle) & (chosen_item == oracle),
        torch.full_like(attribution, 4),
        attribution,
    )
    return {
        "recall_oracle_survives": recall,
        "coarse_oracle_survives": coarse,
        "stage_attribution": attribution,
        "route_valid_counts": candidates["route_valid_counts"],
        "unique_recall_count": candidates["unique_recall_count"],
    }


def materialize_selected(
    policy, user_ids, state, candidates, choice, fine_choice, fine_scores,
    mix_scores, coarse_mask, coarse_keep, device,
):
    """Materialize one exposure while preserving every candidate-stage score."""
    batch_index = torch.arange(len(user_ids), device=device)
    names = (
        "topics", "quality", "is_poi", "commerce", "poi_quality", "inventory",
        "same_city", "search_match", "retarget_match", "fulfillment", "candidate_topic",
        "item_ids", "content_type", "ad_value", "live_value", "duration",
        "popularity", "author", "route_bits", "recall_score", "coarse_score",
    )
    selected = {name: candidates[name][batch_index, choice] for name in names}
    if "stay_nonlinear" in candidates:
        selected["stay_nonlinear"] = candidates["stay_nonlinear"][batch_index, choice]
    selected["fine_scores"] = fine_scores
    selected["mix_scores"] = mix_scores
    selected["fine_choice"] = fine_choice
    selected["final_choice"] = choice
    true_affinity = torch.einsum(
        "bkd,bd->bk", candidates["topics"], state["interest"]
    )
    true_feed_utility = true_affinity + 0.45 * candidates["quality"]
    chosen_utility = true_feed_utility[batch_index, choice]
    audit_coarse_survives = (
        (candidates["item_ids"] == candidates["audit_oracle_item"][:, None])
        & coarse_mask
    ).any(dim=1)
    selected.update(
        stage_diagnostics(candidates, choice, fine_choice, audit_coarse_survives)
    )
    selected["coarse_pass_fraction"] = torch.full_like(
        chosen_utility, coarse_keep / fine_scores.shape[1]
    ) * candidates["coarse_pass_fraction"]
    selected["oracle_regret"] = torch.maximum(
        candidates["merged_oracle_utility"], candidates["audit_oracle_utility"]
    ) - chosen_utility
    selected["poi_candidate_fraction"] = candidates["is_poi"].float().mean(dim=1)
    if policy.multi_queue:
        organic = candidates["content_type"] == 0
        best_organic = true_feed_utility.masked_fill(~organic, -1e9).max(dim=1).values
        selected["organic_opportunity_cost"] = torch.clamp(
            best_organic - chosen_utility, min=0.0
        ) * (selected["content_type"] == 2)
    else:
        selected["organic_opportunity_cost"] = torch.zeros(
            len(user_ids), device=device
        )
    return selected


def select_candidate(policy, user_ids, state, candidates, device, step, config=None):
    if hasattr(policy, "select_candidate"):
        if config is None:
            raise ValueError("artifact policy requires the tensor config")
        return policy.select_candidate(
            user_ids, state, candidates, device, step, config
        )
    score, observed_affinity = _fine_score(
        policy, state["eligible"], user_ids, state, candidates
    )
    coarse_mask, coarse_keep = _coarse_mask(
        policy, observed_affinity, candidates
    )
    score = score.masked_fill(~coarse_mask, -1e9)
    fine_choice = score.argmax(dim=1)
    fine_scores = score.clone()
    if policy.multi_queue:
        is_live = candidates["content_type"] == 1
        is_ad = candidates["content_type"] == 2
        score += is_live * policy.live_weight * candidates["live_value"]
        score += is_ad * policy.ad_weight * candidates["ad_value"]
        ad_allowed = (
            (state["ads_served"] < policy.max_ads_per_session)
            & ((step - state["last_ad_step"]) > policy.min_ad_gap)
        )
        live_allowed = state["live_served"] < policy.max_live_per_session
        score = score.masked_fill(is_ad & ~ad_allowed[:, None], -1e9)
        score = score.masked_fill(is_live & ~live_allowed[:, None], -1e9)
    choice = score.argmax(dim=1)
    return materialize_selected(
        policy, user_ids, state, candidates, choice, fine_choice, fine_scores,
        score, coarse_mask, coarse_keep, device,
    )
