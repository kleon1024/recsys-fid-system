"""GPU-native visibility filtering and reversible score adjustments."""

from __future__ import annotations


def _eligibility(candidates, state, step, config, base_eligible):
    safety_eligible = candidates["predicted_integrity_risk"] < (
        config.max_predicted_integrity_risk
    )
    poi = candidates["is_poi"].bool()
    poi_allowed = (
        (state["poi_served"] < config.max_poi_per_session)
        & ((step - state["last_poi_step"]) > config.min_poi_gap)
    )
    eligible = base_eligible & safety_eligible & (~poi | poi_allowed[:, None])
    no_candidate = ~eligible.any(dim=1)
    if no_candidate.any():
        fallback_risk = candidates["predicted_integrity_risk"].masked_fill(
            ~base_eligible, float("inf")
        )
        safest = fallback_risk.argmin(dim=1)
        eligible[no_candidate, safest[no_candidate]] = True
    return eligible, no_candidate


def govern_scores(scores, candidates, state, step, config, base_eligible=None):
    """Apply visibility first, then diversity and exploration adjustments."""
    if base_eligible is None:
        base_eligible = scores > -1e8
    eligible, fallback = _eligibility(
        candidates, state, step, config, base_eligible
    )
    adjusted = scores.masked_fill(~eligible, -1e9)
    repeated_cluster = (
        candidates["duplicate_cluster"] == state["last_duplicate_cluster"][:, None]
    ) & (state["last_duplicate_cluster"][:, None] >= 0)
    repeated_author = (
        candidates["author"] == state["last_author"][:, None]
    ) & (state["last_author"][:, None] >= 0)
    adjusted = adjusted - repeated_cluster * config.repeated_cluster_penalty
    adjusted = adjusted - repeated_author * config.repeated_author_penalty
    adjusted = adjusted + candidates["creator_need"] * config.new_creator_boost
    diagnostics = {
        "governance_eligible": eligible,
        "governance_repeated_cluster": repeated_cluster,
        "governance_repeated_author": repeated_author,
        "governance_fallback": fallback,
    }
    return adjusted, diagnostics
