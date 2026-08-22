"""Device-resident batched Feed trajectory simulator.

The semantic simulator remains the contract oracle. This engine deliberately
uses dense synthetic candidates so millions of users can exercise sequential
state transitions and A/B power without Python objects per user.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt
from time import perf_counter

import torch


@dataclass(frozen=True)
class TensorFeedConfig:
    users: int = 1_000_000
    steps: int = 24
    candidates: int = 20
    topics: int = 12
    batch_users: int = 25_000
    seed: int = 20260823
    device: str = "cuda:0"
    count_inactive_play_bug: bool = False


@dataclass(frozen=True)
class TensorPolicy:
    name: str
    affinity_weight: float
    quality_weight: float
    freshness_weight: float
    fatigue_match_penalty: float = 0.0
    eligible_fraction: float = 1.0
    observation_noise: float = 0.12
    realtime_interest_rate: float = 0.06
    uid_collision_weight: float = 0.0


POPULAR = TensorPolicy("quality_baseline", 0.0, 1.0, 0.15)
PERSONALIZED = TensorPolicy("personalized_rank", 1.0, 0.45, 0.10, 0.12)
PERSONALIZED_1PCT = TensorPolicy(
    "personalized_rank_1pct_trigger", 1.0, 0.45, 0.10, 0.12, 0.01
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _choose(
    policy, eligible, user_ids, observed_interest, fatigue, topics, quality, freshness
):
    observed_affinity = torch.einsum("bkd,bd->bk", topics, observed_interest)
    affinity_weight = eligible * policy.affinity_weight
    quality_weight = eligible * policy.quality_weight + (1.0 - eligible)
    freshness_weight = eligible * policy.freshness_weight + 0.15 * (1.0 - eligible)
    score = (
        affinity_weight * observed_affinity
        + quality_weight * quality
        + freshness_weight * freshness
        - eligible
        * policy.fatigue_match_penalty
        * fatigue[:, None]
        * observed_affinity.clamp_min(0)
    )
    candidate_index = torch.arange(score.shape[1], device=score.device)[None, :]
    collision = torch.sin(user_ids[:, None] * 0.013 + candidate_index * 12.9898)
    score += eligible * policy.uid_collision_weight * collision
    return score.argmax(dim=1)


def _sample_response(generator, active, fatigue, satisfaction, affinity, quality, device):
    users = len(active)
    duration = 3.0 + 177.0 * torch.rand(users, generator=generator, device=device)
    play_draw = (
        torch.rand(users, generator=generator, device=device)
        < torch.sigmoid(3.0 + 0.4 * affinity - 0.6 * fatigue)
    )
    played = play_draw & active
    stay_log_mean = (
        0.45 + 1.7 * affinity + 0.55 * quality + 0.20 * satisfaction - fatigue
    )
    stay = torch.minimum(
        duration,
        torch.exp(
            stay_log_mean
            + 0.65 * torch.randn(users, generator=generator, device=device)
        ),
    ) * played
    long_view = (stay >= torch.minimum(torch.full_like(stay, 10.0), duration)) & active
    hlt = (stay >= torch.minimum(torch.full_like(stay, 30.0), duration)) & active
    like = (
        torch.rand(users, generator=generator, device=device)
        < torch.sigmoid(-4.2 + 1.8 * affinity + 0.8 * quality)
    ) & played
    negative = (
        torch.rand(users, generator=generator, device=device)
        < torch.sigmoid(-5.0 - 1.7 * affinity - 0.8 * quality + 2.0 * fatigue)
    ) & active
    return stay, long_view, hlt, like, negative, played, play_draw


def _accumulate_cells(cell_stats, user_ids, user_metrics):
    bucket = torch.remainder(user_ids * 1_664_525 + 1_013_904_223, 2**31)
    assigned = bucket < 2**30
    rates = user_metrics[:, 1:] / user_metrics[:, :1].clamp_min(1.0)
    for cell, mask in enumerate((~assigned, assigned)):
        values = rates[mask]
        cell_stats[cell, :, 0] += mask.sum()
        cell_stats[cell, :, 1] += values.sum(dim=0)
        cell_stats[cell, :, 2] += values.square().sum(dim=0)


def _render_cells(cell_stats):
    names = ("stay_per_exposure", "lt_rate", "hlt_rate", "negative_rate")
    report = {}
    for cell, cell_name in enumerate(("control", "treatment")):
        report[cell_name] = {}
        for metric, name in enumerate(names):
            count, total, total_square = cell_stats[cell, metric]
            mean = total / count
            variance = (total_square - total.square() / count) / (count - 1.0)
            report[cell_name][name] = {
                "users": int(count),
                "mean": float(mean),
                "variance": float(variance),
            }
    return report


def combine_tensor_ab(control_report, treatment_report):
    """Combine stable control/treatment cells from two common-random worlds."""
    report = {}
    control_cell = control_report["experiment_cells"]["control"]
    treatment_cell = treatment_report["experiment_cells"]["treatment"]
    for name in control_cell:
        control = control_cell[name]
        treatment = treatment_cell[name]
        difference = treatment["mean"] - control["mean"]
        standard_error = sqrt(
            control["variance"] / control["users"]
            + treatment["variance"] / treatment["users"]
        )
        z_score = difference / max(standard_error, 1e-12)
        report[name] = {
            "control_mean": control["mean"],
            "treatment_mean": treatment["mean"],
            "relative_lift": difference / control["mean"],
            "standard_error": standard_error,
            "confidence_interval": (
                difference - 1.96 * standard_error,
                difference + 1.96 * standard_error,
            ),
            "p_value": erfc(abs(z_score) / sqrt(2.0)),
        }
    return report


def _simulate_batches(config, policy, generator, device):
    totals = torch.zeros(10, dtype=torch.float64, device=device)
    cell_stats = torch.zeros(2, 4, 3, dtype=torch.float64, device=device)
    for offset in range(0, config.users, config.batch_users):
        users = min(config.batch_users, config.users - offset)
        user_ids = torch.arange(offset, offset + users, device=device, dtype=torch.int64)
        trigger_bucket = torch.remainder(
            user_ids * 1_103_515_245 + 12_345, 2**31
        ).float() / float(2**31)
        eligible = (trigger_bucket < policy.eligible_fraction).float()[:, None]
        interest = torch.nn.functional.normalize(
            torch.randn(users, config.topics, generator=generator, device=device), dim=1
        )
        observed_interest = torch.nn.functional.normalize(
            interest
            + policy.observation_noise
            * torch.randn(interest.shape, generator=generator, device=device),
            dim=1,
        )
        satisfaction = torch.zeros(users, device=device)
        fatigue = torch.zeros(users, device=device)
        active = torch.ones(users, dtype=torch.bool, device=device)
        sessions = torch.ones(users, device=device)
        returned_sessions = torch.zeros(users, device=device)
        user_metrics = torch.zeros(users, 5, device=device)
        for _ in range(config.steps):
            topics = torch.nn.functional.normalize(
                torch.randn(
                    users,
                    config.candidates,
                    config.topics,
                    generator=generator,
                    device=device,
                ),
                dim=2,
            )
            quality = torch.rand(users, config.candidates, generator=generator, device=device)
            freshness = torch.rand(users, config.candidates, generator=generator, device=device)
            choice = _choose(
                policy,
                eligible,
                user_ids,
                observed_interest,
                fatigue,
                topics,
                quality,
                freshness,
            )
            batch_index = torch.arange(users, device=device)
            selected_topic = topics[batch_index, choice]
            selected_quality = quality[batch_index, choice]
            true_affinity = (selected_topic * interest).sum(dim=1)
            stay, long_view, hlt, like, negative, played, play_draw = _sample_response(
                generator,
                active,
                fatigue,
                satisfaction,
                true_affinity,
                selected_quality,
                device,
            )
            user_metrics += torch.stack(
                (active, stay, long_view, hlt, negative), dim=1
            ).float()
            totals += torch.stack(
                (
                    active.sum(),
                    stay.sum(),
                    long_view.sum(),
                    hlt.sum(),
                    like.sum(),
                    negative.sum(),
                    (play_draw if config.count_inactive_play_bug else played).sum(),
                    (stay >= 3.0).sum(),
                    sessions.sum() * 0.0,
                    returned_sessions.sum() * 0.0,
                )
            ).to(torch.float64)
            engagement = long_view.float() + like.float()
            satisfaction = torch.clamp(
                0.82 * satisfaction + 0.10 * engagement - 0.24 * negative.float(),
                -1.0,
                1.0,
            )
            fatigue = torch.clamp(0.72 * fatigue + 0.08 * long_view.float(), 0.0, 1.0)
            update = long_view.float()[:, None]
            interest = torch.nn.functional.normalize(
                interest * (1.0 - 0.10 * update) + selected_topic * 0.10 * update,
                dim=1,
            )
            observed_interest = torch.nn.functional.normalize(
                observed_interest * (1.0 - policy.realtime_interest_rate * update)
                + selected_topic * policy.realtime_interest_rate * update,
                dim=1,
            )
            leave = (
                torch.rand(users, generator=generator, device=device)
                < torch.sigmoid(-3.4 - 1.2 * satisfaction + 1.7 * fatigue)
            ) & active
            returned = leave & (
                torch.rand(users, generator=generator, device=device)
                < torch.sigmoid(1.0 + 1.6 * satisfaction - 1.1 * fatigue)
            )
            returned_sessions += returned
            sessions += returned
            active &= ~leave | returned
        totals[8] += sessions.sum()
        totals[9] += returned_sessions.sum()
        _accumulate_cells(cell_stats, user_ids, user_metrics)
    return totals, cell_stats


def run_tensor_feed(
    config: TensorFeedConfig,
    policy: TensorPolicy,
) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.current_device()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or 0)
    _sync(device)
    started = perf_counter()
    totals, cell_stats = _simulate_batches(config, policy, generator, device)
    _sync(device)
    seconds = perf_counter() - started
    values = totals.cpu().numpy()
    exposures = max(values[0], 1.0)
    report = {
        "config": asdict(config),
        "policy": asdict(policy),
        "experiment_cells": _render_cells(cell_stats.cpu()),
        "metrics": {
            "exposures": int(values[0]),
            "stay_per_exposure": float(values[1] / exposures),
            "lt_rate": float(values[2] / exposures),
            "hlt_rate": float(values[3] / exposures),
            "like_rate": float(values[4] / exposures),
            "negative_rate": float(values[5] / exposures),
            "play_rate": float(values[6] / exposures),
            "play_3s_rate": float(values[7] / exposures),
            "sessions_per_user": float(values[8] / config.users),
            "returned_sessions_per_user": float(values[9] / config.users),
        },
        "performance": {
            "seconds": seconds,
            "users_per_second": config.users / seconds,
            "requests_per_second": values[0] / seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device.index or 0))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    return report
