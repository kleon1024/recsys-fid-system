"""Ensemble uncertainty, paired structural noise, and free-running rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import BINARY_ACTIONS, STOCHASTIC_ACTIONS, WorldModelConfig
from .network import NeuralSCM


@dataclass(frozen=True)
class StructuralNoise:
    latent: torch.Tensor
    mixture: torch.Tensor
    stay: torch.Tensor
    actions: torch.Tensor

    @classmethod
    def generate(cls, rows: int, config: WorldModelConfig, device, seed: int):
        generator = torch.Generator(device=device).manual_seed(seed)
        return cls(
            torch.randn(rows, config.latent_dim, generator=generator, device=device),
            torch.rand(rows, generator=generator, device=device),
            torch.randn(rows, generator=generator, device=device),
            torch.rand(rows, len(STOCHASTIC_ACTIONS), generator=generator, device=device),
        )


class WorldModelEnsemble(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        members = []
        for index in range(config.ensemble_members):
            with torch.random.fork_rng():
                torch.manual_seed(config.seed + index * 10_007)
                members.append(NeuralSCM(config))
        self.members = nn.ModuleList(members)

    @torch.inference_mode()
    def predict(self, batch):
        member_probabilities = []
        member_stay = []
        zero_noise = torch.zeros(
            len(batch["labels"]), self.config.latent_dim,
            device=batch["labels"].device,
        )
        for member in self.members:
            member.eval()
            output = member(batch, latent_noise=zero_noise)
            member_probabilities.append(self._all_probabilities(
                output, batch["selected_features"]
            ))
            member_stay.append((
                torch.softmax(output.stay_mixture_logits, dim=1) * output.stay_mean
            ).sum(dim=1))
        probabilities = torch.stack(member_probabilities)
        stay = torch.stack(member_stay)
        return {
            "probability_mean": probabilities.mean(dim=0),
            "probability_std": probabilities.std(dim=0),
            "stay_mean": stay.mean(dim=0),
            "stay_std": stay.std(dim=0),
        }

    @staticmethod
    def _all_probabilities(output, selected_features):
        modeled = {
            action.name: torch.sigmoid(output.logits[action.name])
            for action in STOCHASTIC_ACTIONS
        }
        scale = torch.exp(output.stay_log_scale).clamp_min(1e-3)
        normal = torch.distributions.Normal(output.stay_mean, scale)
        mixture = torch.softmax(output.stay_mixture_logits, dim=1)
        log_181 = torch.log(torch.tensor(181.0, device=selected_features.device))
        duration = torch.expm1(selected_features[:, 12].clamp(0.0, 1.0) * log_181)
        def survival(seconds):
            threshold = torch.log1p(seconds.clamp_min(0.0)) / log_181
            component_survival = (1.0 - normal.cdf(threshold[:, None])).clamp(0.0, 1.0)
            return modeled["play"] * (mixture * component_survival).sum(dim=1)
        modeled.update({
            "play_3s": survival(torch.full_like(duration, 3.0)),
            "complete_play": survival(0.95 * duration),
            "long_view": survival(torch.minimum(torch.full_like(duration, 18.0), duration)),
            "quality_long_view": survival(torch.minimum(torch.full_like(duration, 30.0), duration)),
        })
        return torch.stack([modeled[action.name] for action in BINARY_ACTIONS], dim=1)

    @torch.inference_mode()
    def sample_members(self, batch, noise: StructuralNoise):
        return [
            member.sample(
                batch, noise.latent, noise.mixture, noise.stay, noise.actions
            )
            for member in self.members
        ]

    @torch.inference_mode()
    def rollout(self, batch, steps: int, seed: int):
        sequence = batch["sequence"].clone()
        events = []
        for step in range(steps):
            current = {**batch, "sequence": sequence}
            noise = StructuralNoise.generate(
                len(sequence), self.config, sequence.device, seed + step
            )
            samples = self.sample_members(current, noise)
            event = torch.stack([sample["event"] for sample in samples]).float().mean(dim=0)
            events.append(event)
            sequence = torch.roll(sequence, shifts=-1, dims=1)
            sequence[:, -1] = event
        return torch.stack(events, dim=1)
