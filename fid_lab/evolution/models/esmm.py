"""Entire-space click and post-click conversion model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ESMMOutput:
    click_logit: torch.Tensor
    conversion_given_click_logit: torch.Tensor
    pctr: torch.Tensor
    pcvr: torch.Tensor
    pctcvr: torch.Tensor


class ESMM(nn.Module):
    """Predict CTR and conditional CVR while training conversion in impression space."""

    def __init__(self, inputs: int, width: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(inputs, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.click_tower = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.cvr_tower = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, features: torch.Tensor) -> ESMMOutput:
        shared = self.shared(features)
        click_logit = self.click_tower(shared).squeeze(1)
        cvr_logit = self.cvr_tower(shared).squeeze(1)
        pctr = torch.sigmoid(click_logit)
        pcvr = torch.sigmoid(cvr_logit)
        return ESMMOutput(click_logit, cvr_logit, pctr, pcvr, pctr * pcvr)

    def entire_space_loss(self, features, click, conversion, mature_mask=None):
        if torch.any(conversion > click):
            raise ValueError("conversion label must imply click in the ESMM funnel")
        output = self(features)
        mask = torch.ones_like(click) if mature_mask is None else mature_mask.float()
        click_loss = nn.functional.binary_cross_entropy(
            output.pctr, click.float(), reduction="none"
        )
        conversion_loss = nn.functional.binary_cross_entropy(
            output.pctcvr, conversion.float(), reduction="none"
        )
        denominator = mask.sum().clamp_min(1.0)
        return ((click_loss + conversion_loss) * mask).sum() / denominator
