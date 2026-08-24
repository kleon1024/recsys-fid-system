from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import ContentKind
from fid_lab.simulation.digital_twin.platform.lifecycle import (
    ContentLifecycle,
    LifecycleConfig,
    classify_lifecycle,
)


def test_lifecycle_separates_recent_hot_evergreen_and_expired_corpora():
    config = LifecycleConfig(
        ticks_per_day=10,
        recent_days=30,
        cold_start_days=2,
        hot_min_impressions=12,
        hot_engagement_rate=0.18,
    )
    lifecycle = classify_lifecycle(
        active=torch.tensor([False, True, True, True, True, True, True]),
        content_kind=torch.tensor([
            int(ContentKind.SHORT_VIDEO),
            int(ContentKind.SHORT_VIDEO),
            int(ContentKind.PHOTO),
            int(ContentKind.ARTICLE),
            int(ContentKind.CARD),
            int(ContentKind.SHORT_VIDEO),
            int(ContentKind.PRODUCT),
        ]),
        publish_time=torch.tensor([395, 395, 350, 350, 0, 0, 0]),
        evergreen_eligible=torch.tensor([
            False, False, False, False, True, False, False,
        ]),
        recent_impressions=torch.tensor([
            0.0, 0.0, 5.0, 20.0, 0.0, 0.0, 0.0,
        ]),
        recent_engagements=torch.tensor([
            0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0,
        ]),
        logical_time=400,
        config=config,
    )
    assert lifecycle.tolist() == [
        int(ContentLifecycle.RESERVED),
        int(ContentLifecycle.COLD_START),
        int(ContentLifecycle.RECENT),
        int(ContentLifecycle.HOT),
        int(ContentLifecycle.EVERGREEN),
        int(ContentLifecycle.EXPIRED),
        int(ContentLifecycle.EVERGREEN),
    ]
