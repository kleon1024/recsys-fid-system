"""Run main-Feed extraction, delayed labels, realtime features, and parity audit."""

from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from ..media.materializer import MediaAsset, MediaFeatureMaterializer
from .consistency import FullPathConsistencyAuditor
from .contracts import FeedAction, FeedImpression, ViewerBehaviorEvent
from .samples import PoiFeedJoiner
from .streaming import ViewerFeatureOperator


VERSIONS = {
    "media": "media-encoder-v1",
    "feature": "poi-feed-features-v1",
    "model": "feed-transformer-mmoe-v1",
    "index": "poi-video-index-v1",
}


def synthetic_events(
    seed: int = 71, impressions_count: int = 1_200
) -> tuple[list[FeedImpression], list[FeedAction], ViewerFeatureOperator]:
    rng = np.random.default_rng(seed)
    base_time = 1_700_000_000
    operator = ViewerFeatureOperator()
    for index in range(2_000):
        event_time = base_time - 604_800 + index * 240
        event = ViewerBehaviorEvent(
            event_id=f"history-{index}",
            viewer_id=int(rng.integers(160)),
            category_id=int(rng.integers(8)),
            action=str(rng.choice(("view", "long_view", "anchor_click", "favorite"))),
            event_time=event_time,
            received_at=event_time + int(rng.integers(30)),
        )
        operator.ingest(event, watermark=event.received_at - 60)
    impressions: list[FeedImpression] = []
    actions: list[FeedAction] = []
    action_rates = {
        "long_view": 0.34,
        "anchor_click": 0.12,
        "detail_view": 0.07,
        "favorite": 0.035,
        "order": 0.012,
        "negative_feedback": 0.02,
    }
    for index in range(impressions_count):
        event_time = base_time + index * 30
        anchored = rng.random() < 0.36
        features = rng.normal(0.0, 1.0, size=10)
        features[2] += 0.7 if anchored else -0.2
        impression = FeedImpression(
            impression_id=f"imp-{index}",
            viewer_id=int(rng.integers(160)),
            author_id=int(rng.integers(220)),
            video_id=10_000 + index,
            poi_id=20_000 + int(rng.integers(300)) if anchored else None,
            category_id=int(rng.integers(8)),
            event_time=event_time,
            base_features=tuple(float(value) for value in features),
            media_version=VERSIONS["media"],
            feature_version=VERSIONS["feature"],
            model_version=VERSIONS["model"],
            index_version=VERSIONS["index"],
        )
        impressions.append(impression)
        if not anchored:
            continue
        quality = 1.0 / (1.0 + np.exp(-features[2]))
        for task, base_rate in action_rates.items():
            probability = min(base_rate * (0.55 + quality), 0.95)
            if rng.random() >= probability:
                continue
            delay = int(rng.integers(20, 240))
            actions.append(
                FeedAction(
                    action_id=f"{task}-{index}",
                    impression_id=impression.impression_id,
                    action=task,
                    event_time=event_time + delay,
                    received_at=event_time + delay + int(rng.integers(60)),
                )
            )
    return impressions, actions, operator


def run_demo() -> dict[str, object]:
    impressions, actions, operator = synthetic_events()
    rng = np.random.default_rng(99)
    media = MediaFeatureMaterializer(8).materialize(
        MediaAsset(
            10_000,
            rng.normal(size=(4, 8)).astype(np.float32),
            rng.normal(size=8).astype(np.float32),
            impressions[0].event_time,
        )
    )
    watermark = max(value.event_time for value in impressions) + 700_000
    joined = PoiFeedJoiner().build(impressions, actions, operator, watermark)
    features = np.asarray([value.features for value in joined.examples])
    positives = set(range(20))
    audit = FullPathConsistencyAuditor().audit(
        joined.examples,
        VERSIONS,
        features,
        features.copy(),
        positives,
        set(range(30)),
        set(range(19)),
    )
    label_rates = {
        task: float(np.mean([value.labels[task] for value in joined.examples]))
        for task in joined.examples[0].labels
    }
    return {
        "main_impressions": joined.main_impressions,
        "anchored_impressions": joined.anchored_impressions,
        "examples": len(joined.examples),
        "label_rates": label_rates,
        "stream": asdict(operator.report()),
        "media": {
            "version": media.encoder_version,
            "embedding_norm": float(np.linalg.norm(media.content_embedding)),
            "content_hash_length": len(media.content_hash),
        },
        "consistency": asdict(audit),
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
