"""Tensor policy that changes only the ANN route scorer."""

from __future__ import annotations

from ...feed_loop.tensor_cascade import select_candidate
from ...feed_loop.tensor_policies import PERSONALIZED
from .features import category_centers, catalog_item_features, live_query_features
from .models.bundle import corpus_hash


class TensorPoiRetrievalPolicy:
    eligible_fraction = PERSONALIZED.eligible_fraction
    observation_noise = PERSONALIZED.observation_noise
    local_observation_noise = PERSONALIZED.local_observation_noise
    realtime_interest_rate = PERSONALIZED.realtime_interest_rate
    multi_queue = PERSONALIZED.multi_queue

    def __init__(self, bundle):
        self.bundle = bundle
        self.name = f"poi_ann_{bundle.name}"
        self._indexed = False
        self._category_centers = None

    def describe(self):
        return {
            "name": self.name,
            "stage": "retrieval_ann_route",
            "model": self.bundle.name,
            "corpus_sha256": self.bundle.corpus_sha256,
        }

    def _ensure_index(self, catalog):
        if self._indexed:
            return
        features = catalog_item_features(catalog)
        actual = corpus_hash(features)
        if actual != self.bundle.corpus_sha256:
            raise ValueError("POI retrieval model/catalog version mismatch")
        self.bundle.index(features)
        self._category_centers = category_centers(catalog)
        self._indexed = True

    def score_ann_pool(self, config, state, catalog, ann_pool, step):
        del config, step
        self._ensure_index(catalog)
        return self.bundle.score_pool(
            live_query_features(state, self._category_centers), ann_pool
        )

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        return select_candidate(
            PERSONALIZED, user_ids, state, candidates, device, step, config
        )
