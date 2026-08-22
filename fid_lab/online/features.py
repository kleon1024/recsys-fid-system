"""Online feature join using the same FID registry as offline experiments."""

from __future__ import annotations

from ..fid import FidCodec, FidVersion
from ..schema import DEFAULT_SCHEMA, FeatureRegistry
from .catalog import CATEGORIES, ItemCatalog
from .domain import Candidate, RequestContext


COUNTRY_IDS = {"SG": 0, "US": 1, "GB": 2}


class OnlineFeatureService:
    version = "online-features-v1"

    def __init__(
        self,
        catalog: ItemCatalog,
        registry: FeatureRegistry = DEFAULT_SCHEMA,
        codec: FidCodec | None = None,
    ) -> None:
        self.catalog = catalog
        self.registry = registry
        self.codec = codec or FidCodec(FidVersion.V2)

    def encode(self, request: RequestContext, candidates: list[Candidate]) -> list[Candidate]:
        if request.country not in COUNTRY_IDS:
            raise ValueError(f"unregistered country: {request.country}")
        encoded: list[Candidate] = []
        for candidate in candidates:
            item = self.catalog.get(candidate.item_id)
            row = {
                "user_id": request.user_id,
                "age_bucket": request.user_id % 6,
                "item_id": item.item_id,
                "category": CATEGORIES.index(item.category),
                "country": COUNTRY_IDS[request.country],
                "device": request.device,
                "hour_bucket": request.hour_bucket,
            }
            fids, buckets = self.registry.encode_row(row, self.codec)
            encoded.append(candidate.update(feature_fids=fids, feature_buckets=buckets))
        return encoded
