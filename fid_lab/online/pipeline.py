"""End-to-end online recommendation orchestration with stage-level evidence."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from .catalog import ItemCatalog
from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from .domain import RecommendationResult, RequestContext, StageTrace
from .features import OnlineFeatureService
from .stages.mixing import MixedRanker
from .stages.policy import ConstrainedPolicyOptimizer, RankingRuleEngine
from .stages.ranking import CoarseRanker, EligibilityFilter, FineRanker, ValueTree
from .stages.retrieval import FreshRecall, LocalVikingIndex, PopularRecall, RecallMerger


T = TypeVar("T")


class RecommendationPipeline:
    def __init__(
        self, catalog: ItemCatalog, config: PipelineConfig = DEFAULT_PIPELINE_CONFIG
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.viking = LocalVikingIndex(catalog)
        self.popular = PopularRecall(catalog)
        self.fresh = FreshRecall(catalog)
        self.merger = RecallMerger(config.recall)
        self.filter = EligibilityFilter(catalog)
        self.features = OnlineFeatureService(catalog)
        self.coarse = CoarseRanker(catalog)
        self.fine = FineRanker(catalog, ValueTree(config.value_tree))
        self.rules = RankingRuleEngine(catalog, config.rules, config.fresh_age_hours)
        self.policy = ConstrainedPolicyOptimizer(
            catalog, config.policy, config.fresh_age_hours
        )
        self.mixer = MixedRanker(catalog, config.mix)

    @staticmethod
    def timed(
        stage: str, input_count: int, operation: Callable[[], T]
    ) -> tuple[T, StageTrace]:
        started = perf_counter()
        output = operation()
        output_count = len(output) if hasattr(output, "__len__") else 0
        trace = StageTrace(
            stage=stage,
            input_count=input_count,
            output_count=output_count,
            latency_ms=round((perf_counter() - started) * 1000.0, 3),
        )
        return output, trace

    def recommend(self, request: RequestContext) -> RecommendationResult:
        traces: list[StageTrace] = []
        limits = self.config.limits
        route_hits, trace = self.timed(
            "recall",
            len(self.catalog.items),
            lambda: {
                "viking": self.viking.recall(request, limits.vector_recall),
                "popular": self.popular.recall(request, limits.popular_recall),
                "fresh": self.fresh.recall(request, limits.fresh_recall),
            },
        )
        trace = StageTrace(trace.stage, trace.input_count, sum(map(len, route_hits.values())), trace.latency_ms)
        traces.append(trace)
        candidates, trace = self.timed(
            "recall_merge", sum(map(len, route_hits.values())),
            lambda: self.merger.merge(route_hits, limits.merged_recall),
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "eligibility", len(candidates), lambda: self.filter.apply(request, candidates)
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "feature_join", len(candidates), lambda: self.features.encode(request, candidates)
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "coarse_rank", len(candidates), lambda: self.coarse.rank(request, candidates, limits.coarse_rank)
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "fine_rank_value_tree", len(candidates),
            lambda: self.fine.rank(request, candidates, limits.fine_rank),
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "ranking_rules", len(candidates), lambda: self.rules.apply(candidates)
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "copp_policy", len(candidates),
            lambda: self.policy.select(candidates, limits.policy_pool),
        )
        traces.append(trace)
        candidates, trace = self.timed(
            "mixed_rank", len(candidates), lambda: self.mixer.mix(request, candidates)
        )
        traces.append(trace)
        return RecommendationResult(
            request_id=request.request_id,
            items=tuple(candidates),
            traces=tuple(traces),
            artifact_versions={
                "pipeline": self.config.version,
                "catalog": self.catalog.version,
                "vector_index": self.viking.version,
                "online_features": self.features.version,
                "fid_layout": self.features.codec.version.value,
                "coarse_model": self.coarse.version,
                "fine_model": self.fine.version,
                "copp_implementation": self.policy.implementation,
            },
        )
