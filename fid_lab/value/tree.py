"""Business-specific Value Tree formulas kept outside the LT metric contract."""

from __future__ import annotations

from .contracts import (
    BUSINESS_TREE_WEIGHTS,
    BusinessValueBreakdown,
    BusinessValueSignals,
)


class BusinessValueTree:
    version = "business-value-trees-v1"

    @staticmethod
    def _local_consumption(signals: BusinessValueSignals) -> float:
        weights = BUSINESS_TREE_WEIGHTS
        terminal = (
            (signals.poi_favorite, weights["poi_favorite"]),
            (signals.poi_detail, weights["poi_detail"]),
            (signals.anchor_click, weights["anchor_click"]),
        )
        return next((value for happened, value in terminal if happened), 0.0)

    def evaluate(self, signals: BusinessValueSignals) -> BusinessValueBreakdown:
        weights = BUSINESS_TREE_WEIGHTS
        feed = (
            weights["quality_view"] * signals.quality_view
            + weights["meaningful_engagement"] * signals.meaningful_engagements
            + weights["negative_feedback"] * signals.negative_feedback
        )
        local_consumption = self._local_consumption(signals)
        local_transaction = (
            weights["closed_loop_payment"] * signals.closed_loop_payment
            + weights["open_loop_conversion"]
            * signals.open_loop_verified_conversion
            + weights["contribution_margin"] * signals.contribution_margin
        )
        return BusinessValueBreakdown(
            self.version,
            feed,
            local_consumption,
            local_transaction,
            weights["quality_adjusted_supply"] * signals.quality_adjusted_supply,
            weights["ad_value"] * signals.ad_value
            + weights["live_value"] * signals.live_value,
        )
