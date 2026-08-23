"""Platform LT conversion over experiment-measured metrics only."""

from __future__ import annotations

from dataclasses import asdict

from .contracts import DEFAULT_LT_CONFIG, LTMetricBreakdown, LTMetricConfig, LTMetricVector


class LTMetricContainer:
    def __init__(self, config: LTMetricConfig = DEFAULT_LT_CONFIG) -> None:
        self.config = config

    def _rate(self, name: str) -> float:
        try:
            return self.config.rates[name].unit_value
        except KeyError as error:
            raise ValueError(f"missing LT metric exchange rate: {name}") from error

    def manifest(self) -> dict[str, object]:
        return asdict(self.config)

    def evaluate(self, metrics: LTMetricVector) -> LTMetricBreakdown:
        stay = metrics.stay_minutes * self._rate("stay_minute")
        active_days = metrics.active_days * self._rate("active_day")
        accepted_commercialization = (
            metrics.accepted_commercialization_value
            * self._rate("accepted_commercialization_unit")
        )
        return LTMetricBreakdown(
            self.config.version,
            stay,
            active_days,
            accepted_commercialization,
            stay + active_days + accepted_commercialization,
        )
