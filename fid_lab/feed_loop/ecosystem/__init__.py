"""Coupled consumer Feed and creator supply simulation."""

from .contracts import EcosystemConfig
from .runner import run_ecosystem

__all__ = ["EcosystemConfig", "run_ecosystem"]
