"""Distinct recommendation surfaces sharing model infrastructure."""

from .contracts import SURFACE_SPECS, SurfaceSpec, TaskSpec
from .experiment import SurfaceReport, run_surface_suite
from .model import build_surface_model

__all__ = [
    "SURFACE_SPECS",
    "SurfaceReport",
    "SurfaceSpec",
    "TaskSpec",
    "build_surface_model",
    "run_surface_suite",
]
