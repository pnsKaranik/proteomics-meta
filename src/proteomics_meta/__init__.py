"""Sample-aware proteomics meta-analysis engine.

Public entry points:
    from proteomics_meta import run_pipeline, PipelineConfig
"""
from __future__ import annotations

from typing import Any

from .config import MODE_DESCRIPTIONS, PipelineConfig, get_pipeline_mode

__version__ = "0.2.0"
__all__ = [
    "PipelineConfig",
    "get_pipeline_mode",
    "MODE_DESCRIPTIONS",
    "run_pipeline",
    "run_pipeline_initial",
    "__version__",
]

# The engine pulls in TensorFlow, so it is imported lazily to keep lightweight
# consumers (config, stats, networks) free of the heavy dependency.
_LAZY = {"run_pipeline", "run_pipeline_initial"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
