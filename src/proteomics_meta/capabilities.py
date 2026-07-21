"""Centralized optional-dependency detection and shared logging.

Every module imports capability flags and library handles from here rather than
performing its own guarded imports, so the availability logic lives in one place
and the heavy scientific stack degrades gracefully when a package is missing.
"""
from __future__ import annotations

import logging


def _configure_root() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("proteomics_meta")


logger = _configure_root()


def get_logger(name: str) -> logging.Logger:
    return logger.getChild(name.rsplit(".", 1)[-1])


# ── Louvain community detection ────────────────────────────────────────────────
try:
    import community as community_louvain

    LOUVAIN_AVAILABLE = True
except ImportError:
    community_louvain = None
    LOUVAIN_AVAILABLE = False

# ── SHAP ───────────────────────────────────────────────────────────────────────
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

# ── UMAP ───────────────────────────────────────────────────────────────────────
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False


class _GoStub:
    """Minimal stand-in for plotly.graph_objects when advanced libs are absent."""

    Figure = type(
        "Figure",
        (),
        {"write_html": lambda *a, **k: None, "update_layout": lambda *a, **k: None},
    )


# ── Advanced visualization / enrichment stack ──────────────────────────────────
try:
    import gseapy as gp
    import kmapper as km
    import phate
    import plotly.express as px
    import plotly.graph_objects as go
    from adjustText import adjust_text  # noqa: F401

    ADVANCED_LIBS = True
except ImportError:
    phate = None
    km = None
    gp = None
    px = None
    go = _GoStub()
    ADVANCED_LIBS = False


__all__ = [
    "logger",
    "get_logger",
    "LOUVAIN_AVAILABLE",
    "SHAP_AVAILABLE",
    "UMAP_AVAILABLE",
    "ADVANCED_LIBS",
    "community_louvain",
    "shap",
    "umap",
    "phate",
    "km",
    "gp",
    "px",
    "go",
]
