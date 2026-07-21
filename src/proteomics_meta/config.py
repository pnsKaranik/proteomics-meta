"""Sample-aware pipeline configuration and mode routing."""
from __future__ import annotations

from dataclasses import dataclass, field

from .capabilities import ADVANCED_LIBS, UMAP_AVAILABLE


def get_pipeline_mode(n_samples: int) -> str:
    """
    Returns one of: "ultra_sparse" | "low_sample" | "moderate" | "full"

    ultra_sparse : n_samples < 5
    low_sample   : 5  ≤ n_samples < 15
    moderate     : 15 ≤ n_samples < 30
    full         : n_samples ≥ 30
    """
    if n_samples < 5:
        return "ultra_sparse"
    elif n_samples < 15:
        return "low_sample"
    elif n_samples < 30:
        return "moderate"
    return "full"


MODE_DESCRIPTIONS = {
    "ultra_sparse": {
        "label":       "Ultra-sparse (n<5)",
        "network":     "GLasso",
        "clustering":  "Hierarchical (k=2)",
        "viz":         "PCA",
        "pvalues":     "Bootstrap",
        "trajectory":  "Disabled",
        "color":       "badge-orange",
    },
    "low_sample": {
        "label":       "Low-sample (5–14)",
        "network":     "GLasso",
        "clustering":  "GMM (k≤4)",
        "viz":         "UMAP" if UMAP_AVAILABLE else "PCA",
        "pvalues":     "Bootstrap",
        "trajectory":  "Spearman",
        "color":       "badge-orange",
    },
    "moderate": {
        "label":       "Moderate (15–29)",
        "network":     "GLasso + Spearman",
        "clustering":  "GMM (k≤10)",
        "viz":         "UMAP" if UMAP_AVAILABLE else "PCA",
        "pvalues":     "BH FDR",
        "trajectory":  "Spearman",
        "color":       "badge-blue",
    },
    "full": {
        "label":       "Full (n≥30)",
        "network":     "Partial correlation",
        "clustering":  "Louvain / GMM",
        "viz":         "PHATE" if ADVANCED_LIBS else "UMAP/PCA",
        "pvalues":     "BH FDR",
        "trajectory":  "Full",
        "color":       "badge-green",
    },
}


@dataclass
class PipelineConfig:
    work_dir: str        = "Meta_Analysis_Results"
    iterations: int      = 3
    epochs: int          = 60
    latent_dim: int      = 10
    learning_rate: float = 0.002
    beta_vae: float      = 1.0
    n_clusters: int | None = None
    gene_sets: list      = field(default_factory=lambda: [
        "KEGG_2021_Human", "GO_Biological_Process_2021"
    ])
    use_louvain: bool    = True
    random_seed: int     = 42
    pcorr_threshold: float = 0.10
    pips_alpha: float    = 0.5
    n_bootstrap: int     = 500       # bootstrap CI iterations
    jackknife: bool      = True      # compute jackknife stability

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
