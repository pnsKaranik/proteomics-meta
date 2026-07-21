"""Co-expression / partial-correlation network construction, clustering and centrality."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.mixture import GaussianMixture

from .capabilities import LOUVAIN_AVAILABLE, community_louvain, get_logger

if TYPE_CHECKING:
    from .config import PipelineConfig

logger = get_logger(__name__)


def build_glasso_network(
    X_norm: np.ndarray,
    protein_list: list,
    threshold: float = 0.10,
) -> nx.Graph:
    """
    GraphicalLasso precision matrix → partial correlations → network.

    Routing logic (avoids sklearn CV warnings):
      n_samples < 4  → Spearman correlation (GLasso/CV impossible)
      4 ≤ n < 10     → LedoitWolf (no CV needed, stable shrinkage estimator)
      n ≥ 10         → GraphicalLassoCV (full cross-validated alpha selection)

    Falls back down the chain on any failure.
    """
    n_samples = X_norm.shape[1]
    logger.info("Building network (n=%d proteins, n=%d samples) …",
                X_norm.shape[0], n_samples)

    # ── Ultra-low sample: Spearman is the only reliable option ───────────────
    if n_samples < 4:
        logger.info("n_samples=%d < 4: using Spearman correlation (threshold=0.80).", n_samples)
        return build_spearman_network(X_norm, protein_list, threshold=0.80)

    # ── Low sample: LedoitWolf shrinkage — no CV, always converges ──────────
    if n_samples < 10:
        logger.info("n_samples=%d < 10: using LedoitWolf (no CV).", n_samples)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lw   = LedoitWolf().fit(X_norm.T)
            prec = np.linalg.inv(lw.covariance_ + np.eye(lw.covariance_.shape[0]) * 1e-6)
        except Exception as exc:
            logger.warning("LedoitWolf failed (%s); using Spearman.", exc)
            return build_spearman_network(X_norm, protein_list, threshold=0.80)
    else:
        # ── Full: GraphicalLassoCV ───────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gl   = GraphicalLassoCV(cv=min(5, n_samples - 1)).fit(X_norm.T)
            prec = gl.precision_
        except Exception as exc:
            logger.warning("GraphicalLassoCV failed (%s); trying LedoitWolf.", exc)
            try:
                lw   = LedoitWolf().fit(X_norm.T)
                prec = np.linalg.inv(lw.covariance_ + np.eye(lw.covariance_.shape[0]) * 1e-6)
            except Exception as exc2:
                logger.warning("LedoitWolf also failed (%s); using Spearman.", exc2)
                return build_spearman_network(X_norm, protein_list, threshold=0.70)

    d_diag = np.maximum(np.diag(prec), 1e-9)
    pcorr  = -prec / np.sqrt(np.outer(d_diag, d_diag))
    np.fill_diagonal(pcorr, 0)
    # Replace any NaN/Inf that can arise from near-singular precision matrices
    pcorr  = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)

    mask   = np.triu(np.abs(pcorr) > threshold, k=1)
    ri, ci = np.where(mask)
    G = nx.Graph()
    if len(ri):
        pa = np.array(protein_list)
        G.add_weighted_edges_from(zip(pa[ri], pa[ci], np.abs(pcorr[ri, ci]), strict=False))
    logger.info("Network: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    return G, pcorr


def build_spearman_network(
    X_norm: np.ndarray,
    protein_list: list,
    threshold: float = 0.80,
    max_edges: int = 10_000,
) -> nx.Graph:
    """
    Spearman correlation network for low-sample mode.
    High threshold keeps only strong relationships.
    max_edges cap prevents centrality from running for hours on dense graphs.
    """
    logger.info("Building Spearman network (threshold=%.2f) …", threshold)
    corr = np.array(pd.DataFrame(X_norm).T.corr(method="spearman").values)
    np.fill_diagonal(corr, 0)

    abs_corr = np.abs(corr)
    mask     = np.triu(abs_corr > threshold, k=1)
    ri, ci   = np.where(mask)

    # If still too many edges, keep only the strongest ones
    if len(ri) > max_edges:
        logger.warning(
            "Spearman: %d edges exceeds cap=%d; keeping top-%d by weight.",
            len(ri), max_edges, max_edges,
        )
        weights  = abs_corr[ri, ci]
        top_idx  = np.argsort(weights)[-max_edges:]
        ri, ci   = ri[top_idx], ci[top_idx]

    G = nx.Graph()
    if len(ri):
        pa = np.array(protein_list)
        G.add_weighted_edges_from(zip(pa[ri], pa[ci], abs_corr[ri, ci], strict=False))
    logger.info("Spearman network: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    return G, corr


def build_partial_corr_network(
    X_norm: np.ndarray,
    protein_list: list,
    threshold: float = 0.10,
) -> nx.Graph:
    """Original LedoitWolf partial correlation network (full mode)."""
    lw     = LedoitWolf().fit(X_norm.T)
    prec   = np.linalg.inv(lw.covariance_ + np.eye(lw.covariance_.shape[0]) * 1e-6)
    d_diag = np.maximum(np.diag(prec), 1e-9)
    pcorr  = -prec / np.sqrt(np.outer(d_diag, d_diag))
    np.fill_diagonal(pcorr, 0)
    mask   = np.triu(np.abs(pcorr) > threshold, k=1)
    ri, ci = np.where(mask)
    G = nx.Graph()
    if len(ri):
        pa = np.array(protein_list)
        G.add_weighted_edges_from(zip(pa[ri], pa[ci], np.abs(pcorr[ri, ci]), strict=False))
    logger.info("Partial-corr network: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    return G, pcorr


def build_network(
    X_norm: np.ndarray,
    protein_list: list,
    mode: str,
    threshold: float = 0.10,
):
    """Route to the appropriate network builder based on pipeline mode."""
    if mode == "ultra_sparse":
        return build_glasso_network(X_norm, protein_list, threshold)
    elif mode == "low_sample":
        return build_glasso_network(X_norm, protein_list, threshold)
    elif mode == "moderate":
        # Use GLasso (more robust than partial corr at n<30)
        return build_glasso_network(X_norm, protein_list, threshold)
    else:
        return build_partial_corr_network(X_norm, protein_list, threshold)


def cluster_proteins_adaptive(
    latent: np.ndarray,
    protein_list: list,
    graph: nx.Graph,
    config: PipelineConfig,
    mode: str,
) -> np.ndarray:
    """
    Route to appropriate clustering based on pipeline mode.

    ultra_sparse → Hierarchical (k=2)
    low_sample   → GMM (max k=4)
    moderate     → GMM (max k=10)
    full         → Louvain (if available) else GMM
    """
    n = len(protein_list)

    if mode == "ultra_sparse":
        logger.info("Clustering: Hierarchical (k=2, ultra-sparse mode) …")
        if n < 2:
            return np.zeros(n, dtype=int)
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances
            # Euclidean distance — always symmetric, no NaN issues
            dist = pairwise_distances(latent, metric="euclidean")
            dist = (dist + dist.T) / 2.0   # enforce perfect symmetry
            np.fill_diagonal(dist, 0.0)
            cond = squareform(dist, checks=False)
            Z    = linkage(cond, method="ward")
            return fcluster(Z, t=2, criterion="maxclust") - 1
        except Exception as exc:
            logger.warning("Hierarchical clustering failed: %s; KMeans fallback.", exc)
            try:
                from sklearn.cluster import KMeans
                return KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(latent)
            except Exception:
                return np.zeros(n, dtype=int)

    elif mode == "low_sample":
        max_k = min(4, n - 1)
        n_c   = int(config.n_clusters) if config.n_clusters else max(2, min(max_k, n // 3))
        logger.info("Clustering: GMM (%d components, low-sample mode) …", n_c)
        return GaussianMixture(n_components=n_c, random_state=config.random_seed,
                               n_init=3).fit_predict(latent)

    elif mode == "moderate":
        max_k = min(10, n - 1)
        n_c   = int(config.n_clusters) if config.n_clusters else min(max_k, max(5, n // 50))
        logger.info("Clustering: GMM (%d components, moderate mode) …", n_c)
        return GaussianMixture(n_components=n_c, random_state=config.random_seed,
                               n_init=3).fit_predict(latent)

    else:
        # Full mode: Louvain preferred
        if config.use_louvain and LOUVAIN_AVAILABLE and graph.number_of_edges() > 0:
            logger.info("Clustering: Louvain …")
            partition = community_louvain.best_partition(graph, random_state=config.random_seed)
            labels    = np.array([partition.get(p, -1) for p in protein_list])
            logger.info("Louvain: %d communities.", len(set(labels)) - (1 if -1 in labels else 0))
            return labels
        n_c = int(config.n_clusters) if config.n_clusters else min(15, max(5, n // 50))
        logger.info("Clustering: GMM (%d components, full mode) …", n_c)
        return GaussianMixture(n_components=n_c, random_state=config.random_seed,
                               n_init=3).fit_predict(latent)


def calculate_network_centrality(graph: nx.Graph, protein_list: list) -> pd.DataFrame:
    logger.info("Network centrality …")
    if graph.number_of_edges() == 0:
        return pd.DataFrame(
            {"Betweenness_Centrality": 0.0, "Eigenvector_Centrality": 0.0},
            index=protein_list,
        )
    n           = graph.number_of_nodes()
    k_sample    = min(100, n) if n > 500 else None
    betweenness = nx.betweenness_centrality(graph, k=k_sample, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=600, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector did not converge; using degree centrality.")
        eigenvector = nx.degree_centrality(graph)
    return pd.DataFrame({
        "Betweenness_Centrality": pd.Series(betweenness),
        "Eigenvector_Centrality": pd.Series(eigenvector),
    }).reindex(protein_list).fillna(0.0)


def build_ppi_network(df_ppi: pd.DataFrame, df_final: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from STRING interactions, annotated with master scores."""
    G_ppi = nx.Graph()
    if df_ppi.empty:
        return G_ppi
    score_map = dict(zip(df_final["Gene_Symbol"], df_final.get("Master_Score", pd.Series()), strict=False))
    class_map = dict(zip(df_final["Gene_Symbol"], df_final.get("Anomaly_Class", pd.Series()), strict=False))
    for _, row in df_ppi.iterrows():
        G_ppi.add_edge(row["gene_a"], row["gene_b"], weight=float(row["score"]))
    for node in G_ppi.nodes():
        G_ppi.nodes[node]["master_score"] = float(score_map.get(node, 0))
        G_ppi.nodes[node]["anomaly_class"] = str(class_map.get(node, "Unknown"))
    return G_ppi
