"""Pure statistical routines: robust stats, bootstrap, consensus, differential expression."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm
from sklearn.preprocessing import RobustScaler

from .capabilities import get_logger

logger = get_logger(__name__)


def calculate_robust_stats(values: np.ndarray):
    """MAD Z-scores + one-tailed BH FDR."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    if mad == 0:
        mad = np.std(values) + 1e-9

    z_scores     = 0.6745 * (values - median) / mad
    raw_p_values = norm.sf(z_scores)

    try:
        bh_p = false_discovery_control(raw_p_values, method="bh")
    except Exception:
        n     = len(raw_p_values)
        order = np.argsort(raw_p_values)
        adj   = raw_p_values[order] * n / np.arange(1, n + 1)
        adj   = np.minimum.accumulate(adj[::-1])[::-1]
        bh_p  = np.empty(n)
        bh_p[order] = np.clip(adj, 0, 1)

    return z_scores, raw_p_values, bh_p


def calculate_bootstrap_pvalues(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Empirical p-values via permutation/bootstrap.
    Used in ultra_sparse and low_sample modes where BH FDR lacks power.

    p[i] = fraction of bootstrap samples where a random draw ≥ values[i]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    null   = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    p      = np.mean(null >= values[np.newaxis, :], axis=0)
    p      = np.clip(p, 1 / n_bootstrap, 1.0)
    logger.info("Bootstrap p-values computed (%d iterations).", n_bootstrap)
    return p


def compute_bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Parametric bootstrap CIs for the master score vector.

    Since we cannot re-run the full pipeline n_bootstrap times, we add
    Gaussian noise scaled to the observed score variance and recompute
    the score distribution. This gives empirical ± bounds that reflect
    the sensitivity of the scores to small perturbations.

    Returns dict with keys: 'mean', 'std', 'ci_low', 'ci_high'
    each an ndarray of shape (n_proteins,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n    = len(scores)
    noise_std = np.std(scores) * 0.05   # 5 % noise scale
    boot_scores = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        noise = rng.normal(0, noise_std, size=n)
        boot_scores[b] = np.clip(scores + noise, 0, None)

    alpha = (1 - ci) / 2
    return {
        "mean":    boot_scores.mean(axis=0),
        "std":     boot_scores.std(axis=0),
        "ci_low":  np.quantile(boot_scores, alpha,     axis=0),
        "ci_high": np.quantile(boot_scores, 1 - alpha, axis=0),
    }


def calculate_consensus_score(df: pd.DataFrame) -> np.ndarray:
    """
    v6 weights:
      30 % SHAP | 30 % Eigenvector | 15 % −log10(BH-p) | 10 % Betweenness
       5 % Latent entropy | 10 % Expression specificity
    """
    scaler = RobustScaler()
    p_col  = "P_Value_BH" if "P_Value_BH" in df.columns else "P_Value"

    def _scale(arr: np.ndarray) -> np.ndarray:
        arr    = np.clip(arr, None, np.nanpercentile(arr, 99))
        scaled = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
        scaled = scaled - scaled.min()
        return scaled / (scaled.max() + 1e-9)

    shap_s  = _scale(df["SHAP_Importance"].values)
    topo    = _scale(df["Eigenvector_Centrality"].values)
    p_score = _scale(-np.log10(df[p_col].clip(lower=1e-20).values))
    btwn    = _scale(df.get("Betweenness_Centrality",
                             pd.Series(np.zeros(len(df)))).values)

    score = shap_s * 0.30 + topo * 0.30 + p_score * 0.15 + btwn * 0.10

    if "Latent_Entropy" in df.columns:
        score += _scale(df["Latent_Entropy"].values) * 0.05
    else:
        score += shap_s * 0.05

    if "Expression_Entropy" in df.columns:
        score += _scale(1.0 - df["Expression_Entropy"].values) * 0.10
    else:
        score += topo * 0.10

    return score


def compute_differential_expression(
    df_log: pd.DataFrame,
    cluster_labels: np.ndarray,
    protein_list: list,
    gene_names: list,
) -> pd.DataFrame:
    """
    Per-protein fold change and effect size between clusters.
    Works with any number of clusters — compares each cluster vs all others.

    Returns DataFrame with columns:
        Gene_Symbol, Protein_ID, Cluster_ID,
        Mean_InCluster, Mean_OutCluster,
        Log2FC, Abs_Log2FC, Cohen_d, Direction
    """
    logger.info("Differential expression (fold change per cluster) …")
    records = []
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    for c_id in unique_clusters:
        in_mask  = cluster_labels == c_id
        out_mask = ~in_mask
        in_idx   = np.where(in_mask)[0]
        out_idx  = np.where(out_mask)[0]
        if len(in_idx) == 0 or len(out_idx) == 0:
            continue

        # Mean expression per protein, averaged over samples
        # df_log: proteins × samples
        df_in  = df_log.iloc[in_idx]    # proteins in this cluster
        df_out = df_log.iloc[out_idx]   # proteins outside

        mean_in_per_prot  = df_in.mean(axis=1)   # mean over samples
        mean_out_per_prot = df_out.mean(axis=1)

        for i, pid in enumerate(protein_list):
            gene = gene_names[i]
            if pid not in mean_in_per_prot.index or pid not in mean_out_per_prot.index:
                continue
            if in_mask[i]:
                mi  = float(mean_in_per_prot.loc[pid])
                mo  = float(mean_out_per_prot.loc[pid]) if pid in mean_out_per_prot.index else 0.0
            else:
                continue  # only compute for proteins that ARE in the cluster

            log2fc  = mi - mo
            std_in  = float(df_in.loc[pid].std())   if pid in df_in.index  else 1e-9
            std_out = float(df_out.loc[pid].std())  if pid in df_out.index else 1e-9
            pooled  = np.sqrt((std_in**2 + std_out**2) / 2 + 1e-9)
            cohen_d = log2fc / pooled

            records.append({
                "Cluster_ID":      int(c_id),
                "Protein_ID":      pid,
                "Gene_Symbol":     gene,
                "Mean_InCluster":  round(mi,           4),
                "Mean_OutCluster": round(mo,           4),
                "Log2FC":          round(log2fc,       4),
                "Abs_Log2FC":      round(abs(log2fc),  4),
                "Cohen_d":         round(cohen_d,      4),
                "Direction":       "UP" if log2fc > 0 else "DOWN",
            })

    if not records:
        logger.warning("Differential expression: no records computed.")
        return pd.DataFrame()

    df_de = pd.DataFrame(records).sort_values("Abs_Log2FC", ascending=False)
    logger.info("Differential expression: %d records across %d clusters.",
                len(df_de), len(unique_clusters))
    return df_de


def compute_volcano_data(df_final: pd.DataFrame, df_de: pd.DataFrame) -> pd.DataFrame:
    """
    Merge master results with differential expression for volcano plot.
    x-axis = Log2FC,  y-axis = -log10(P_Value_BH)
    """
    if df_de.empty or df_final.empty:
        return pd.DataFrame()

    p_col = "P_Value_BH" if "P_Value_BH" in df_final.columns else "P_Value"
    base  = df_final[["Gene_Symbol", "Protein_ID", p_col,
                       "Master_Score", "Anomaly_Class", "Cluster_ID"]].copy()

    # Use cluster 0 differential expression as default
    de_c0 = df_de[df_de["Cluster_ID"] == df_de["Cluster_ID"].min()][
        ["Protein_ID", "Log2FC", "Abs_Log2FC", "Cohen_d", "Direction"]
    ]
    merged = base.merge(de_c0, on="Protein_ID", how="left")
    merged["Neg_Log10_P"] = -np.log10(merged[p_col].clip(lower=1e-20))
    merged["Significant"] = (merged[p_col] < 0.05) & (merged["Abs_Log2FC"] > 0.5)
    return merged.fillna(0)


def compute_heatmap_data(
    df_log: pd.DataFrame,
    df_final: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Returns log2-normalised expression matrix for top_n proteins by Master Score.
    Rows = proteins (gene symbols), Columns = samples.
    """
    if df_final.empty or df_log.empty:
        return pd.DataFrame()

    top = df_final.sort_values("Master_Score", ascending=False).head(top_n)
    pids = [p for p in top["Protein_ID"] if p in df_log.index]
    if not pids:
        return pd.DataFrame()

    hm = df_log.loc[pids].copy()
    gene_map = dict(zip(top["Protein_ID"], top["Gene_Symbol"], strict=False))
    hm.index = [gene_map.get(p, p) for p in hm.index]

    # Z-score normalise per protein for better visual contrast
    hm = hm.sub(hm.mean(axis=1), axis=0).div(hm.std(axis=1) + 1e-9, axis=0)
    return hm
