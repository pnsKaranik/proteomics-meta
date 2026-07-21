"""Meta-analysis engine: VAE ensemble, feature importance, reporting and orchestration.

Pure statistical, network and I/O helpers live in the sibling modules
(:mod:`stats`, :mod:`networks`, :mod:`entropy`, :mod:`external`); this module holds
the TensorFlow-dependent VAE core and the pipeline orchestration that ties
everything together.
"""
from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
        from tensorflow.keras.callbacks import Callback, EarlyStopping
        from tensorflow.keras.layers import BatchNormalization, Dense, Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        tf.get_logger().setLevel("ERROR")
    except ImportError as exc:
        raise ImportError(
            "TensorFlow 2.x is required for the VAE engine.\n"
            "Fix: pip uninstall keras -y && pip install tensorflow==2.13.0\n"
            f"Original error: {exc}"
        ) from exc

from .capabilities import (
    ADVANCED_LIBS,
    LOUVAIN_AVAILABLE,
    SHAP_AVAILABLE,
    UMAP_AVAILABLE,
    get_logger,
    go,
    gp,
    km,
    phate,
    px,
    shap,
    umap,
)
from .config import MODE_DESCRIPTIONS, PipelineConfig, get_pipeline_mode
from .entropy import compute_expression_entropy, compute_latent_entropy
from .external import fetch_string_interactions, get_gene_names_optimized
from .networks import (
    build_network,
    build_ppi_network,
    calculate_network_centrality,
    cluster_proteins_adaptive,
)
from .stats import (
    calculate_bootstrap_pvalues,
    calculate_consensus_score,
    calculate_robust_stats,
    compute_bootstrap_ci,
    compute_differential_expression,
    compute_heatmap_data,
    compute_volcano_data,
)

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")

# Public surface consumed by the Streamlit app and CLI. Listing re-exported
# helpers here keeps the app's single import block working and documents the API.
__all__ = [
    "run_pipeline",
    "run_pipeline_initial",
    "run_dynamic_critic",
    "generate_html_report",
    "save_detailed_excel",
    "calculate_consensus_score",
    "compute_bootstrap_ci",
    "compute_differential_expression",
    "compute_volcano_data",
    "compute_heatmap_data",
    "fetch_string_interactions",
    "build_ppi_network",
    "get_pipeline_mode",
    "MODE_DESCRIPTIONS",
    "SHAP_AVAILABLE",
    "LOUVAIN_AVAILABLE",
    "ADVANCED_LIBS",
]


class KLWarmupCallback(Callback):
    def __init__(self, beta_var, beta_target: float, warmup_fraction: float = 0.20):
        super().__init__()
        self.beta_var        = beta_var
        self.beta_target     = beta_target
        self.warmup_fraction = warmup_fraction
        self.total_epochs    = 1

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", 1)

    def on_epoch_begin(self, epoch, logs=None):
        warmup_epochs = max(1, int(self.total_epochs * self.warmup_fraction))
        frac = min(1.0, (epoch + 1) / warmup_epochs)
        self.beta_var.assign(frac * self.beta_target)


class ProteomicsVAE(Model):
    """
    β-VAE subclass with KL always in computation graph.
    get_full_encoder() returns (z_mean, z_log_var) needed for latent entropy.
    """
    def __init__(self, input_dim: int, latent_dim: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_beta")

        self.enc_d1  = Dense(128, activation="elu",    name="enc_1")
        self.enc_bn1 = BatchNormalization(name="enc_bn1")
        self.enc_d2  = Dense(64,  activation="elu",    name="enc_2")
        self.enc_bn2 = BatchNormalization(name="enc_bn2")
        self.z_mean_layer    = Dense(latent_dim, name="z_mean")
        self.z_log_var_layer = Dense(latent_dim, name="z_log_var")

        self.dec_d1  = Dense(64,        activation="elu",    name="dec_1")
        self.dec_bn1 = BatchNormalization(name="dec_bn1")
        self.dec_d2  = Dense(128,       activation="elu",    name="dec_2")
        self.dec_bn2 = BatchNormalization(name="dec_bn2")
        self.dec_out = Dense(input_dim, activation="linear", name="dec_out")

    def encode(self, x, training=False):
        h = self.enc_bn1(self.enc_d1(x),  training=training)
        h = self.enc_bn2(self.enc_d2(h),  training=training)
        return self.z_mean_layer(h), self.z_log_var_layer(h)

    def decode(self, z, training=False):
        h = self.dec_bn1(self.dec_d1(z), training=training)
        h = self.dec_bn2(self.dec_d2(h), training=training)
        return self.dec_out(h)

    def reparameterise(self, z_mean, z_log_var):
        return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))

    def call(self, x, training=False):
        z_mean, z_log_var = self.encode(x, training=training)
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        ) / tf.cast(self.input_dim, tf.float32)
        self.add_loss(self.beta * kl)
        return self.decode(self.reparameterise(z_mean, z_log_var), training=training)

    def get_encoder(self) -> Model:
        x = Input(shape=(self.input_dim,), name="encoder_input")
        z_mean, _ = self.encode(x, training=False)
        return Model(x, z_mean, name="encoder")

    def get_full_encoder(self) -> Model:
        """Returns both z_mean and z_log_var — needed for latent entropy."""
        x = Input(shape=(self.input_dim,), name="full_enc_input")
        z_mean, z_log_var = self.encode(x, training=False)
        return Model(x, [z_mean, z_log_var], name="full_encoder")


def compute_jackknife_stability(
    X_norm: np.ndarray,
    protein_list: list,
    config: PipelineConfig,
    top_k_fraction: float = 0.20,
) -> np.ndarray:
    """
    Leave-one-out jackknife over samples.

    For each leave-one-out fold:
      1. Train a single VAE on the reduced dataset
      2. Compute reconstruction errors
      3. Flag the top-k proteins (top_k_fraction of total) as "significant"

    stability_score[i] = fraction of folds where protein i was in top-k
    Returns array of shape (n_proteins,) with values in [0, 1].

    With n=3 samples this yields 3 folds ([S1,S2], [S0,S2], [S0,S1]).
    A protein scoring 1.0 was top-k in ALL folds → very robust finding.
    """
    n_samples  = X_norm.shape[1]
    n_proteins = X_norm.shape[0]
    top_k      = max(1, int(n_proteins * top_k_fraction))

    if n_samples < 2:
        logger.warning("Jackknife requires ≥ 2 samples; returning zeros.")
        return np.zeros(n_proteins)

    logger.info("Jackknife stability: %d folds …", n_samples)
    hit_counts = np.zeros(n_proteins, dtype=float)

    for leave_out in range(n_samples):
        idx = [i for i in range(n_samples) if i != leave_out]
        X_fold = X_norm[:, idx]           # (n_proteins, n_samples-1)

        try:
            K.clear_session()
            input_dim  = X_fold.shape[1]
            latent_dim = min(config.latent_dim, max(2, input_dim - 1))
            vae = ProteomicsVAE(input_dim, latent_dim)
            vae.compile(optimizer=Adam(config.learning_rate), loss="mse")
            batch_size = max(4, min(32, n_proteins // 8))
            vae.fit(
                X_fold, X_fold, verbose=0,
                epochs=max(20, config.epochs // 3),
                batch_size=batch_size,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                           KLWarmupCallback(vae.beta, config.beta_vae, 0.20)],
            )
            recon   = vae.predict(X_fold, verbose=0)
            errors  = np.mean(np.square(X_fold - recon), axis=1)
            top_idx = np.argsort(errors)[-top_k:]
            hit_counts[top_idx] += 1.0
        except Exception as exc:
            logger.warning("Jackknife fold %d failed: %s", leave_out, exc)

    K.clear_session()
    stability = hit_counts / n_samples
    logger.info("Jackknife complete. Mean stability: %.3f", stability.mean())
    return stability


def compute_embedding(
    latent: np.ndarray,
    mode: str,
    random_seed: int = 42,
):
    """
    Returns (embedding_2d, embedding_3d, method_used).

    ultra_sparse → PCA
    low_sample   → UMAP (fallback PCA)
    moderate     → UMAP (fallback PCA)
    full         → PHATE (fallback UMAP → PCA)
    """
    n = latent.shape[0]
    lc = np.nan_to_num(latent)

    def _pca_2d():
        n_comp = min(2, n - 1, lc.shape[1])
        coords = PCA(n_components=n_comp, random_state=random_seed).fit_transform(lc)
        if coords.shape[1] < 2:
            coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
        return coords

    def _pca_3d():
        n_comp = min(3, n - 1, lc.shape[1])
        coords = PCA(n_components=n_comp, random_state=random_seed).fit_transform(lc)
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((n, 3 - coords.shape[1]))])
        return coords

    def _umap_embed(n_comp):
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn not installed")
        n_neighbors = min(15, max(2, n - 1))
        return umap.UMAP(
            n_components=n_comp,
            n_neighbors=n_neighbors,
            random_state=random_seed,
        ).fit_transform(lc)

    if mode == "ultra_sparse":
        logger.info("Embedding: PCA (ultra-sparse mode)")
        return _pca_2d(), _pca_3d(), "PCA"

    elif mode in ("low_sample", "moderate"):
        try:
            emb2 = _umap_embed(2)
            emb3 = _umap_embed(3)
            logger.info("Embedding: UMAP (%s mode)", mode)
            return emb2, emb3, "UMAP"
        except Exception as exc:
            logger.warning("UMAP failed (%s); using PCA.", exc)
            return _pca_2d(), _pca_3d(), "PCA"

    else:  # full
        if ADVANCED_LIBS:
            try:
                emb2 = phate.PHATE(n_components=2, verbose=0,
                                   random_state=random_seed).fit_transform(lc)
                emb3 = phate.PHATE(n_components=3, verbose=0,
                                   random_state=random_seed).fit_transform(lc)
                logger.info("Embedding: PHATE (full mode)")
                return emb2, emb3, "PHATE"
            except Exception as exc:
                logger.warning("PHATE failed (%s); trying UMAP.", exc)
        try:
            emb2 = _umap_embed(2)
            emb3 = _umap_embed(3)
            logger.info("Embedding: UMAP (PHATE fallback)")
            return emb2, emb3, "UMAP"
        except Exception as exc:
            logger.warning("UMAP failed (%s); using PCA.", exc)
            return _pca_2d(), _pca_3d(), "PCA"


def analyze_trajectory_drivers_safe(
    df_log: pd.DataFrame,
    pseudotime: np.ndarray,
    protein_list: list,
    mode: str = "full",
) -> pd.DataFrame:
    logger.info("Trajectory analysis …")

    if mode == "ultra_sparse":
        logger.info("Trajectory disabled in ultra-sparse mode.")
        return pd.DataFrame()

    if df_log.shape[1] != len(pseudotime):
        logger.warning("Skipping trajectory: shape mismatch (%d vs %d).",
                       df_log.shape[1], len(pseudotime))
        return pd.DataFrame()

    if len(pseudotime) < 3:
        logger.warning("Trajectory skipped: fewer than 3 samples.")
        return pd.DataFrame()

    subset       = df_log.loc[df_log.index.intersection(protein_list)]
    corrs, pvals = [], []
    for _, row in subset.iterrows():
        rho, pval = spearmanr(row.values, pseudotime)
        corrs.append(float(rho)  if not np.isnan(rho)  else 0.0)
        pvals.append(float(pval) if not np.isnan(pval) else 1.0)

    try:
        bh_pvals = false_discovery_control(np.array(pvals), method="bh")
    except Exception:
        bh_pvals = np.array(pvals)

    return pd.DataFrame({
        "Protein_ID":             subset.index,
        "Trajectory_Correlation": corrs,
        "Trajectory_PVal_Raw":    pvals,
        "Trajectory_PVal_BH":     bh_pvals,
    }).set_index("Protein_ID")


def compute_shap_importance(
    encoders: list,
    X_norm: np.ndarray,
    n_background: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute per-protein importance scores.

    Strategy (in order of preference):
      1. SHAP DeepExplainer — best, but fails with very few samples
      2. Gradient-based sensitivity (|∂output/∂input|) — works always
      3. L2-norm of latent representation — fast fallback
      4. Reconstruction error rank — last resort

    With n_samples < 10, DeepExplainer often returns all-zero values
    due to insufficient background diversity. We detect this and fall through.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_proteins = X_norm.shape[0]
    bg_size    = min(n_background, X_norm.shape[0])
    bg_idx     = rng.choice(n_proteins, size=bg_size, replace=False)
    background = X_norm[bg_idx]
    all_imp    = []

    # ── Strategy 1: SHAP DeepExplainer ───────────────────────────────────────
    if SHAP_AVAILABLE:
        for enc in encoders:
            try:
                sv  = shap.DeepExplainer(enc, background).shap_values(X_norm)
                if isinstance(sv, list):
                    imp = np.mean([np.abs(s).mean(axis=1) for s in sv], axis=0)
                else:
                    imp = np.abs(sv).mean(axis=1)
                imp = np.asarray(imp).flatten()
                # Detect all-zero output (common with n<10 samples)
                if imp.max() > 1e-9:
                    all_imp.append(imp)
                else:
                    logger.warning("SHAP returned all-zero for one encoder — skipping.")
            except Exception as exc:
                logger.warning("SHAP DeepExplainer failed (%s).", exc)

    if all_imp:
        importance = np.mean(all_imp, axis=0).flatten()
        logger.info("Ensemble SHAP from %d/%d encoders.", len(all_imp), len(encoders))
        vmin, vmax = importance.min(), importance.max()
        return (importance - vmin) / (vmax - vmin + 1e-9)

    # ── Strategy 2: Gradient-based sensitivity ────────────────────────────────
    # |∂(latent_mean)/∂input| summed over latent dims — works with any n_samples
    logger.info("SHAP fallback: gradient-based sensitivity …")
    try:
        import tensorflow as tf
        grad_imps = []
        for enc in encoders:
            try:
                X_tf = tf.constant(X_norm, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(X_tf)
                    z = enc(X_tf, training=False)
                    # Use L2 norm of latent as scalar output
                    out = tf.reduce_sum(tf.square(z), axis=1)
                grads = tape.gradient(out, X_tf)   # (n_proteins, n_samples)
                if grads is not None:
                    imp = tf.reduce_sum(tf.abs(grads), axis=1).numpy().flatten()
                    if imp.max() > 1e-9:
                        grad_imps.append(imp)
            except Exception as exc:
                logger.warning("Gradient sensitivity failed for encoder: %s", exc)

        if grad_imps:
            importance = np.mean(grad_imps, axis=0).flatten()
            logger.info("Gradient sensitivity: success (%d encoders).", len(grad_imps))
            vmin, vmax = importance.min(), importance.max()
            return (importance - vmin) / (vmax - vmin + 1e-9)
    except Exception as exc:
        logger.warning("Gradient strategy failed: %s", exc)

    # ── Strategy 3: L2-norm of latent representations ─────────────────────────
    logger.info("SHAP fallback: L2-norm of ensemble latent …")
    try:
        latent_preds = [e.predict(X_norm, verbose=0) for e in encoders]
        mean_latent  = np.mean(latent_preds, axis=0)   # (n_proteins, latent_dim)
        importance   = np.linalg.norm(mean_latent, axis=1).flatten()
        if importance.max() > 1e-9:
            logger.info("L2-norm fallback: success.")
            vmin, vmax = importance.min(), importance.max()
            return (importance - vmin) / (vmax - vmin + 1e-9)
    except Exception as exc:
        logger.warning("L2-norm fallback failed: %s", exc)

    # ── Strategy 4: Variance of latent activations ────────────────────────────
    logger.info("SHAP fallback: latent variance …")
    try:
        all_latents  = np.stack([e.predict(X_norm, verbose=0) for e in encoders])
        # Variance across ensemble members — high variance = unstable = important
        importance   = np.var(all_latents, axis=0).sum(axis=1).flatten()
        vmin, vmax   = importance.min(), importance.max()
        logger.info("Latent variance fallback: success.")
        return (importance - vmin) / (vmax - vmin + 1e-9)
    except Exception as exc:
        logger.warning("Latent variance fallback failed: %s", exc)

    # ── Strategy 5: Uniform (last resort) ────────────────────────────────────
    logger.warning("All SHAP strategies failed — returning uniform importance.")
    return np.ones(n_proteins) / n_proteins


def compute_pips(
    graph: nx.Graph,
    protein_list: list,
    recon_errors: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Closed-form graph heat diffusion PIPS score."""
    n = len(protein_list)
    if graph.number_of_edges() == 0:
        r = recon_errors / (recon_errors.max() + 1e-9)
        return r

    node_list  = list(graph.nodes())
    m          = len(node_list)
    A          = nx.to_numpy_array(graph, nodelist=node_list, weight="weight")
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_is       = np.diag(d_inv_sqrt)
    L_norm     = np.eye(m) - D_is @ A @ D_is

    try:
        M_inv = np.linalg.inv(np.eye(m) + alpha * L_norm)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(np.eye(m) + alpha * L_norm)

    r_full    = recon_errors / (recon_errors.max() + 1e-9)
    prot_idx  = {p: i for i, p in enumerate(protein_list)}
    r_graph   = np.array([r_full[prot_idx[p]] if p in prot_idx else 0.0 for p in node_list])

    pips_graph = M_inv @ r_graph
    pips = np.zeros(n)
    for j, p in enumerate(node_list):
        if p in prot_idx:
            pips[prot_idx[p]] = pips_graph[j]

    pmin, pmax = pips.min(), pips.max()
    return (pips - pmin) / (pmax - pmin + 1e-9)


def run_dynamic_critic(
    df_features: pd.DataFrame,
    contamination: float = 0.1,
    network_sensitivity: float = 0.5,
):
    logger.info("Critic (contamination=%.2f) …", contamination)
    core_cols  = ["Reconstruction_Error", "Latent_Connectivity", "Eigenvector_Centrality"]
    extra_cols = ["SHAP_Importance", "Reconstruction_Error_CV", "Z_Score",
                  "Latent_Entropy", "Expression_Entropy"]
    available  = core_cols + [c for c in extra_cols if c in df_features.columns]

    X = RobustScaler().fit_transform(df_features[available].fillna(0).values)

    scores = IsolationForest(contamination=0.1,
                             random_state=42, n_estimators=200).fit(X).decision_function(X)
    labels = IsolationForest(contamination=contamination,
                             random_state=42, n_estimators=200).fit_predict(X)

    avg_deg = df_features["Latent_Connectivity"].mean()
    if np.isnan(avg_deg):
        avg_deg = 0.0
    deg_thr = max(1.0, avg_deg) * (1.0 + network_sensitivity)
    degrees = df_features["Latent_Connectivity"].values

    classes = np.select(
        [labels == 1, (labels != 1) & (degrees >= deg_thr)],
        ["Validated_Signal", "Biological_Discovery"],
        default="Technical_Noise",
    )
    return classes.tolist(), scores


def plot_3d_protein_atlas(embedding_3d, protein_list, cluster_labels,
                           gene_names, anomaly_classes=None,
                           method_label: str = "PHATE"):
    if embedding_3d is None:
        return go.Figure()
    try:
        col_names = [f"{method_label}_1", f"{method_label}_2", f"{method_label}_3"]
        df_3d = pd.DataFrame(embedding_3d, columns=col_names)
        df_3d["Protein"] = protein_list
        df_3d["Gene"]    = gene_names if len(gene_names) == len(protein_list) else protein_list
        df_3d["Cluster"] = cluster_labels.astype(str)
        col, cmap = "Cluster", px.colors.qualitative.Dark24
        if anomaly_classes is not None:
            df_3d["Anomaly_Class"] = anomaly_classes
            col  = "Anomaly_Class"
            cmap = {"Validated_Signal": "#00E5FF", "Biological_Discovery": "#FFD700",
                    "Technical_Noise": "rgba(200,200,200,0.2)"}
        return px.scatter_3d(
            df_3d, x=col_names[0], y=col_names[1], z=col_names[2],
            color=col, hover_name="Gene", hover_data=["Protein"],
            title=f"3D Atlas — {method_label}", opacity=0.7, size_max=5,
            color_discrete_map=cmap if isinstance(cmap, dict) else None,
            color_discrete_sequence=cmap if isinstance(cmap, list) else None,
        )
    except Exception as exc:
        logger.warning("3D plot error: %s", exc)
        return go.Figure()


def plot_2d_embedding_drivers(embedding_2d, protein_list, cluster_labels,
                               gene_names, shap_scores,
                               method_label: str = "PHATE"):
    if embedding_2d is None:
        return go.Figure()
    try:
        col_names = [f"{method_label}_1", f"{method_label}_2"]
        emb = np.array(embedding_2d)
        # Guard: embedding must be 2D array with exactly 2 columns
        if emb.ndim != 2 or emb.shape[1] < 2:
            logger.warning("2D embedding wrong shape %s; skipping plot.", emb.shape)
            return go.Figure()
        df_2d = pd.DataFrame(emb[:, :2], columns=col_names)
        df_2d["Protein"] = protein_list
        df_2d["Gene"]    = gene_names if len(gene_names) == len(protein_list) else protein_list
        df_2d["Cluster"] = np.array(cluster_labels).astype(str)
        shap_arr = np.array(shap_scores).flatten()
        if len(shap_arr) != len(protein_list):
            shap_arr = np.full(len(protein_list), 0.1)
        df_2d["SHAP_Impact"] = shap_arr.tolist()
        df_2d["Plot_Size"]   = (shap_arr + 0.05).tolist()
        return px.scatter(
            df_2d, x=col_names[0], y=col_names[1], color="Cluster",
            size="Plot_Size", hover_name="Gene",
            hover_data=["Protein", "SHAP_Impact"],
            title=f"2D {method_label} — bubble = SHAP",
            color_discrete_sequence=px.colors.qualitative.Dark24, size_max=25,
        )
    except Exception as exc:
        logger.warning("2D embedding error: %s", exc)
        return go.Figure()


def visualize_network_static(G: nx.Graph, output_path: str, gene_map: dict = None):
    try:
        plt.figure(figsize=(14, 12))
        pos   = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        sizes = [G.degree(n) * 100 + 50 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#d62728", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color="grey")
        labels = {n: gene_map.get(n, n) for n in G.nodes()} if gene_map else None
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")
        plt.title(f"Driver Network | {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(output_path, dpi=300); plt.close()
    except Exception as exc:
        logger.warning("Network viz error: %s", exc)


def save_detailed_excel(state, df_final, drivers_df, output_path, config=None):
    logger.info("Excel export: %s", output_path)
    try:
        edge_data = []
        G = state.get("G_initial")
        if G:
            for u, v, d in G.edges(data=True):
                if d.get("weight", 0) > 0.05:
                    edge_data.append({
                        "Source": u, "Target": v,
                        "Source_Gene": state["gene_map"].get(u, u),
                        "Target_Gene": state["gene_map"].get(v, v),
                        "Weight": d["weight"],
                    })

        mode       = state.get("pipeline_mode", "unknown")
        mode_desc  = MODE_DESCRIPTIONS.get(mode, {})
        emb_method = state.get("embedding_method", "PCA")

        with pd.ExcelWriter(output_path, engine="openpyxl") as w:
            pd.DataFrame({
                "Run timestamp":    [time.strftime("%Y-%m-%d %H:%M:%S")],
                "Pipeline mode":    [mode_desc.get("label", mode)],
                "Network method":   [mode_desc.get("network", "?")],
                "Clustering":       [mode_desc.get("clustering", "?")],
                "Visualisation":    [emb_method],
                "p-value method":   [mode_desc.get("pvalues", "BH FDR")],
                "Proteins":         [len(df_final)],
                "Latent dim":       [config.latent_dim  if config else "?"],
                "Iterations":       [config.iterations  if config else "?"],
                "Epochs":           [config.epochs      if config else "?"],
                "Beta VAE":         [config.beta_vae    if config else "?"],
                "SHAP":             ["Ensemble DeepExplainer" if SHAP_AVAILABLE else "L2-norm"],
                "FDR":              ["BH one-tailed"],
                "Edge threshold":   [config.pcorr_threshold if config else 0.10],
                "PIPS alpha":       [config.pips_alpha  if config else 0.5],
                "Jackknife":        [config.jackknife   if config else False],
                "Bootstrap CI":     [config.n_bootstrap if config else 0],
            }).T.to_excel(w, sheet_name="Summary", header=False)

            df_final.to_excel(w, sheet_name="Master_Report", index=False)

            if not drivers_df.empty:
                drivers_df.to_excel(w, sheet_name="Top_Drivers", index=False)

            if "df_log" in state:
                state["df_log"].to_excel(w, sheet_name="Normalized_Expression")

            if "latent_space" in state:
                ls = state["latent_space"]
                pd.DataFrame(ls, index=state["protein_list"],
                             columns=[f"Latent_{i+1}" for i in range(ls.shape[1])]
                             ).to_excel(w, sheet_name="Latent_Space")

            if edge_data:
                (pd.DataFrame(edge_data)
                 .sort_values("Weight", ascending=False)
                 .head(50_000)
                 .to_excel(w, sheet_name="Network_Edges", index=False))

            enr = state.get("enrichment_df")
            if enr is not None and not enr.empty:
                enr.to_excel(w, sheet_name="Enrichment_Analysis", index=False)

            if "stability_scores" in state:
                pd.DataFrame(state["stability_scores"]).to_excel(
                    w, sheet_name="VAE_Stability", index=False)

            if "PIPS_Score" in df_final.columns:
                pips_cols = [c for c in ["Gene_Symbol", "PIPS_Score", "Master_Score",
                                          "Anomaly_Class", "P_Value_BH"] if c in df_final.columns]
                df_final[pips_cols].sort_values("PIPS_Score", ascending=False).to_excel(
                    w, sheet_name="PIPS_Ranking", index=False)

            sh_cols = [c for c in ["Gene_Symbol", "Latent_Entropy", "Expression_Entropy",
                                    "PIPS_Score", "Master_Score", "Anomaly_Class"] if c in df_final.columns]
            if any(c in df_final.columns for c in ["Latent_Entropy", "Expression_Entropy"]):
                sort_col = "Latent_Entropy" if "Latent_Entropy" in df_final.columns else "Gene_Symbol"
                df_final[sh_cols].sort_values(sort_col, ascending=False).to_excel(
                    w, sheet_name="Shannon_Info", index=False)

            # Jackknife sheet
            if "Jackknife_Stability" in df_final.columns:
                jk_cols = [c for c in ["Gene_Symbol", "Jackknife_Stability",
                                        "Master_Score", "Anomaly_Class"] if c in df_final.columns]
                df_final[jk_cols].sort_values("Jackknife_Stability", ascending=False).to_excel(
                    w, sheet_name="Jackknife_Stability", index=False)

            # Bootstrap CI sheet
            for ci_col in ["Master_Score_CI_Low", "Master_Score_CI_High", "Master_Score_Std"]:
                if ci_col not in df_final.columns:
                    break
            else:
                ci_cols = [c for c in ["Gene_Symbol", "Master_Score",
                                        "Master_Score_CI_Low", "Master_Score_CI_High",
                                        "Master_Score_Std", "Anomaly_Class"] if c in df_final.columns]
                df_final[ci_cols].sort_values("Master_Score", ascending=False).to_excel(
                    w, sheet_name="Bootstrap_CI", index=False)

    except Exception as exc:
        logger.error("Excel export failed: %s", exc)


def generate_html_report(
    state: dict,
    df_final: pd.DataFrame,
    df_de: pd.DataFrame,
    df_ppi: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Generate a standalone HTML report with all analysis results.
    Embeds Plotly charts as JSON, tables as HTML.
    Returns path to the generated file.
    """
    logger.info("Generating HTML report …")
    try:
        import plotly.io as pio

        pipeline_mode = state.get("pipeline_mode", "unknown")
        n_samples     = state.get("n_samples", 0)
        enr_df        = state.get("enrichment_df", pd.DataFrame())
        config        = state.get("config")
        p_col         = "P_Value_BH" if "P_Value_BH" in df_final.columns else "P_Value"

        # ── Charts ────────────────────────────────────────────────────────────
        charts_html = ""

        # 1. Master score distribution
        try:
            fig1 = px.histogram(df_final, x="Master_Score", color="Anomaly_Class",
                                nbins=50, barmode="overlay", title="Master Score Distribution",
                                color_discrete_map={"Validated_Signal":"#007ACC",
                                                     "Biological_Discovery":"#DCDCAA",
                                                     "Technical_Noise":"#F44747"})
            fig1.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                               plot_bgcolor="#1a1a2e", height=350)
            charts_html += f'<div class="chart">{pio.to_html(fig1, full_html=False, include_plotlyjs=False)}</div>'
        except Exception: pass

        # 2. Top 20 proteins bar chart
        try:
            top20 = df_final.sort_values("Master_Score", ascending=False).head(20)
            fig2  = px.bar(top20, x="Master_Score", y="Gene_Symbol", orientation="h",
                           color="Anomaly_Class", title="Top 20 Proteins by Master Score",
                           color_discrete_map={"Validated_Signal":"#007ACC",
                                               "Biological_Discovery":"#DCDCAA",
                                               "Technical_Noise":"#F44747"})
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                               plot_bgcolor="#1a1a2e", height=500,
                               yaxis=dict(autorange="reversed"))
            charts_html += f'<div class="chart">{pio.to_html(fig2, full_html=False, include_plotlyjs=False)}</div>'
        except Exception: pass

        # 3. Enrichment bar chart per cluster
        if not enr_df.empty:
            try:
                enr_top = enr_df.sort_values("Combined Score", ascending=False).head(20).copy()
                enr_top["Cluster_Label"] = "Cluster " + enr_top["Cluster"].astype(str)
                enr_top["-log10p"] = -np.log10(enr_top["Adjusted P-value"].clip(lower=1e-20))
                fig3 = px.bar(enr_top, x="-log10p", y="Term", color="Cluster_Label",
                              orientation="h", title="Top Enriched Pathways",
                              hover_data=["Combined Score", "Adjusted P-value"])
                fig3.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                                   plot_bgcolor="#1a1a2e", height=600,
                                   yaxis=dict(autorange="reversed"))
                charts_html += f'<div class="chart">{pio.to_html(fig3, full_html=False, include_plotlyjs=False)}</div>'
            except Exception: pass

        # 4. Volcano plot
        if not df_de.empty:
            try:
                vol = compute_volcano_data(df_final, df_de)
                if not vol.empty:
                    vol["Label"] = vol.apply(
                        lambda r: r["Gene_Symbol"] if r["Significant"] else "", axis=1)
                    fig4 = px.scatter(vol, x="Log2FC", y="Neg_Log10_P",
                                      color="Anomaly_Class", hover_name="Gene_Symbol",
                                      title="Volcano Plot — Log2FC vs Significance",
                                      color_discrete_map={"Validated_Signal":"#007ACC",
                                                          "Biological_Discovery":"#DCDCAA",
                                                          "Technical_Noise":"#F44747"})
                    fig4.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#888")
                    fig4.add_vline(x=0.5,  line_dash="dash", line_color="#888")
                    fig4.add_vline(x=-0.5, line_dash="dash", line_color="#888")
                    fig4.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                                       plot_bgcolor="#1a1a2e", height=450)
                    charts_html += f'<div class="chart">{pio.to_html(fig4, full_html=False, include_plotlyjs=False)}</div>'
            except Exception: pass

        # 5. STRING PPI network
        if not df_ppi.empty:
            try:
                top_genes = set(df_final.sort_values("Master_Score", ascending=False)
                                .head(30)["Gene_Symbol"])
                df_ppi_top = df_ppi[
                    df_ppi["gene_a"].isin(top_genes) | df_ppi["gene_b"].isin(top_genes)
                ].head(100)
                if not df_ppi_top.empty:
                    G_vis = nx.Graph()
                    score_m = dict(zip(df_final["Gene_Symbol"], df_final["Master_Score"], strict=False))
                    class_m = dict(zip(df_final["Gene_Symbol"], df_final["Anomaly_Class"], strict=False))
                    for _, r in df_ppi_top.iterrows():
                        G_vis.add_edge(r["gene_a"], r["gene_b"], weight=r["score"])
                    pos = nx.spring_layout(G_vis, seed=42, k=0.5)
                    edge_x, edge_y = [], []
                    for e in G_vis.edges():
                        x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
                        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
                    node_x = [pos[n][0] for n in G_vis.nodes()]
                    node_y = [pos[n][1] for n in G_vis.nodes()]
                    node_colors = ["#007ACC" if class_m.get(n)=="Validated_Signal"
                                   else "#DCDCAA" if class_m.get(n)=="Biological_Discovery"
                                   else "#F44747" for n in G_vis.nodes()]
                    node_sizes  = [max(8, score_m.get(n, 0)*30) for n in G_vis.nodes()]
                    fig5 = go.Figure()
                    fig5.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                              line=dict(color="#444", width=0.8),
                                              hoverinfo="none"))
                    fig5.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                              marker=dict(size=node_sizes, color=node_colors,
                                                          line=dict(width=1, color="#222")),
                                              text=list(G_vis.nodes()),
                                              textposition="top center",
                                              textfont=dict(size=9, color="#DDD"),
                                              hovertext=[f"{n}<br>Score: {score_m.get(n,0):.3f}"
                                                         for n in G_vis.nodes()],
                                              hoverinfo="text"))
                    fig5.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                                       plot_bgcolor="#1a1a2e", height=550,
                                       title="STRING PPI Network — Top 30 Proteins",
                                       showlegend=False,
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    charts_html += f'<div class="chart">{pio.to_html(fig5, full_html=False, include_plotlyjs=False)}</div>'
            except Exception: pass

        # 6. Cluster comparison bar chart
        if not df_de.empty:
            try:
                top_de = df_de.groupby("Cluster_ID").apply(
                    lambda x: x.nlargest(10, "Abs_Log2FC")).reset_index(drop=True)
                top_de["Cluster_Label"] = "Cluster " + top_de["Cluster_ID"].astype(str)
                fig6 = px.bar(top_de, x="Log2FC", y="Gene_Symbol", color="Cluster_Label",
                              facet_col="Cluster_Label", orientation="h",
                              title="Top Differentially Expressed Proteins per Cluster",
                              color_discrete_sequence=["#007ACC","#DCDCAA","#66bb6a","#ffa726"])
                fig6.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e",
                                   plot_bgcolor="#1a1a2e", height=500)
                charts_html += f'<div class="chart">{pio.to_html(fig6, full_html=False, include_plotlyjs=False)}</div>'
            except Exception: pass

        # 7. Gene Ontology treemap
        if not enr_df.empty:
            try:
                enr_tree = enr_df.copy()
                enr_tree["Cluster_Label"] = "Cluster " + enr_tree["Cluster"].astype(str)
                enr_tree["Score"] = enr_tree["Combined Score"].clip(lower=0.1)
                fig7 = px.treemap(enr_tree, path=["Cluster_Label","Gene_set","Term"],
                                  values="Score",
                                  color="Adjusted P-value",
                                  color_continuous_scale="RdBu_r",
                                  title="Gene Ontology Treemap")
                fig7.update_layout(paper_bgcolor="#1a1a2e", height=550)
                charts_html += f'<div class="chart">{pio.to_html(fig7, full_html=False, include_plotlyjs=False)}</div>'
            except Exception: pass

        # ── Tables ────────────────────────────────────────────────────────────
        def df_to_html_table(df, max_rows=50, title=""):
            if df.empty:
                return f"<p style='color:#888'>No data available for {title}.</p>"
            rows = df.head(max_rows).to_html(
                index=False, classes="data-table",
                border=0, justify="left",
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            )
            return f"<h3>{title}</h3>{rows}"

        top_drivers_html = df_to_html_table(
            df_final.sort_values("Master_Score", ascending=False).head(50)
            [[c for c in ["Gene_Symbol","Master_Score","PIPS_Score","Jackknife_Stability",
                          "Anomaly_Class","SHAP_Importance",p_col,"Cluster_ID"] if c in df_final.columns]],
            title="Top 50 Proteins by Master Score"
        )

        de_html = df_to_html_table(
            df_de.sort_values("Abs_Log2FC", ascending=False).head(50) if not df_de.empty else pd.DataFrame(),
            title="Top Differential Expression Results"
        )

        ppi_html = df_to_html_table(
            df_ppi.sort_values("score", ascending=False).head(50) if not df_ppi.empty else pd.DataFrame(),
            title="Top STRING Interactions"
        )

        enr_html = df_to_html_table(
            enr_df.sort_values("Combined Score", ascending=False).head(50) if not enr_df.empty else pd.DataFrame(),
            title="Pathway Enrichment Results"
        )

        # ── Build HTML ────────────────────────────────────────────────────────
        cfg_html = ""
        if config:
            cfg_html = f"""
            <div class="meta-grid">
                <div class="meta-item"><span class="meta-label">Pipeline Mode</span><span class="meta-value">{pipeline_mode}</span></div>
                <div class="meta-item"><span class="meta-label">Samples</span><span class="meta-value">{n_samples}</span></div>
                <div class="meta-item"><span class="meta-label">Proteins</span><span class="meta-value">{len(df_final):,}</span></div>
                <div class="meta-item"><span class="meta-label">Clusters</span><span class="meta-value">{len(np.unique(df_final['Cluster_ID']))}</span></div>
                <div class="meta-item"><span class="meta-label">Validated</span><span class="meta-value">{len(df_final[df_final['Anomaly_Class']=='Validated_Signal'])}</span></div>
                <div class="meta-item"><span class="meta-label">Discoveries</span><span class="meta-value">{len(df_final[df_final['Anomaly_Class']=='Biological_Discovery'])}</span></div>
                <div class="meta-item"><span class="meta-label">Mean Master Score</span><span class="meta-value">{df_final['Master_Score'].mean():.4f}</span></div>
                <div class="meta-item"><span class="meta-label">STRING PPIs</span><span class="meta-value">{len(df_ppi):,}</span></div>
            </div>"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meta Analysis Engine — Full Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d1a; color: #d4d4d4; font-family: 'Segoe UI', sans-serif; padding: 0; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             padding: 40px; border-bottom: 2px solid #007ACC; }}
  .header h1 {{ font-size: 2.2rem; color: #fff; margin-bottom: 8px; }}
  .header .subtitle {{ color: #858585; font-size: 1rem; }}
  .header .badge {{ display:inline-block; padding:3px 10px; border-radius:3px; font-size:.75rem;
                    margin:4px 4px 0 0; }}
  .badge-blue   {{ background:#1e3a5f; color:#64b5f6; }}
  .badge-orange {{ background:#3a2a1a; color:#ffa726; }}
  .badge-green  {{ background:#1a3a2a; color:#66bb6a; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 20px; }}
  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 1.3rem; color: #007ACC; border-bottom: 1px solid #333;
                    padding-bottom: 8px; margin-bottom: 20px; }}
  .chart {{ background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 6px;
            padding: 16px; margin-bottom: 24px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin-bottom: 24px; }}
  .meta-item {{ background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 6px;
                padding: 14px; text-align: center; }}
  .meta-label {{ display: block; font-size: .72rem; color: #888; text-transform: uppercase;
                 letter-spacing: .5px; margin-bottom: 6px; }}
  .meta-value {{ display: block; font-size: 1.5rem; color: #fff; font-family: 'Consolas', monospace; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  .data-table th {{ background: #0f3460; color: #64b5f6; padding: 8px 12px;
                    text-align: left; border-bottom: 2px solid #007ACC; }}
  .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #1e1e3e; color: #ccc; }}
  .data-table tr:nth-child(even) {{ background: #13132a; }}
  .data-table tr:hover {{ background: #1e1e3e; }}
  h3 {{ color: #DCDCAA; margin: 24px 0 12px 0; font-size: 1rem; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .toc {{ background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 6px;
          padding: 20px; margin-bottom: 30px; }}
  .toc a {{ color: #64b5f6; text-decoration: none; display: block;
            padding: 4px 0; font-size: .9rem; }}
  .toc a:hover {{ color: #007ACC; }}
  .timestamp {{ color: #555; font-size: .75rem; margin-top: 8px; }}
</style>
</head>
<body>

<div class="header">
  <h1>🧬 Meta Analysis Engine — Full Report</h1>
  <div class="subtitle">Proteomics VAE Meta-Analysis Pipeline v7</div>
  <div style="margin-top:12px;">
    <span class="badge badge-blue">Mode: {pipeline_mode}</span>
    <span class="badge badge-orange">n_samples: {n_samples}</span>
    <span class="badge badge-green">n_proteins: {len(df_final):,}</span>
    <span class="badge badge-blue">SHAP: Ensemble DeepExplainer</span>
    <span class="badge badge-orange">FDR: Bootstrap + BH</span>
  </div>
  <div class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>

<div class="container">

  <div class="toc">
    <strong style="color:#fff;">Contents</strong><br><br>
    <a href="#summary">① Run Summary</a>
    <a href="#scores">② Master Score Distribution</a>
    <a href="#top-proteins">③ Top Proteins</a>
    <a href="#enrichment">④ Pathway Enrichment</a>
    <a href="#volcano">⑤ Volcano Plot</a>
    <a href="#de">⑥ Differential Expression</a>
    <a href="#ppi">⑦ STRING PPI Network</a>
    <a href="#treemap">⑧ GO Treemap</a>
    <a href="#tables">⑨ Full Data Tables</a>
  </div>

  <div class="section" id="summary">
    <div class="section-title">① Run Summary</div>
    {cfg_html}
  </div>

  <div class="section" id="scores">
    <div class="section-title">② Master Score Distribution</div>
    {charts_html.split('</div>', 1)[0] + '</div>' if charts_html else ''}
  </div>

  <div class="section" id="top-proteins">
    <div class="section-title">③ Top Proteins by Master Score</div>
    {"".join(charts_html.split("</div>")[1:3]) + "</div>" if len(charts_html.split("</div>")) > 2 else ""}
  </div>

  <div class="section" id="enrichment">
    <div class="section-title">④ Pathway Enrichment</div>
    {"".join(charts_html.split("</div>")[3:5]) + "</div>" if len(charts_html.split("</div>")) > 4 else ""}
  </div>

  <div class="section" id="volcano">
    <div class="section-title">⑤ Volcano Plot</div>
    {"".join(charts_html.split("</div>")[5:7]) + "</div>" if len(charts_html.split("</div>")) > 6 else ""}
  </div>

  <div class="section" id="de">
    <div class="section-title">⑥ Differential Expression per Cluster</div>
    {"".join(charts_html.split("</div>")[7:9]) + "</div>" if len(charts_html.split("</div>")) > 8 else ""}
  </div>

  <div class="section" id="ppi">
    <div class="section-title">⑦ STRING PPI Network</div>
    {"".join(charts_html.split("</div>")[9:11]) + "</div>" if len(charts_html.split("</div>")) > 10 else ""}
  </div>

  <div class="section" id="treemap">
    <div class="section-title">⑧ Gene Ontology Treemap</div>
    {"".join(charts_html.split("</div>")[11:13]) + "</div>" if len(charts_html.split("</div>")) > 12 else ""}
  </div>

  <div class="section" id="tables">
    <div class="section-title">⑨ Full Data Tables</div>
    {top_drivers_html}
    <br>
    {enr_html}
    <br>
    {de_html}
    <br>
    {ppi_html}
  </div>

</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("HTML report saved: %s", output_path)
        return output_path

    except Exception as exc:
        logger.error("HTML report failed: %s", exc)
        return ""


def run_pipeline_initial(
    df_raw: pd.DataFrame,
    config_dict: dict,
    status_callback: Callable | None = None,
) -> dict:
    config = PipelineConfig.from_dict(config_dict)
    rng    = np.random.default_rng(config.random_seed)
    tf.random.set_seed(config.random_seed)

    def log(msg, prog=None):
        logger.info(msg)
        if status_callback:
            status_callback(msg, prog)

    WORK_DIR = config.work_dir
    os.makedirs(WORK_DIR, exist_ok=True)
    log(f"Starting. Output: {WORK_DIR}", 0.0)

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    log("Step 1: Parsing …", 0.02)
    is_long = ("Run" in df_raw.columns and
               ("Protein.Group" in df_raw.columns or "Protein.Ids" in df_raw.columns))
    if is_long:
        feat = "Protein.Group" if "Protein.Group" in df_raw.columns else "Protein.Ids"
        val  = "PG.MaxLFQ"     if "PG.MaxLFQ"     in df_raw.columns else "Precursor.Quantity"
        try:
            df = df_raw.pivot_table(index=feat, columns="Run", values=val, aggfunc="max")
        except Exception:
            df = df_raw.select_dtypes(include=[np.number])
    else:
        df = df_raw.select_dtypes(include=[np.number])
    df = df.replace(0, np.nan).dropna(how="all")

    # ── Determine pipeline mode early ────────────────────────────────────────
    n_samples     = df.shape[1]
    pipeline_mode = get_pipeline_mode(n_samples)
    mode_info     = MODE_DESCRIPTIONS[pipeline_mode]
    log(f"Pipeline mode: {mode_info['label']} "
        f"(network={mode_info['network']}, "
        f"clustering={mode_info['clustering']}, "
        f"viz={mode_info['viz']})", 0.03)

    # ── 2. Normalise + impute ─────────────────────────────────────────────────
    log("Step 2: Log2 normalise + impute …", 0.05)
    df_log = np.log2(df.clip(lower=1e-9))
    for col in df_log.columns:
        mask = df_log[col].isnull()
        if mask.sum():
            mu  = df_log[col].mean()
            std = df_log[col].std()
            if np.isnan(std) or std == 0:
                std = 0.1
            imputed = rng.normal(mu - 1.8 * std, 0.3 * std, mask.sum())
            # FIX: cast to column dtype to avoid FutureWarning
            df_log.loc[mask, col] = imputed.astype(df_log[col].dtype)
    df_log       = df_log.fillna(0.0)
    protein_list = df_log.index.tolist()
    log(f"  {len(protein_list)} proteins × {n_samples} samples. "
        f"Mode: {pipeline_mode}", 0.08)

    # ── 3. Gene mapping ───────────────────────────────────────────────────────
    log("Step 3: Gene symbols …", 0.10)
    base_ids         = [str(p).split("_")[0].split(";")[0] for p in protein_list]
    gene_dict        = get_gene_names_optimized(
        list(set(base_ids)), os.path.join(WORK_DIR, "gene_cache.json"))
    final_gene_names = [gene_dict.get(b, p) for b, p in zip(base_ids, protein_list, strict=False)]
    gene_map_lookup  = dict(zip(protein_list, final_gene_names, strict=False))

    # ── 4. β-VAE ensemble ────────────────────────────────────────────────────
    log("Step 4: β-VAE ensemble …", 0.13)
    K.clear_session()
    X_norm    = RobustScaler().fit_transform(df_log.T).T.astype(np.float32)
    X_norm    = np.nan_to_num(X_norm)
    n_prot    = X_norm.shape[0]
    input_dim = X_norm.shape[1]
    # Adapt latent_dim to available samples
    latent_dim = min(config.latent_dim, max(2, input_dim - 1))
    batch_size = max(8, min(64, n_prot // 8))
    log(f"  {n_prot}×{input_dim} | batch={batch_size} | latent_dim={latent_dim}", 0.14)

    vaes, encoders, latent_accum, recon_errors = [], [], [], []
    for i in range(config.iterations):
        log(f"  VAE {i+1}/{config.iterations} …", 0.14 + i * 0.06)
        vae = ProteomicsVAE(input_dim, latent_dim)
        vae.compile(optimizer=Adam(config.learning_rate), loss="mse")
        vae.fit(
            X_norm, X_norm, verbose=0,
            epochs=config.epochs, batch_size=batch_size,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                KLWarmupCallback(vae.beta, config.beta_vae, 0.20),
            ],
        )
        enc = vae.get_encoder()
        vaes.append(vae); encoders.append(enc)
        latent_accum.append(enc.predict(X_norm, verbose=0))
        recon = vae.predict(X_norm, verbose=0)
        recon_errors.append(np.mean(np.square(X_norm - recon), axis=1))

    final_latent = np.mean(latent_accum, axis=0)
    final_mse    = np.mean(recon_errors, axis=0)

    if len(recon_errors) > 1:
        es           = np.stack(recon_errors)
        stability_df = pd.DataFrame({
            "Protein_ID":       protein_list,
            "Recon_Error_CV":   es.std(0) / (es.mean(0) + 1e-9),
            "Recon_Error_Mean": es.mean(0),
        })
    else:
        stability_df = pd.DataFrame()

    # ── 5. SHAP ───────────────────────────────────────────────────────────────
    log("Step 5: Ensemble SHAP …", 0.36)
    shap_vals = compute_shap_importance(encoders, X_norm, rng=rng)

    # ── 6. Latent entropy ─────────────────────────────────────────────────────
    log("Step 6: Latent-space Shannon entropy …", 0.43)
    latent_entropy = compute_latent_entropy(vaes, X_norm)

    # ── 7. Expression entropy ─────────────────────────────────────────────────
    log("Step 7: Expression Shannon entropy …", 0.47)
    expression_entropy = compute_expression_entropy(X_norm)

    # ── 8. Network (adaptive) ─────────────────────────────────────────────────
    log(f"Step 8: Network ({mode_info['network']}) …", 0.50)
    G, pcorr = build_network(X_norm, protein_list, pipeline_mode, config.pcorr_threshold)

    # ── 9. Centrality ─────────────────────────────────────────────────────────
    log("Step 9: Centrality …", 0.55)
    centrality_df = calculate_network_centrality(G, protein_list).reindex(protein_list).fillna(0.0)
    degrees       = [G.degree(p) if G.has_node(p) else 0 for p in protein_list]

    # ── 10. Clustering (adaptive) ─────────────────────────────────────────────
    log(f"Step 10: Clustering ({mode_info['clustering']}) …", 0.59)
    cluster_labels = cluster_proteins_adaptive(final_latent, protein_list, G, config, pipeline_mode)

    # ── 11. Stats + p-values (adaptive) ──────────────────────────────────────
    log(f"Step 11: Stats + {mode_info['pvalues']} …", 0.62)
    z_scores, raw_p, bh_p = calculate_robust_stats(final_mse)
    if pipeline_mode in ("ultra_sparse", "low_sample"):
        log("  Using bootstrap p-values (low sample count) …", 0.63)
        bh_p = calculate_bootstrap_pvalues(final_mse, n_bootstrap=2000, rng=rng)

    # ── 12. PIPS ──────────────────────────────────────────────────────────────
    log("Step 12: PIPS …", 0.66)
    pips_scores = compute_pips(G, protein_list, final_mse, alpha=config.pips_alpha)

    # ── 13. Jackknife stability (new in v6) ───────────────────────────────────
    jackknife_stability = np.ones(len(protein_list))
    if config.jackknife and n_samples >= 2:
        log("Step 13a: Jackknife stability …", 0.67)
        jackknife_stability = compute_jackknife_stability(X_norm, protein_list, config)
    else:
        log("Step 13a: Jackknife skipped (disabled or single sample).", 0.67)

    # ── 13b. Pathway enrichment ───────────────────────────────────────────────
    log("Step 13b: Pathway enrichment …", 0.70)
    enrichment_full_df = pd.DataFrame()
    if ADVANCED_LIBS:
        try:
            reports = []
            for c_id in [c for c in np.unique(cluster_labels) if c != -1]:
                genes = [final_gene_names[i]
                         for i in np.where(cluster_labels == c_id)[0]
                         if isinstance(final_gene_names[i], str) and len(final_gene_names[i]) > 1][:300]
                if len(genes) > 5:
                    enr = gp.enrichr(
                        gene_list=genes,
                        gene_sets=config.gene_sets,
                        organism="human",
                    ).results
                    enr["Cluster"] = c_id
                    reports.append(enr[enr["Adjusted P-value"] < 0.1].head(10))
            if reports:
                enrichment_full_df = pd.concat(reports, ignore_index=True)
                enrichment_full_df.to_csv(
                    os.path.join(WORK_DIR, "Cluster_Enrichment_Reports.csv"), index=False)
        except Exception as exc:
            logger.warning("Enrichment failed: %s", exc)

    # ── 14. Embedding (adaptive) ──────────────────────────────────────────────
    log(f"Step 14: Embedding ({mode_info['viz']}) …", 0.76)
    embedding_2d, embedding_3d, embedding_method = compute_embedding(
        final_latent, pipeline_mode, config.random_seed)

    # ── Pseudotime ────────────────────────────────────────────────────────────
    pseudotime = np.zeros(len(protein_list))
    if embedding_2d is not None and len(protein_list) > 2:
        try:
            from sklearn.neighbors import kneighbors_graph
            k     = min(10, len(protein_list) - 1)
            knn_g = kneighbors_graph(embedding_2d, n_neighbors=k,
                                     mode="distance", include_self=False)
            G_knn = nx.from_scipy_sparse_array(knn_g)
            root  = int(np.argmin(final_mse))
            if nx.is_connected(G_knn):
                lengths = nx.single_source_dijkstra_path_length(G_knn, root)
                geo     = np.array([lengths.get(i, np.nan) for i in range(len(protein_list))])
            else:
                geo = embedding_2d[:, 0]
            nan_mask = np.isnan(geo)
            if nan_mask.any():
                geo[nan_mask] = np.nanmax(geo)
            pseudotime = (geo - geo.min()) / (geo.max() - geo.min() + 1e-9)
        except Exception as exc:
            logger.warning("Pseudotime failed: %s", exc)

    # ── Topology mapper (full mode only) ──────────────────────────────────────
    if ADVANCED_LIBS and pipeline_mode == "full":
        try:
            mapper   = km.KeplerMapper(verbose=0)
            graph_km = mapper.map(
                mapper.fit_transform(embedding_2d, projection=[0, 1]),
                final_latent,
                cover=km.Cover(n_cubes=10, perc_overlap=0.2),
            )
            mapper.visualize(graph_km,
                             path_html=os.path.join(WORK_DIR, "DataDriven_Topology.html"),
                             title="Meta Analysis Topology")
        except Exception as exc:
            logger.warning("KeplerMapper failed: %s", exc)

    # ── 15. Trajectory (adaptive) ─────────────────────────────────────────────
    log("Step 15: Trajectory …", 0.84)
    traj_df = analyze_trajectory_drivers_safe(
        df_log, pseudotime, protein_list, mode=pipeline_mode)

    # ── 16. Visualisations ────────────────────────────────────────────────────
    log("Step 16: Visualisations …", 0.88)
    if ADVANCED_LIBS:
        try:
            plot_3d_protein_atlas(
                embedding_3d, protein_list, cluster_labels, final_gene_names,
                method_label=embedding_method,
            ).write_html(os.path.join(WORK_DIR, f"3D_Protein_Atlas_{embedding_method}.html"))
            plot_2d_embedding_drivers(
                embedding_2d, protein_list, cluster_labels, final_gene_names, shap_vals,
                method_label=embedding_method,
            ).write_html(os.path.join(WORK_DIR, f"2D_{embedding_method}_SHAP_Drivers.html"))
        except Exception as exc:
            logger.warning("Viz save failed: %s", exc)

    # ── 17. Assemble ──────────────────────────────────────────────────────────
    log("Step 17: Assembling …", 0.94)
    recon_cv = stability_df["Recon_Error_CV"].values if not stability_df.empty else np.zeros(len(protein_list))

    def _1d(arr, n, fill=0.0):
        """Guarantee a 1-D numpy array of length n."""
        a = np.asarray(arr).flatten()
        if len(a) == n:
            return a
        logger.warning("Array length mismatch: got %d, expected %d — filling with %s", len(a), n, fill)
        return np.full(n, fill, dtype=float)

    n = len(protein_list)
    df_base = pd.DataFrame({
        "Protein_ID":              protein_list,
        "Gene_Symbol":             final_gene_names,
        "Reconstruction_Error":    _1d(final_mse,           n),
        "Reconstruction_Error_CV": _1d(recon_cv,            n),
        "SHAP_Importance":         _1d(shap_vals,           n),
        "Latent_Entropy":          _1d(latent_entropy,      n),
        "Expression_Entropy":      _1d(expression_entropy,  n),
        "PIPS_Score":              _1d(pips_scores,         n),
        "Jackknife_Stability":     _1d(jackknife_stability, n, fill=1.0),
        "Latent_Connectivity":     _1d(degrees,             n),
        "Eigenvector_Centrality":  _1d(centrality_df["Eigenvector_Centrality"].values, n),
        "Betweenness_Centrality":  _1d(centrality_df["Betweenness_Centrality"].values, n),
        "Cluster_ID":              _1d(cluster_labels,      n, fill=-1),
        "Z_Score":                 _1d(z_scores,            n),
        "P_Value":                 _1d(raw_p,               n, fill=1.0),
        "P_Value_BH":              _1d(bh_p,                n, fill=1.0),
        "Pseudotime":              _1d(pseudotime,          n),
    }).set_index("Protein_ID")

    df_base = df_base.join(traj_df, how="left")
    df_base.reset_index(inplace=True)

    # ── 18. Differential Expression ───────────────────────────────────────────
    log("Step 18: Differential expression …", 0.95)
    df_de = pd.DataFrame()
    try:
        df_de = compute_differential_expression(
            df_log, cluster_labels, protein_list, final_gene_names)
        if not df_de.empty:
            df_de.to_csv(os.path.join(WORK_DIR, "Differential_Expression.csv"), index=False)
    except Exception as exc:
        logger.warning("Differential expression failed: %s", exc)

    # ── 19. STRING PPI Network ────────────────────────────────────────────────
    log("Step 19: STRING PPI network …", 0.97)
    df_ppi = pd.DataFrame()
    try:
        shap_flat = np.asarray(shap_vals).flatten()
        n_genes   = len(final_gene_names)
        top_idx   = [int(i) for i in np.argsort(shap_flat)[-100:] if int(i) < n_genes]
        top_genes = [final_gene_names[i] for i in top_idx
                     if isinstance(final_gene_names[i], str) and len(final_gene_names[i]) > 1]
        df_ppi = fetch_string_interactions(top_genes, score_threshold=400)
        if not df_ppi.empty:
            df_ppi.to_csv(os.path.join(WORK_DIR, "STRING_PPI_Network.csv"), index=False)
    except Exception as exc:
        logger.warning("STRING PPI failed: %s", exc)

    state = {
        "protein_list":     protein_list,
        "gene_names":       final_gene_names,
        "gene_map":         gene_map_lookup,
        "latent_space":     final_latent,
        "pcorr_matrix":     pcorr,
        "phate_2d":         embedding_2d,
        "phate_3d":         embedding_3d,
        "embedding_2d":     embedding_2d,
        "embedding_3d":     embedding_3d,
        "embedding_method": embedding_method,
        "df_base":          df_base,
        "df_de":            df_de,
        "df_ppi":           df_ppi,
        "G_initial":        G,
        "work_dir":         WORK_DIR,
        "enrichment_df":    enrichment_full_df,
        "df_log":           df_log,
        "stability_scores": stability_df,
        "config":           config,
        "pipeline_mode":    pipeline_mode,
        "n_samples":        n_samples,
    }
    log(f"Phase 1 complete. Mode: {pipeline_mode}", 1.0)
    return state


def run_pipeline(df_raw, config_dict, status_callback=None):
    state    = run_pipeline_initial(df_raw, config_dict, status_callback)
    df_final = state["df_base"].copy()

    classes, scores          = run_dynamic_critic(df_final)
    df_final["Anomaly_Class"] = classes
    df_final["ML_Confidence"] = scores
    df_final["Master_Score"]  = calculate_consensus_score(df_final)

    config = state.get("config")
    if config and config.n_bootstrap > 0:
        rng = np.random.default_rng(config.random_seed)
        ci  = compute_bootstrap_ci(df_final["Master_Score"].values,
                                   n_bootstrap=config.n_bootstrap, rng=rng)
        df_final["Master_Score_Mean"]    = ci["mean"]
        df_final["Master_Score_Std"]     = ci["std"]
        df_final["Master_Score_CI_Low"]  = ci["ci_low"]
        df_final["Master_Score_CI_High"] = ci["ci_high"]

    if status_callback:
        status_callback("Driver network …", 0.93)

    sig = df_final[df_final["P_Value_BH"] < 0.05].sort_values("Master_Score", ascending=False)
    sig.to_csv(os.path.join(state["work_dir"], "Significant_Drivers_BH05.csv"), index=False)
    df_final.to_csv(os.path.join(state["work_dir"],
                                 "Meta_Analysis_Comprehensive_Report.csv"), index=False)

    sig_ids = sig["Protein_ID"].tolist()
    G_full  = state["G_initial"]
    if len(sig_ids) > 1 and G_full.number_of_nodes():
        dn = G_full.subgraph(sig_ids).copy()
        if dn.number_of_nodes():
            nx.write_edgelist(dn, os.path.join(state["work_dir"],
                                               "Targeted_Driver_Network.edgelist"))
            visualize_network_static(dn,
                os.path.join(state["work_dir"], "Targeted_Network_Visual.png"),
                state["gene_map"])

    # ── Heatmap data ──────────────────────────────────────────────────────────
    heatmap_df = compute_heatmap_data(state["df_log"], df_final, top_n=50)
    if not heatmap_df.empty:
        heatmap_df.to_csv(os.path.join(state["work_dir"], "Heatmap_Top50.csv"))

    # ── HTML Report ───────────────────────────────────────────────────────────
    state["df_final_full"] = df_final
    generate_html_report(
        state, df_final,
        state.get("df_de", pd.DataFrame()),
        state.get("df_ppi", pd.DataFrame()),
        os.path.join(state["work_dir"], "Meta_Analysis_Full_Report.html"),
    )

    save_detailed_excel(state, df_final, sig,
                        os.path.join(state["work_dir"],
                                     "Meta_Analysis_Detailed_Analysis.xlsx"),
                        config=state["config"])
    return {"df_final": df_final}
