"""
Meta Analysis Engine — app.py  (v6: Sample-Aware Adaptive Pipeline UI)

Changes vs v5:
  • Pipeline mode badge in sidebar and dashboard (ultra_sparse / low_sample /
    moderate / full) with colour coding.
  • Jackknife Stability tab: bar chart + scatter vs Master Score.
  • Bootstrap CI tab: master score distribution with error bars.
  • 3D manifold tab label adapts to embedding method (PHATE / UMAP / PCA).
  • All previous features retained (PIPS, Shannon, Compare, Chatbot, etc.)
"""

import html
import os
import time
import logging

import numpy as np
from io import StringIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.decomposition import PCA

try:
    from streamlit_lottie import st_lottie
    LOTTIE_OK = True
except ImportError:
    LOTTIE_OK = False

from proteomics_meta.engine import (
    run_pipeline_initial,
    run_dynamic_critic,
    calculate_consensus_score,
    compute_bootstrap_ci,
    compute_differential_expression,
    compute_volcano_data,
    compute_heatmap_data,
    fetch_string_interactions,
    build_ppi_network,
    generate_html_report,
    save_detailed_excel,
    SHAP_AVAILABLE,
    LOUVAIN_AVAILABLE,
    ADVANCED_LIBS,
    MODE_DESCRIPTIONS,
    get_pipeline_mode,
)
from proteomics_meta.db import ResultsDB
from proteomics_meta.chatbot import OllamaClient, ProteomicsChatbot, OllamaError

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Meta Analysis Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #1E1E1E; }
    [data-testid="stSidebar"] { background-color: #252526; border-right: 1px solid #333; }
    [data-testid="stHeader"]  { background-color: #1E1E1E; }
    h1,h2,h3,h4,h5,h6 { color:#CCCCCC !important; font-family:'Segoe UI',sans-serif; font-weight:600; }
    p,label,.stMarkdown,.stText,li { color:#D4D4D4 !important; font-family:'Segoe UI',sans-serif; }
    .stCaption { color:#858585 !important; }

    .metric-card { background:#2D2D2D; border:1px solid #454545; border-radius:0;
                   padding:15px; text-align:center; transition:all .2s; }
    .metric-card:hover { border-color:#007ACC; background:#37373D; }
    .metric-title  { color:#AAAAAA; font-size:.8rem; text-transform:uppercase; letter-spacing:.5px; }
    .metric-value  { color:#FFF; font-size:1.8rem; font-weight:500; font-family:'Consolas',monospace; }
    .metric-accent { color:#007ACC; font-size:.85rem; }

    div.stButton>button { background:#007ACC; color:white; border:none;
                          padding:.5rem 1rem; border-radius:2px; }
    div.stButton>button:hover  { background:#0062A3; }
    div.stButton>button:active { background:#005A9E; }

    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stMultiSelect>div>div>div {
        background:#3C3C3C !important; color:#CCCCCC !important;
        border:1px solid #3C3C3C !important; border-radius:2px; }
    .stTextInput>div>div>input:focus { border:1px solid #007ACC !important; }

    .stTabs [data-baseweb="tab-list"] { gap:20px; background:transparent; border-bottom:1px solid #2D2D2D; }
    .stTabs [data-baseweb="tab"]      { height:40px; color:#969696; background:transparent; border:none; border-radius:0; }
    .stTabs [aria-selected="true"]    { background:transparent !important; color:#FFF !important;
                                        border-bottom:2px solid #007ACC !important; }

    ::-webkit-scrollbar       { width:12px; height:12px; }
    ::-webkit-scrollbar-track { background:#1E1E1E; }
    ::-webkit-scrollbar-thumb { background:#424242; }

    .stProgress>div>div>div>div { background:#007ACC; }

    .chat-user      { background:#2D2D2D; border-left:3px solid #007ACC;
                      padding:10px 14px; margin:6px 0; border-radius:0 6px 6px 0; }
    .chat-assistant { background:#252526; border-left:3px solid #DCDCAA;
                      padding:10px 14px; margin:6px 0; border-radius:0 6px 6px 0; }
    .chat-label     { font-size:.72rem; text-transform:uppercase; letter-spacing:.5px;
                      color:#858585; margin-bottom:4px; }

    .run-card { background:#2D2D2D; border:1px solid #3a3a3a; padding:10px 14px;
                margin:4px 0; border-radius:4px; }
    .run-card:hover { border-color:#007ACC; }

    .delta-pos { color:#4CAF50; font-weight:600; }
    .delta-neg { color:#F44747; font-weight:600; }

    .dot-green  { display:inline-block; width:8px; height:8px; background:#4CAF50;
                  border-radius:50%; margin-right:6px; }
    .dot-red    { display:inline-block; width:8px; height:8px; background:#F44747;
                  border-radius:50%; margin-right:6px; }

    .badge { padding:2px 8px; border-radius:3px; font-size:.75rem; }
    .badge-blue   { background:#1e3a5f; color:#64b5f6; }
    .badge-green  { background:#1a3a2a; color:#66bb6a; }
    .badge-orange { background:#3a2a1a; color:#ffa726; }
    .badge-purple { background:#2a1a3a; color:#ce93d8; }
    .badge-red    { background:#3a1a1a; color:#ef9a9a; }

    .mode-banner { background:#2D2D2D; border:1px solid #454545;
                   border-radius:4px; padding:10px 16px; margin-bottom:12px; }
    .mode-banner-title { color:#AAAAAA; font-size:.75rem; text-transform:uppercase;
                         letter-spacing:.5px; margin-bottom:4px; }
    .mode-banner-value { color:#FFF; font-size:1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SINGLETONS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db(path: str = None) -> ResultsDB:
    if path is None:
        path = os.environ.get("SQLITE_DB_PATH", "meta_analysis.db")
    return ResultsDB(path)

@st.cache_resource
def get_ollama_client(base_url: str = "http://localhost:11434") -> OllamaClient:
    return OllamaClient(base_url)


db     = get_db()
ollama = get_ollama_client()

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_PLOTLY_DARK = dict(
    template="plotly_dark", paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
    font=dict(color="#D4D4D4", family="Segoe UI"),
)
_CLASS_COLORS = {
    "Validated_Signal":     "#007ACC",
    "Biological_Discovery": "#DCDCAA",
    "Technical_Noise":      "#F44747",
}
_CRITIC_COLS = [
    "Reconstruction_Error", "Latent_Connectivity", "Eigenvector_Centrality",
    "SHAP_Importance", "Reconstruction_Error_CV", "Z_Score",
    "Latent_Entropy", "Expression_Entropy",
]


def pdark(fig, **extra):
    base = dict(**_PLOTLY_DARK)
    base["xaxis"] = dict(gridcolor="#333")
    base["yaxis"] = dict(gridcolor="#333")
    base["height"] = 520
    base.update(extra)
    fig.update_layout(**base)
    return fig


def render_metric(col, title, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-accent">{sub}</div>
    </div>""", unsafe_allow_html=True)


def render_mode_banner(mode: str, n_samples: int):
    info  = MODE_DESCRIPTIONS.get(mode, {})
    label = info.get("label", mode)
    color_map = {
        "ultra_sparse": "#ffa726",
        "low_sample":   "#ffa726",
        "moderate":     "#64b5f6",
        "full":         "#66bb6a",
    }
    color = color_map.get(mode, "#CCCCCC")
    st.markdown(f"""
    <div class="mode-banner">
        <div class="mode-banner-title">Pipeline Mode — {n_samples} sample(s) detected</div>
        <div class="mode-banner-value" style="color:{color}">⚙ {label}</div>
        <div style="margin-top:6px;font-size:.8rem;color:#858585;">
            Network: <b style="color:#CCCCCC">{info.get('network','?')}</b> &nbsp;|&nbsp;
            Clustering: <b style="color:#CCCCCC">{info.get('clustering','?')}</b> &nbsp;|&nbsp;
            Visualisation: <b style="color:#CCCCCC">{info.get('viz','?')}</b> &nbsp;|&nbsp;
            p-values: <b style="color:#CCCCCC">{info.get('pvalues','?')}</b> &nbsp;|&nbsp;
            Trajectory: <b style="color:#CCCCCC">{info.get('trajectory','?')}</b>
        </div>
    </div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_lottie(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _ensure_title_case(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        title = "_".join(w.capitalize() for w in col.split("_"))
        if title != col and title in _CRITIC_COLS:
            rename[col] = title
    return df.rename(columns=rename) if rename else df


@st.cache_data(show_spinner=False)
def cached_critic(df_json: str, contamination: float, net_sens: float):
    df = pd.read_json(StringIO(df_json), orient="split")
    df = _ensure_title_case(df)
    return run_dynamic_critic(df, contamination=contamination,
                              network_sensitivity=net_sens)


@st.cache_data(show_spinner=False)
def cached_consensus(df_json: str):
    df = pd.read_json(StringIO(df_json), orient="split")
    df = _ensure_title_case(df)
    return calculate_consensus_score(df).tolist()


@st.cache_data(show_spinner=False)
def cached_pca(latent_json: str):
    arr = np.array(pd.read_json(latent_json, orient="split"))
    n   = arr.shape[0]
    n_c = min(3, n - 1, arr.shape[1])
    coords = PCA(n_components=n_c).fit_transform(arr)
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((n, 3 - coords.shape[1]))])
    return coords


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

lottie_dna = load_lottie("https://assets5.lottiefiles.com/packages/lf20_w5h9ryub.json")

with st.sidebar:
    st.markdown("### 🧬 **Meta Analysis** `v6`")
    st.caption("Workspace: Proteomics Analysis")
    shap_lbl    = "DeepExplainer" if SHAP_AVAILABLE    else "L2-norm proxy"
    cluster_lbl = "Louvain"       if LOUVAIN_AVAILABLE else "GMM"
    st.markdown(
        f'<span class="badge badge-blue">SHAP: {shap_lbl}</span> '
        f'<span class="badge badge-green">Cluster: {cluster_lbl}</span> '
        f'<span class="badge badge-orange">FDR: BH</span> '
        f'<span class="badge badge-purple">PIPS: ✓</span> '
        f'<span class="badge badge-purple">JK: ✓</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload Data", type=["tsv", "csv", "parquet", "txt"])
    result_dir    = st.text_input("Output directory", value="Meta_Analysis_Results")
    run_name      = st.text_input("Run name (optional)", value="",
                                   placeholder="e.g. condition_A_vs_B")

    with st.expander("⚙️ Runtime Config", expanded=False):
        iterations    = st.slider("VAE iterations",  1, 10, 3)
        epochs        = st.slider("Epochs",         50, 300, 60)
        latent_dim    = st.slider("Latent dim",      2, 50, 10)
        learning_rate = st.select_slider("Learning rate",
                            [0.01, 0.005, 0.002, 0.001, 0.0001], value=0.002)
        beta_vae      = st.slider("β-VAE weight", 0.1, 4.0, 1.0, 0.1)
        use_louvain   = st.checkbox("Use Louvain clustering", value=LOUVAIN_AVAILABLE)
        manual_clust  = st.checkbox("Manual cluster count")
        n_clusters    = st.slider("Cluster count", 2, 30, 10) if manual_clust else None

    with st.expander("🔬 GSEA Libraries", expanded=False):
        gsea_libs = st.multiselect("Libraries",
            ["KEGG_2021_Human", "GO_Biological_Process_2021",
             "MSigDB_Hallmark_2020", "Reactome_2022", "WikiPathways_2021_Human"],
            default=["KEGG_2021_Human", "GO_Biological_Process_2021"])

    with st.expander("🎚️ Critic & Network", expanded=False):
        contamination   = st.slider("Contamination",        0.01, 0.30, 0.10, 0.01)
        net_sens        = st.slider("Network sensitivity",  0.0,  1.0,  0.5,  0.05)
        pcorr_threshold = st.slider("|pcorr| edge threshold", 0.05, 0.50, 0.10, 0.01)
        pips_alpha      = st.slider("PIPS diffusion α",     0.1,  0.9,  0.5,  0.05)

    with st.expander("🔁 Robustness", expanded=False):
        do_jackknife = st.checkbox("Jackknife stability", value=True,
            help="Leave-one-out stability scoring. Adds Jackknife_Stability column.")
        n_bootstrap  = st.slider("Bootstrap CI iterations", 0, 2000, 500, 100,
            help="0 = disabled. Computes confidence intervals for Master Score.")

    run_btn = False
    if uploaded_file and "pipeline_state" not in st.session_state:
        run_btn = st.button("▶ RUN ANALYSIS", width='stretch')

    # ── Run history ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗄️ Run History")
    all_runs = db.list_runs()

    if not all_runs:
        st.caption("No saved runs yet.")
    else:
        for run in all_runs[:8]:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<div class="run-card">'
                    f'<b style="color:#CCCCCC">#{run["run_id"]} {run["name"]}</b><br>'
                    f'<span style="color:#858585;font-size:.75rem">'
                    f'{run["created_at"]} · {run["n_proteins"]} proteins</span>'
                    f'</div>', unsafe_allow_html=True)
            with c2:
                if st.button("Load", key=f"load_{run['run_id']}"):
                    st.session_state["loaded_run_id"] = run["run_id"]
                    st.session_state.pop("pipeline_state", None)
                    st.rerun()

        if len(all_runs) > 1:
            st.markdown("---")
            st.caption("Quick compare:")
            run_options = {f"#{r['run_id']} {r['name']}": r["run_id"] for r in all_runs}
            sel_a = st.selectbox("Run A", list(run_options.keys()), key="cmp_a")
            sel_b = st.selectbox("Run B", list(run_options.keys()),
                                  index=min(1, len(run_options) - 1), key="cmp_b")
            if st.button("Compare →", width='stretch'):
                st.session_state["compare_ids"] = [run_options[sel_a], run_options[sel_b]]
                st.rerun()

    if "pipeline_state" in st.session_state:
        st.markdown("---")
        xl_path   = os.path.join(result_dir, "Meta_Analysis_Detailed_Analysis.xlsx")
        html_path = os.path.join(result_dir, "Meta_Analysis_Full_Report.html")
        if os.path.exists(xl_path):
            with open(xl_path, "rb") as f:
                st.download_button("📊 Download Excel", data=f,
                    file_name="Meta_Analysis_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch')
        if os.path.exists(html_path):
            with open(html_path, "rb") as f:
                st.download_button("📄 Download HTML Report", data=f,
                    file_name="Meta_Analysis_Full_Report.html",
                    mime="text/html",
                    width='stretch')
        if st.button("🔄 New Analysis", width='stretch'):
            for k in ["pipeline_state", "results_ready", "_db_run_id"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Ollama ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Ollama")
    ollama_url_input = st.text_input("URL", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                                      key="ollama_url", label_visibility="collapsed")
    ollama         = get_ollama_client(ollama_url_input)
    ollama_running = ollama.is_running()

    if ollama_running:
        st.markdown('<span class="dot-green"></span> Connected', unsafe_allow_html=True)
        models         = ollama.list_models()
        selected_model = st.selectbox("Model", models, key="ollama_model") if models else "llama3"
        if not models:
            st.caption("No models. Run: `ollama pull llama3`")
    else:
        st.markdown('<span class="dot-red"></span> Offline — run `ollama serve`',
                    unsafe_allow_html=True)
        selected_model = "llama3"


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if run_btn and uploaded_file:
    try:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "parquet":
            df_input = pd.read_parquet(uploaded_file)
        elif ext in ("tsv", "txt"):
            df_input = pd.read_csv(uploaded_file, sep="\t", engine="python")
        else:
            df_input = pd.read_csv(uploaded_file, sep=",", engine="python")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    config = {
        "work_dir":         result_dir,
        "iterations":       iterations,
        "epochs":           epochs,
        "latent_dim":       latent_dim,
        "learning_rate":    learning_rate,
        "beta_vae":         beta_vae,
        "n_clusters":       n_clusters,
        "gene_sets":        gsea_libs,
        "use_louvain":      use_louvain,
        "pcorr_threshold":  pcorr_threshold,
        "pips_alpha":       pips_alpha,
        "jackknife":        do_jackknife,
        "n_bootstrap":      n_bootstrap,
    }

    with st.status("🔬 Running analysis …", expanded=True) as status:
        def _cb(msg, prog=None):
            status.write(msg)

        try:
            state = run_pipeline_initial(df_input, config, _cb)
            st.session_state["pipeline_state"] = state
            st.session_state["results_ready"]  = True
            status.update(label="✅ Complete", state="complete", expanded=False)
        except Exception as e:
            import traceback
            status.update(label="❌ Failed", state="error")
            st.error(f"**Error type:** `{type(e).__name__}`")
            st.error(f"**Message:** {e}")
            st.code(traceback.format_exc(), language="python")
            st.stop()

    # rerun OUTSIDE the st.status context — avoids Streamlit treating it as an error
    if st.session_state.get("results_ready"):
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  BUILD df_base FROM ACTIVE PIPELINE STATE
# ──────────────────────────────────────────────────────────────────────────────

df_base = None
p_col   = "P_Value_BH"
state   = None
drivers = validated = discoveries = None

if "pipeline_state" in st.session_state:
    state       = st.session_state["pipeline_state"]
    df_raw_base = state["df_base"].copy()

    classes, scores = cached_critic(df_raw_base.to_json(orient="split"),
                                    contamination, net_sens)
    df_base = df_raw_base.copy()
    df_base["Anomaly_Class"] = classes
    df_base["ML_Confidence"] = scores
    df_base["Master_Score"]  = cached_consensus(df_base.to_json(orient="split"))

    # Bootstrap CI
    config_obj = state.get("config")
    if config_obj and config_obj.n_bootstrap > 0:
        rng_ci = np.random.default_rng(config_obj.random_seed)
        ci     = compute_bootstrap_ci(df_base["Master_Score"].values,
                                      n_bootstrap=config_obj.n_bootstrap,
                                      rng=rng_ci)
        df_base["Master_Score_CI_Low"]  = ci["ci_low"]
        df_base["Master_Score_CI_High"] = ci["ci_high"]
        df_base["Master_Score_Std"]     = ci["std"]

    p_col       = "P_Value_BH" if "P_Value_BH" in df_base.columns else "P_Value"
    drivers     = df_base[df_base[p_col] < 0.05].sort_values("Master_Score", ascending=False)
    validated   = df_base[df_base["Anomaly_Class"] == "Validated_Signal"]
    discoveries = df_base[df_base["Anomaly_Class"] == "Biological_Discovery"]

    # ── Save Excel + DB exactly once ─────────────────────────────────────────
    if "_db_run_id" not in st.session_state:
        save_detailed_excel(
            state, df_base, drivers,
            os.path.join(result_dir, "Meta_Analysis_Detailed_Analysis.xlsx"),
            config=state.get("config"),
        )
        try:
            rid = db.save_run(
                state, df_base, state.get("config"),
                name=run_name or f"Run {time.strftime('%Y-%m-%d %H:%M')}",
                dataset_name=getattr(uploaded_file, "name", "unknown") if uploaded_file else "unknown",
            )
            st.session_state["_db_run_id"] = rid
            st.toast(f"✅ Saved to database (run #{rid})", icon="🗄️")
        except Exception as exc:
            st.warning(f"DB save failed: {exc}")
            st.session_state["_db_run_id"] = "failed"


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

if df_base is not None and state is not None:
    pipeline_mode  = state.get("pipeline_mode", "full")
    n_samples_run  = state.get("n_samples", 0)
    embedding_meth = state.get("embedding_method", "PCA")
    cfg            = state.get("config")

    # Mode banner
    render_mode_banner(pipeline_mode, n_samples_run)

    if cfg:
        with st.expander("ℹ️ Run configuration", expanded=False):
            ca, cb, cc, cd = st.columns(4)
            ca.caption(f"Latent dim: **{cfg.latent_dim}** | Iter: **{cfg.iterations}** | Epochs: **{cfg.epochs}**")
            cb.caption(f"β-VAE: **{cfg.beta_vae}** | SHAP: **{shap_lbl}**")
            cc.caption(f"Network: **{MODE_DESCRIPTIONS[pipeline_mode]['network']}** | "
                       f"Clustering: **{MODE_DESCRIPTIONS[pipeline_mode]['clustering']}**")
            cd.caption(f"p-values: **{MODE_DESCRIPTIONS[pipeline_mode]['pvalues']}** | "
                       f"Jackknife: **{'ON' if cfg.jackknife else 'OFF'}** | "
                       f"Bootstrap: **{cfg.n_bootstrap}**")

    st.markdown("## 🔎 Dashboard Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    render_metric(m1, "FEATURES",    f"{len(df_base):,}",       "Total proteins")
    render_metric(m2, "VALIDATED",   f"{len(validated)}",        "High confidence")
    render_metric(m3, "DISCOVERIES", f"{len(discoveries)}",      "New signals")
    render_metric(m4, "SIG DRIVERS", f"{len(drivers)}",          "BH p < 0.05")
    render_metric(m5, "MASTER SCORE",f"{df_base['Master_Score'].mean():.3f}", "Mean")
    jk_mean = df_base["Jackknife_Stability"].mean() if "Jackknife_Stability" in df_base.columns else 0
    render_metric(m6, "JK STABILITY",f"{jk_mean:.2f}", "Mean stability")
    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🧠 CRITIC",
        f"🌍 3D {embedding_meth}",
        "🎯 TOP DRIVERS",
        "📊 DATA GRID",
        "🧬 ENRICHMENT",
        "🌳 GO TREEMAP",
        "🔥 HEATMAP",
        "🌋 VOLCANO",
        "🔗 STRING PPI",
        "📈 STABILITY",
        "🔬 PIPS",
        "🔁 JACKKNIFE",
        "📉 BOOTSTRAP CI",
        "🆚 COMPARE",
        "🤖 CHATBOT",
    ])

    # ── TAB 0: Critic ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### VAE Reconstruction Error vs. Network Connectivity")
        fig = px.scatter(
            df_base, x="Reconstruction_Error", y="Latent_Connectivity",
            color="Anomaly_Class", size="Master_Score",
            hover_data=["Gene_Symbol", "Cluster_ID", p_col],
            color_discrete_map=_CLASS_COLORS, log_x=True,
        )
        st.plotly_chart(pdark(fig), width='stretch')

    # ── TAB 1: 3D Manifold ────────────────────────────────────────────────────
    with tabs[1]:
        emb_3d = state.get("embedding_3d") if state.get("embedding_3d") is not None else state.get("phate_3d")
        if emb_3d is not None:
            col_names  = [f"{embedding_meth}_1", f"{embedding_meth}_2", f"{embedding_meth}_3"]
            title_3d   = f"3D Protein Atlas — {embedding_meth} manifold"
        else:
            latent_json = pd.DataFrame(state["latent_space"]).to_json(orient="split")
            emb_3d      = cached_pca(latent_json)
            col_names   = ["PC1", "PC2", "PC3"]
            title_3d    = "3D Protein Atlas — PCA fallback"

        df_3d          = pd.DataFrame(emb_3d, columns=col_names)
        df_3d["Class"] = df_base["Anomaly_Class"].values
        df_3d["Gene"]  = df_base["Gene_Symbol"].values

        fig3d = px.scatter_3d(
            df_3d, x=col_names[0], y=col_names[1], z=col_names[2],
            color="Class", hover_name="Gene",
            color_discrete_map={
                "Validated_Signal":     "#007ACC",
                "Biological_Discovery": "#DCDCAA",
                "Technical_Noise":      "rgba(244,71,71,0.2)",
            },
            opacity=0.8, title=title_3d,
        )
        fig3d.update_layout(
            template="plotly_dark", paper_bgcolor="#1E1E1E",
            font=dict(color="#D4D4D4"),
            scene=dict(
                xaxis=dict(backgroundcolor="#1E1E1E", gridcolor="#333"),
                yaxis=dict(backgroundcolor="#1E1E1E", gridcolor="#333"),
                zaxis=dict(backgroundcolor="#1E1E1E", gridcolor="#333"),
            ), height=700,
        )
        st.plotly_chart(fig3d, width='stretch')

    # ── TAB 2: Top Drivers ────────────────────────────────────────────────────
    with tabs[2]:
        cl, cr = st.columns([1, 2])
        with cl:
            top20 = df_base.sort_values("Master_Score", ascending=False).head(20)
            sel   = st.radio("Select:", top20["Gene_Symbol"].tolist(),
                              label_visibility="collapsed")
        with cr:
            row  = df_base[df_base["Gene_Symbol"] == sel].iloc[0]
            st.markdown(f"#### Profile: `{sel}`")
            sig  = min(-np.log10(float(row[p_col]) + 1e-20), 20)
            cats = ["SHAP", "Eigenvector", "–log10(BH p)", "Confidence",
                    "Betweenness", "PIPS", "Latent Entropy", "Expr Specificity",
                    "JK Stability"]
            vals = [
                float(row["SHAP_Importance"]),
                float(row["Eigenvector_Centrality"]) * 10,
                sig,
                float(row["ML_Confidence"]) + 0.5,
                float(row.get("Betweenness_Centrality", 0)) * 100,
                float(row.get("PIPS_Score", 0)) * 10,
                float(row.get("Latent_Entropy", 0)) * 10,
                (1.0 - float(row.get("Expression_Entropy", 0.5))) * 10,
                float(row.get("Jackknife_Stability", 0)) * 10,
            ]
            vm   = max(vals) + 0.01
            disp = [v / vm for v in vals]
            fig_r = go.Figure(go.Scatterpolar(
                r=disp + [disp[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(0,122,204,0.2)",
                line=dict(color="#007ACC"),
            ))
            fig_r.update_layout(
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
                           bgcolor="#1E1E1E"),
                paper_bgcolor="#1E1E1E", font=dict(color="#D4D4D4"), height=380,
            )
            st.plotly_chart(fig_r, width='stretch')
            st.dataframe(
                pd.DataFrame({"Metric": cats, "Value": [f"{v:.4f}" for v in vals]}),
                hide_index=True, width='stretch',
            )

        st.markdown("---")
        st.markdown("##### Top 20 by Master Score")
        show_cols = ["Gene_Symbol", "Master_Score", "PIPS_Score",
                     "Jackknife_Stability", "Anomaly_Class", p_col]
        st.dataframe(
            top20[[c for c in show_cols if c in top20.columns]],
            height=380, width='stretch',
        )

    # ── TAB 3: Data Grid ──────────────────────────────────────────────────────
    with tabs[3]:
        gc1, gc2, gc3 = st.columns(3)
        search = gc1.text_input("Search gene (regex):", "")
        cf     = gc2.multiselect("Cluster:", sorted(df_base["Cluster_ID"].unique()))
        af     = gc3.selectbox("Class:", ["All"] + list(df_base["Anomaly_Class"].unique()))
        disp   = df_base.copy()
        if search:
            disp = disp[disp["Gene_Symbol"].astype(str).str.contains(
                search, case=False, na=False, regex=True)]
        if cf:
            disp = disp[disp["Cluster_ID"].isin(cf)]
        if af != "All":
            disp = disp[disp["Anomaly_Class"] == af]
        priority_cols = [
            "Gene_Symbol", "Anomaly_Class", "Master_Score", "PIPS_Score",
            "Jackknife_Stability", "SHAP_Importance", p_col,
            "Eigenvector_Centrality", "Betweenness_Centrality",
            "Cluster_ID", "Reconstruction_Error",
        ]
        shown = [c for c in priority_cols if c in disp.columns]
        rest  = [c for c in disp.columns if c not in shown]
        st.dataframe(disp[shown + rest], width='stretch', height=560)

    # ── TAB 4: Enrichment ─────────────────────────────────────────────────────
    with tabs[4]:
        enr_path = os.path.join(result_dir, "Cluster_Enrichment_Reports.csv")
        if os.path.exists(enr_path):
            edf   = pd.read_csv(enr_path)
            c_sel = st.selectbox("Cluster:", sorted(edf["Cluster"].unique()))
            sub   = edf[edf["Cluster"] == c_sel].head(15).copy()
            sub["-log10(adj.p)"] = -np.log10(sub["Adjusted P-value"].clip(lower=1e-20))
            fig_e = px.bar(
                sub.sort_values("-log10(adj.p)"),
                x="-log10(adj.p)", y="Term", color="Odds Ratio",
                color_continuous_scale="Blues", orientation="h",
                title=f"Top pathways — Cluster {c_sel}",
                hover_data=["Combined Score", "Adjusted P-value"],
            )
            fig_e.update_layout(
                **_PLOTLY_DARK,
                height=480, yaxis=dict(autorange="reversed", gridcolor="#333"),
            )
            st.plotly_chart(fig_e, width='stretch')
        else:
            st.info("No enrichment data available. Requires gseapy and ADVANCED_LIBS.")

    # ── TAB 5: GO Treemap ─────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### 🌳 Gene Ontology Treemap")
        enr_path = os.path.join(result_dir, "Cluster_Enrichment_Reports.csv")
        if os.path.exists(enr_path):
            edf_tree = pd.read_csv(enr_path)
            edf_tree["Cluster_Label"] = "Cluster " + edf_tree["Cluster"].astype(str)
            edf_tree["Score"] = edf_tree["Combined Score"].clip(lower=0.1)
            fig_tree = px.treemap(
                edf_tree, path=["Cluster_Label", "Gene_set", "Term"],
                values="Score", color="Adjusted P-value",
                color_continuous_scale="RdBu_r",
                title="GO Treemap — size=Combined Score, color=adj.p-value",
            )
            fig_tree.update_layout(paper_bgcolor="#1E1E1E", height=600,
                                   font=dict(color="#D4D4D4"))
            st.plotly_chart(fig_tree, width='stretch')

            # Cluster comparison bar chart
            st.markdown("---")
            st.markdown("##### Pathway comparison across clusters")
            fig_cmp = px.bar(
                edf_tree.sort_values("Combined Score", ascending=False).head(30),
                x="Combined Score", y="Term", color="Cluster_Label",
                orientation="h", barmode="group",
                title="Top Pathways — Cluster Comparison",
            )
            fig_cmp.update_layout(
                **_PLOTLY_DARK,
                height=600, yaxis=dict(autorange="reversed", gridcolor="#333"),
            )
            st.plotly_chart(fig_cmp, width='stretch')
        else:
            st.info("No enrichment data available.")

    # ── TAB 6: Heatmap ────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("#### 🔥 Expression Heatmap — Top 50 Proteins")
        st.caption("Z-score normalised log2 expression. Rows = proteins, Columns = samples.")
        hm_path = os.path.join(result_dir, "Heatmap_Top50.csv")
        if os.path.exists(hm_path):
            hm_df = pd.read_csv(hm_path, index_col=0)
            fig_hm = px.imshow(
                hm_df, aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Expression Heatmap (Z-score)",
                labels=dict(x="Sample", y="Protein", color="Z-score"),
            )
            fig_hm.update_layout(
                template="plotly_dark", paper_bgcolor="#1E1E1E",
                font=dict(color="#D4D4D4"), height=max(400, len(hm_df) * 12),
            )
            st.plotly_chart(fig_hm, width='stretch')
        else:
            # Compute on-the-fly from state
            if state and "df_log" in state:
                hm_df = compute_heatmap_data(state["df_log"], df_base, top_n=50)
                if not hm_df.empty:
                    fig_hm = px.imshow(
                        hm_df, aspect="auto", color_continuous_scale="RdBu_r",
                        title="Expression Heatmap (Z-score)",
                        labels=dict(x="Sample", y="Protein", color="Z-score"),
                    )
                    fig_hm.update_layout(
                        template="plotly_dark", paper_bgcolor="#1E1E1E",
                        font=dict(color="#D4D4D4"),
                        height=max(400, len(hm_df) * 12),
                    )
                    st.plotly_chart(fig_hm, width='stretch')
                else:
                    st.info("Could not compute heatmap.")
            else:
                st.info("Heatmap not available.")

    # ── TAB 7: Volcano Plot ───────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("#### 🌋 Volcano Plot — Differential Expression")
        st.caption("x = Log2 Fold Change (cluster vs rest), y = −log10(BH p-value). "
                   "Dashed lines: |Log2FC| > 0.5 and p < 0.05.")
        de_path = os.path.join(result_dir, "Differential_Expression.csv")
        df_de_ui = pd.DataFrame()
        if os.path.exists(de_path):
            df_de_ui = pd.read_csv(de_path)
        elif state and "df_de" in state:
            df_de_ui = state["df_de"]

        if not df_de_ui.empty:
            clusters_de = sorted(df_de_ui["Cluster_ID"].unique())
            sel_cluster = st.selectbox("Cluster for Volcano:", clusters_de, key="vol_cluster")
            vol_data = compute_volcano_data(
                df_base, df_de_ui[df_de_ui["Cluster_ID"] == sel_cluster])
            if not vol_data.empty:
                vol_data["Significant"] = (
                    (vol_data[p_col] < 0.05) & (vol_data["Abs_Log2FC"] > 0.5))
                vol_data["Label"] = vol_data.apply(
                    lambda r: r["Gene_Symbol"] if r["Significant"] else "", axis=1)
                fig_vol = px.scatter(
                    vol_data, x="Log2FC", y="Neg_Log10_P",
                    color="Anomaly_Class", hover_name="Gene_Symbol",
                    text="Label",
                    color_discrete_map=_CLASS_COLORS,
                    title=f"Volcano — Cluster {sel_cluster} vs rest",
                    labels={"Log2FC": "Log2 Fold Change",
                            "Neg_Log10_P": "−log10(BH p-value)"},
                )
                fig_vol.add_hline(y=-np.log10(0.05), line_dash="dash",
                                  line_color="#888", annotation_text="p=0.05")
                fig_vol.add_vline(x=0.5,  line_dash="dash", line_color="#888")
                fig_vol.add_vline(x=-0.5, line_dash="dash", line_color="#888")
                fig_vol.update_traces(textposition="top center", textfont_size=9)
                st.plotly_chart(pdark(fig_vol, height=550), width='stretch')

                sig_vol = vol_data[vol_data["Significant"]].sort_values(
                    "Abs_Log2FC", ascending=False)
                st.markdown(f"**{len(sig_vol)} significant proteins** (|Log2FC|>0.5, p<0.05)")
                st.dataframe(sig_vol[[c for c in ["Gene_Symbol","Log2FC","Neg_Log10_P",
                                                   "Anomaly_Class","Master_Score"] if c in sig_vol.columns]
                             ].reset_index(drop=True),
                             width='stretch', hide_index=True)
        else:
            st.info("Run the analysis to generate differential expression data.")

    # ── TAB 8: STRING PPI ─────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown("#### 🔗 STRING Protein-Protein Interaction Network")
        st.caption("Interactions from STRING database (score ≥ 400). "
                   "Node size = Master Score. Colors = Anomaly Class.")
        ppi_path = os.path.join(result_dir, "STRING_PPI_Network.csv")
        df_ppi_ui = pd.DataFrame()
        if os.path.exists(ppi_path):
            df_ppi_ui = pd.read_csv(ppi_path)
        elif state and "df_ppi" in state:
            df_ppi_ui = state["df_ppi"]

        if not df_ppi_ui.empty:
            min_score = st.slider("Min STRING score", 0, 1000, 400, 50, key="ppi_score")
            df_ppi_filt = df_ppi_ui[df_ppi_ui["score"] >= min_score / 1000.0
                                     if df_ppi_ui["score"].max() <= 1.0
                                     else df_ppi_ui["score"] >= min_score]
            top_n_ppi = st.slider("Top N proteins", 10, 100, 30, 5, key="ppi_topn")

            top_genes_ppi = set(df_base.sort_values("Master_Score", ascending=False)
                                .head(top_n_ppi)["Gene_Symbol"])
            df_ppi_top = df_ppi_filt[
                df_ppi_filt["gene_a"].isin(top_genes_ppi) |
                df_ppi_filt["gene_b"].isin(top_genes_ppi)
            ].head(200)

            if not df_ppi_top.empty:
                G_ppi = build_ppi_network(df_ppi_top, df_base)
                pos   = nx.spring_layout(G_ppi, seed=42, k=0.6)
                score_m = dict(zip(df_base["Gene_Symbol"], df_base["Master_Score"]))
                class_m = dict(zip(df_base["Gene_Symbol"], df_base["Anomaly_Class"]))

                edge_x, edge_y = [], []
                for e in G_ppi.edges():
                    x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
                    edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

                node_colors = [
                    "#007ACC" if class_m.get(n) == "Validated_Signal"
                    else "#DCDCAA" if class_m.get(n) == "Biological_Discovery"
                    else "#F44747" for n in G_ppi.nodes()
                ]
                node_sizes  = [max(8, score_m.get(n, 0) * 40) for n in G_ppi.nodes()]

                fig_ppi = go.Figure()
                fig_ppi.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(color="#444", width=0.8), hoverinfo="none"))
                fig_ppi.add_trace(go.Scatter(
                    x=[pos[n][0] for n in G_ppi.nodes()],
                    y=[pos[n][1] for n in G_ppi.nodes()],
                    mode="markers+text",
                    marker=dict(size=node_sizes, color=node_colors,
                                line=dict(width=1, color="#1E1E1E")),
                    text=list(G_ppi.nodes()),
                    textposition="top center",
                    textfont=dict(size=9, color="#DDD"),
                    hovertext=[f"{n}<br>Score: {score_m.get(n,0):.3f}<br>{class_m.get(n,'?')}"
                               for n in G_ppi.nodes()],
                    hoverinfo="text",
                ))
                fig_ppi.update_layout(
                    template="plotly_dark", paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                    height=600, title=f"STRING PPI — Top {top_n_ppi} proteins",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    font=dict(color="#D4D4D4"),
                )
                st.plotly_chart(fig_ppi, width='stretch')
                st.markdown(f"**{G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges**")
                st.dataframe(df_ppi_top.sort_values("score", ascending=False)
                             .reset_index(drop=True),
                             width='stretch', hide_index=True)
            else:
                st.info("No interactions found for selected top proteins.")
        else:
            st.info("STRING PPI data not available. "
                    "Check network connectivity and re-run the analysis.")

    # ── TAB 9: Stability ──────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown("#### VAE Reconstruction Stability (CV across iterations)")
        if "Reconstruction_Error_CV" in df_base.columns:
            fig_cv = px.histogram(
                df_base, x="Reconstruction_Error_CV",
                color="Anomaly_Class", nbins=60, barmode="overlay",
                color_discrete_map=_CLASS_COLORS,
                title="CV distribution by class",
            )
            st.plotly_chart(pdark(fig_cv), width='stretch')
            st.dataframe(
                df_base.sort_values("Reconstruction_Error_CV", ascending=False)
                [["Gene_Symbol", "Reconstruction_Error_CV", "Reconstruction_Error",
                  "Anomaly_Class", p_col]].head(30),
                width='stretch',
            )
        else:
            st.info("Stability requires ≥ 2 VAE iterations.")

    # ── TAB 10: PIPS ──────────────────────────────────────────────────────────
    with tabs[10]:
        st.markdown("#### 🔬 Protein Influence Propagation Score (PIPS)")
        st.caption(
            "PIPS measures how much anomaly signal propagates **from** each protein "
            "through the network. High PIPS + low Master Score = candidate master regulator."
        )

        if "PIPS_Score" not in df_base.columns:
            st.warning("PIPS score not available — the network may have had no edges.")
        else:
            pa, pb = st.columns(2)
            with pa:
                st.markdown("##### Top 20 by PIPS Score")
                top_pips    = df_base.sort_values("PIPS_Score", ascending=False).head(20)
                fig_pips_bar = px.bar(
                    top_pips, x="PIPS_Score", y="Gene_Symbol",
                    orientation="h", color="Anomaly_Class",
                    color_discrete_map=_CLASS_COLORS, title="Top PIPS proteins",
                )
                fig_pips_bar.update_layout(
                    **_PLOTLY_DARK,
                    height=500, yaxis=dict(autorange="reversed", gridcolor="#333"),
                )
                st.plotly_chart(fig_pips_bar, width='stretch')

            with pb:
                st.markdown("##### PIPS vs Master Score")
                fig_pips_sc = px.scatter(
                    df_base, x="Master_Score", y="PIPS_Score",
                    color="Anomaly_Class", hover_name="Gene_Symbol",
                    color_discrete_map=_CLASS_COLORS,
                    title="PIPS vs Master Score",
                )
                mn = min(df_base["Master_Score"].min(), df_base["PIPS_Score"].min())
                mx = max(df_base["Master_Score"].max(), df_base["PIPS_Score"].max())
                fig_pips_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                                      line=dict(color="#555", dash="dash", width=1))
                st.plotly_chart(pdark(fig_pips_sc, height=500), width='stretch')

            st.markdown("---")
            st.markdown("##### Divergence table — high PIPS, low Master Score")
            df_div = df_base.copy()
            df_div["PIPS_minus_Master"] = df_div["PIPS_Score"] - df_div["Master_Score"]
            st.dataframe(
                df_div.sort_values("PIPS_minus_Master", ascending=False).head(30)
                [["Gene_Symbol", "PIPS_Score", "Master_Score",
                  "PIPS_minus_Master", "Anomaly_Class",
                  "Eigenvector_Centrality", p_col]].reset_index(drop=True),
                width='stretch', hide_index=True,
            )

            if "Latent_Entropy" in df_base.columns and "Expression_Entropy" in df_base.columns:
                st.markdown("---")
                st.markdown("##### Shannon Information View")
                df_sh = df_base.copy()
                df_sh["Expression_Specificity"] = 1.0 - df_sh["Expression_Entropy"]
                fig_sh = px.scatter(
                    df_sh, x="Latent_Entropy", y="Expression_Specificity",
                    color="Anomaly_Class", size="PIPS_Score",
                    hover_name="Gene_Symbol",
                    hover_data=["Master_Score", p_col],
                    color_discrete_map=_CLASS_COLORS,
                    title="Shannon Space: Latent Uncertainty vs Expression Specificity",
                )
                st.plotly_chart(pdark(fig_sh, height=500), width='stretch')

                df_sh["Specificity_x_PIPS"] = df_sh["Expression_Specificity"] * df_sh["PIPS_Score"]
                st.markdown("##### Top condition-specific propagators")
                st.dataframe(
                    df_sh.sort_values("Specificity_x_PIPS", ascending=False).head(20)
                    [["Gene_Symbol", "Expression_Specificity", "Latent_Entropy",
                      "PIPS_Score", "Master_Score", "Anomaly_Class", p_col]].reset_index(drop=True),
                    width='stretch', hide_index=True,
                )

    # ── TAB 11: Jackknife Stability ─────────────────────────────────────────────
    with tabs[11]:
        st.markdown("#### 🔁 Jackknife Stability Score")
        st.caption(
            "For each leave-one-out fold, the pipeline is retrained and the top-20% "
            "proteins by reconstruction error are flagged. **Stability = fraction of "
            "folds where a protein was in the top-20%.** Score of 1.0 = flagged in "
            "every fold → most robust finding."
        )

        if "Jackknife_Stability" not in df_base.columns:
            st.info("Jackknife was disabled for this run. Enable it in the Robustness sidebar panel.")
        else:
            jk1, jk2 = st.columns(2)

            with jk1:
                st.markdown("##### Distribution of stability scores")
                fig_jk_hist = px.histogram(
                    df_base, x="Jackknife_Stability",
                    color="Anomaly_Class", nbins=20, barmode="overlay",
                    color_discrete_map=_CLASS_COLORS,
                    title="Stability score distribution",
                )
                st.plotly_chart(pdark(fig_jk_hist, height=400), width='stretch')

            with jk2:
                st.markdown("##### Stability vs Master Score")
                fig_jk_sc = px.scatter(
                    df_base, x="Master_Score", y="Jackknife_Stability",
                    color="Anomaly_Class", hover_name="Gene_Symbol",
                    color_discrete_map=_CLASS_COLORS,
                    title="Master Score vs Jackknife Stability",
                    labels={"Jackknife_Stability": "Stability (0=unstable, 1=robust)"},
                )
                st.plotly_chart(pdark(fig_jk_sc, height=400), width='stretch')

            st.markdown("---")
            st.markdown("##### Top 30 most stable proteins")
            jk_show = ["Gene_Symbol", "Jackknife_Stability", "Master_Score",
                       "PIPS_Score", "Anomaly_Class", p_col]
            st.dataframe(
                df_base.sort_values("Jackknife_Stability", ascending=False).head(30)
                [[c for c in jk_show if c in df_base.columns]].reset_index(drop=True),
                width='stretch', hide_index=True,
            )

            st.markdown("---")
            st.markdown("##### Proteins with high score but low stability ⚠️")
            st.caption("These proteins rank high in the current run but were NOT consistently "
                       "flagged across leave-one-out folds. Interpret with caution.")
            df_unstable = df_base[
                (df_base["Master_Score"] > df_base["Master_Score"].quantile(0.75)) &
                (df_base["Jackknife_Stability"] < 0.5)
            ].sort_values("Master_Score", ascending=False)
            if df_unstable.empty:
                st.success("No high-score / low-stability proteins found.")
            else:
                st.dataframe(
                    df_unstable[[c for c in jk_show if c in df_unstable.columns]].reset_index(drop=True),
                    width='stretch', hide_index=True,
                )

    # ── TAB 12: Bootstrap CI ───────────────────────────────────────────────────
    with tabs[12]:
        st.markdown("#### 📉 Bootstrap Confidence Intervals")
        st.caption(
            "Confidence intervals for the Master Score computed via parametric bootstrap "
            "(Gaussian noise ±5% of score std, 95% CI). Wide intervals indicate proteins "
            "whose ranking is sensitive to small data perturbations."
        )

        ci_cols = ["Master_Score_CI_Low", "Master_Score_CI_High", "Master_Score_Std"]
        if not all(c in df_base.columns for c in ci_cols):
            st.info("Bootstrap CI was disabled (set iterations > 0 in the Robustness panel).")
        else:
            top_ci = df_base.sort_values("Master_Score", ascending=False).head(30).copy()
            top_ci["CI_Width"] = top_ci["Master_Score_CI_High"] - top_ci["Master_Score_CI_Low"]

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=top_ci["Gene_Symbol"],
                y=top_ci["Master_Score"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=top_ci["Master_Score_CI_High"] - top_ci["Master_Score"],
                    arrayminus=top_ci["Master_Score"] - top_ci["Master_Score_CI_Low"],
                    color="#AAAAAA",
                ),
                marker_color="#007ACC",
                name="Master Score ± 95% CI",
            ))
            fig_ci.update_layout(
                **_PLOTLY_DARK,
                xaxis=dict(gridcolor="#333", tickangle=-45),
                yaxis=dict(gridcolor="#333"),
                title="Top 30 proteins — Master Score with 95% Bootstrap CI",
                height=500,
            )
            st.plotly_chart(fig_ci, width='stretch')

            st.markdown("---")
            ci_show = ["Gene_Symbol", "Master_Score",
                       "Master_Score_CI_Low", "Master_Score_CI_High",
                       "Master_Score_Std", "CI_Width", "Anomaly_Class"]
            st.dataframe(
                top_ci[[c for c in ci_show if c in top_ci.columns]].reset_index(drop=True),
                width='stretch', hide_index=True,
            )

    # ── TAB 13: Compare ────────────────────────────────────────────────────────
    with tabs[13]:
        st.markdown("#### 🆚 Cross-Run Comparison")
        all_runs_now = db.list_runs()

        if len(all_runs_now) < 2:
            st.info("Save at least 2 runs to compare.")
        else:
            run_opts = {
                f"#{r['run_id']} — {r['name']} ({r['created_at']})": r["run_id"]
                for r in all_runs_now
            }
            keys_ = list(run_opts.keys())
            pre   = st.session_state.get("compare_ids", [])
            rev   = {v: k for k, v in run_opts.items()}
            def_a = rev.get(pre[0], keys_[0])                        if pre else keys_[0]
            def_b = rev.get(pre[1], keys_[min(1, len(keys_)-1)])     if pre else keys_[min(1, len(keys_)-1)]

            cc1, cc2 = st.columns(2)
            run_a_key = cc1.selectbox("Run A:", keys_, index=keys_.index(def_a))
            run_b_key = cc2.selectbox("Run B:", keys_, index=keys_.index(def_b))
            id_a = run_opts[run_a_key]
            id_b = run_opts[run_b_key]
            data_a = next(r for r in all_runs_now if r["run_id"] == id_a)
            data_b = next(r for r in all_runs_now if r["run_id"] == id_b)

            st.markdown("##### Run summary")
            hdr, va, vb, vd = st.columns([2, 1, 1, 1])
            hdr.caption("Metric"); va.caption(f"Run {id_a}")
            vb.caption(f"Run {id_b}"); vd.caption("Delta")

            def cmp_row(label, key, fmt="{:.0f}"):
                a_ = data_a.get(key, 0) or 0
                b_ = data_b.get(key, 0) or 0
                d_ = a_ - b_ if isinstance(a_, (int, float)) else ""
                cls = "delta-pos" if isinstance(d_, float) and d_ > 0 else "delta-neg"
                hdr.markdown(f"**{label}**")
                va.markdown(str(a_) if not isinstance(a_, float) else fmt.format(a_))
                vb.markdown(str(b_) if not isinstance(b_, float) else fmt.format(b_))
                if isinstance(d_, float):
                    vd.markdown(f'<span class="{cls}">{d_:+.2f}</span>',
                                unsafe_allow_html=True)

            cmp_row("Proteins",      "n_proteins")
            cmp_row("Validated",     "n_validated")
            cmp_row("Discoveries",   "n_discoveries")
            cmp_row("Sig. drivers",  "n_sig_drivers")
            cmp_row("Mean score",    "mean_master_score", "{:.4f}")
            cmp_row("Network edges", "n_network_edges")
            cmp_row("Clusters",      "n_clusters")

            st.markdown("---")
            cdf = db.compare_runs([id_a, id_b])
            if not cdf.empty:
                sa = f"master_score__run{id_a}"
                sb = f"master_score__run{id_b}"
                if sa in cdf.columns and sb in cdf.columns:
                    fig_sc = px.scatter(
                        cdf.dropna(subset=[sa, sb]),
                        x=sa, y=sb, hover_name="gene_symbol",
                        color="score_delta" if "score_delta" in cdf.columns else None,
                        color_continuous_scale="RdBu",
                        title=f"Master score: Run {id_a} vs Run {id_b}",
                        labels={sa: f"Run {id_a} score", sb: f"Run {id_b} score"},
                    )
                    mn = min(cdf[[sa, sb]].min())
                    mx = max(cdf[[sa, sb]].max())
                    fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                                     line=dict(color="#555", dash="dash", width=1))
                    st.plotly_chart(pdark(fig_sc, height=500), width='stretch')

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f"⬆ Higher in Run {id_a}")
                        st.dataframe(
                            cdf.nlargest(15, "score_delta")[
                                ["gene_symbol", sa, sb, "score_delta"]
                            ].reset_index(drop=True),
                            width='stretch', hide_index=True,
                        )
                    with mc2:
                        st.markdown(f"⬇ Higher in Run {id_b}")
                        st.dataframe(
                            cdf.nsmallest(15, "score_delta")[
                                ["gene_symbol", sa, sb, "score_delta"]
                            ].reset_index(drop=True),
                            width='stretch', hide_index=True,
                        )

                    ca_col = f"anomaly_class__run{id_a}"
                    cb_col = f"anomaly_class__run{id_b}"
                    if ca_col in cdf.columns and cb_col in cdf.columns:
                        agree  = cdf.groupby([ca_col, cb_col]).size().reset_index(name="count")
                        fig_ag = px.bar(
                            agree, x=ca_col, y="count", color=cb_col,
                            barmode="group",
                            title=f"Class agreement: Run {id_a} vs Run {id_b}",
                            color_discrete_map=_CLASS_COLORS,
                        )
                        st.plotly_chart(pdark(fig_ag, height=380), width='stretch')

    # ── TAB 14: Chatbot ───────────────────────────────────────────────────────
    with tabs[14]:
        st.markdown("#### 🤖 AI Research Assistant")
        st.caption("Ask questions about your results. The bot reads directly from the database.")

        if not ollama_running:
            st.error("Ollama is offline. Start it with `ollama serve`.")
            st.code("ollama pull llama3\nollama serve", language="bash")
        else:
            all_runs_chat = db.list_runs()
            if not all_runs_chat:
                st.info("No runs in the database yet.")
            else:
                chat_opts    = {f"#{r['run_id']} — {r['name']}": r["run_id"]
                                for r in all_runs_chat}
                chat_keys    = st.multiselect(
                    "Context from runs:", list(chat_opts.keys()),
                    default=list(chat_opts.keys())[:min(2, len(chat_opts))],
                    key="chat_run_sel",
                )
                chat_run_ids = [chat_opts[k] for k in chat_keys]

                chat_col, ctrl_col = st.columns([3, 1])

                with ctrl_col:
                    st.markdown("**Settings**")
                    chat_temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                                          key="chat_temp")
                    top_n_ctx = st.slider("Proteins in context", 10, 100, 30, 5,
                                          key="chat_topn")
                    if st.button("🗑 Clear chat", width='stretch',
                                  key="clear_chat"):
                        st.session_state.pop("chat_history", None)
                        for k in list(st.session_state.keys()):
                            if k.startswith("_chatbot_"):
                                st.session_state.pop(k, None)
                        st.rerun()

                    st.markdown("**Suggestions**")
                    base_qs = [
                        "What are the top 10 proteins by master score?",
                        "Which proteins have the highest PIPS but low master score?",
                        "Which proteins have the highest jackknife stability?",
                        "Explain what 'Biological_Discovery' classification means.",
                        "Which cluster has the most significant pathway enrichment?",
                        "What is the mathematical formula for the master score?",
                        "Show me proteins with the lowest BH-adjusted p-values.",
                    ]
                    if len(chat_run_ids) >= 2:
                        base_qs += [
                            f"Compare Run {chat_run_ids[0]} and Run {chat_run_ids[1]}.",
                            "Which proteins are validated in both runs?",
                        ]
                    for q in base_qs[:6]:
                        label = q[:55] + ("…" if len(q) > 55 else "")
                        if st.button(label, key=f"sq_{hash(q)}",
                                      width='stretch'):
                            st.session_state["chat_prefill"] = q
                            st.rerun()

                with chat_col:
                    bot_key = f"_chatbot_{tuple(chat_run_ids)}_{selected_model}"
                    if bot_key not in st.session_state:
                        st.session_state[bot_key] = ProteomicsChatbot(
                            db, ollama, chat_run_ids, model=selected_model,
                            temperature=chat_temp, top_n_proteins=top_n_ctx,
                        )
                        st.session_state["chat_history"] = []

                    bot: ProteomicsChatbot = st.session_state[bot_key]
                    bot.temperature    = chat_temp
                    bot.top_n_proteins = top_n_ctx
                    bot._context_built_for = ()

                    for turn in st.session_state.get("chat_history", []):
                        label = "You" if turn["role"] == "user" else "Assistant"
                        css   = "chat-user" if turn["role"] == "user" else "chat-assistant"
                        st.markdown(
                            f'<div class="{css}"><div class="chat-label">{label}</div>'
                            f'{turn["content"]}</div>', unsafe_allow_html=True,
                        )

                    prefill  = st.session_state.pop("chat_prefill", "")
                    user_msg = st.chat_input("Ask about your results …", key="chat_input")
                    if not user_msg and prefill:
                        user_msg = prefill

                    if user_msg:
                        safe_user_msg = html.escape(user_msg)
                        st.markdown(
                            f'<div class="chat-user"><div class="chat-label">You</div>'
                            f'{safe_user_msg}</div>', unsafe_allow_html=True,
                        )
                        reply_ph   = st.empty()
                        full_reply = ""
                        try:
                            with st.spinner("Thinking …"):
                                for chunk in bot.stream(user_msg):
                                    full_reply += chunk
                                    reply_ph.markdown(
                                        f'<div class="chat-assistant">'
                                        f'<div class="chat-label">Assistant</div>'
                                        f'{html.escape(full_reply)}▌</div>',
                                        unsafe_allow_html=True,
                                    )
                            reply_ph.markdown(
                                f'<div class="chat-assistant">'
                                f'<div class="chat-label">Assistant</div>'
                                f'{html.escape(full_reply)}</div>',
                                unsafe_allow_html=True,
                            )
                        except OllamaError as e:
                            reply_ph.error(str(e))
                            full_reply = f"[Error: {e}]"

                        hist = st.session_state.get("chat_history", [])
                        hist.append({"role": "user",      "content": safe_user_msg})
                        hist.append({"role": "assistant",  "content": html.escape(full_reply)})
                        st.session_state["chat_history"] = hist
                        bot.history = [{"role": t["role"], "content": t["content"]} for t in hist]
                        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  LOADED RUN VIEW
# ──────────────────────────────────────────────────────────────────────────────

elif "loaded_run_id" in st.session_state:
    rid = st.session_state["loaded_run_id"]
    try:
        run = db.load_run(rid)
    except KeyError:
        st.error(f"Run #{rid} not found.")
        st.stop()

    m = run["meta"]
    st.markdown(f"## 📂 Run #{rid}: {m['name']}")
    st.caption(f"Dataset: {m['dataset_name']} | {m['created_at']}")

    mc1, mc2, mc3, mc4 = st.columns(4)
    render_metric(mc1, "PROTEINS",    str(m["n_proteins"]),    "Total")
    render_metric(mc2, "VALIDATED",   str(m["n_validated"]),   "High confidence")
    render_metric(mc3, "DISCOVERIES", str(m["n_discoveries"]), "New signals")
    render_metric(mc4, "SCORE",       f"{m['mean_master_score']:.3f}", "Mean master")

    lt, rt = st.tabs(["📊 Proteins", "🧬 Enrichment"])
    with lt:
        st.dataframe(run["df_proteins"], width='stretch', height=480)
        if not run["df_proteins"].empty and "master_score" in run["df_proteins"].columns:
            fig_h = px.histogram(
                run["df_proteins"], x="master_score",
                color="anomaly_class", nbins=50, barmode="overlay",
                color_discrete_map={k.lower(): v for k, v in _CLASS_COLORS.items()},
                title="Master score distribution",
            )
            st.plotly_chart(pdark(fig_h), width='stretch')
    with rt:
        if run["df_enrichment"].empty:
            st.info("No enrichment data for this run.")
        else:
            st.dataframe(run["df_enrichment"], width='stretch', height=400)

    st.markdown("---")
    new_notes = st.text_area("Notes:", value=m.get("notes", ""), height=80)
    if st.button("💾 Save notes"):
        db.update_notes(rid, new_notes)
        st.success("Saved.")
    cb1, cb2 = st.columns(2)
    if cb1.button("← Back"):
        st.session_state.pop("loaded_run_id", None)
        st.rerun()
    if cb2.button("🗑 Delete run", type="secondary"):
        db.delete_run(rid)
        st.session_state.pop("loaded_run_id", None)
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  WELCOME SCREEN
# ──────────────────────────────────────────────────────────────────────────────

elif (not uploaded_file
      and "loaded_run_id" not in st.session_state
      and "pipeline_state" not in st.session_state):
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <h1 style='font-size:3.5rem;line-height:1.1;color:#CCCCCC;'>
            <br><span style='color:#007ACC;'>Proteomics Engine</span>
        </h1>
        <p style='font-size:1.1rem;color:#858585;margin-top:20px;
                  border-left:3px solid #007ACC;padding-left:15px;'>
          β-VAE ensemble · Ensemble SHAP · Adaptive p-values · Adaptive clustering ·
          GLasso / Partial-correlation networks · PHATE / UMAP / PCA manifolds ·
          Pathway enrichment ·
          <strong style='color:#ce93d8;'>PIPS — Protein Influence Propagation</strong> ·
          <strong style='color:#ffa726;'>Jackknife Stability</strong> ·
          <strong style='color:#64b5f6;'>Bootstrap CI</strong> ·
          <strong style='color:#DCDCAA;'>SQLite database</strong> ·
          <strong style='color:#4CAF50;'>Ollama AI chatbot</strong> ·
          <strong style='color:#007ACC;'>Cross-run comparison</strong>
        </p>""", unsafe_allow_html=True)
    with c2:
        if LOTTIE_OK and lottie_dna:
            st_lottie(lottie_dna, height=400, key="dna_anim")
