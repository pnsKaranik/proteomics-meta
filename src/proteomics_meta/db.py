"""
database.py — SQLite persistence layer for Meta Analysis Engine.

Schema:
  runs          — one row per pipeline execution (metadata + config)
  proteins      — per-protein results (linked to run_id)
  enrichment    — pathway enrichment rows (linked to run_id)
  network_edges — top network edges (linked to run_id)
  stability     — VAE stability scores (linked to run_id)

All heavy numpy arrays (latent space, pcorr matrix) are stored as
compressed blobs so the DB stays portable.

Public API (used by app.py):
  db = ResultsDB("meta_analysis.db")
  run_id = db.save_run(state, df_final, config)
  runs   = db.list_runs()               → list[dict]
  run    = db.load_run(run_id)          → dict with df_proteins, df_enrichment …
  df     = db.compare_runs([id1, id2])  → wide DataFrame for comparison view
  db.delete_run(run_id)
  db.close()
"""

import gzip
import io
import json
import logging
import sqlite3
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

logger = logging.getLogger("MetaAnalysis.DB")


# ──────────────────────────────────────────────────────────────────────────────
#  BLOB HELPERS  (compress numpy arrays → bytes, decompress back)
# ──────────────────────────────────────────────────────────────────────────────

def _array_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return gzip.compress(buf.getvalue(), compresslevel=6)


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(gzip.decompress(blob)))


def _df_to_blob(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    return gzip.compress(buf.getvalue(), compresslevel=6)


def _blob_to_df(blob: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(gzip.decompress(blob)))


# ──────────────────────────────────────────────────────────────────────────────
#  RESULTSDB
# ──────────────────────────────────────────────────────────────────────────────

class ResultsDB:
    """SQLite-backed store for pipeline runs."""

    DDL = """
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS runs (
        run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    NOT NULL DEFAULT '',
        created_at      TEXT    NOT NULL,
        dataset_name    TEXT    NOT NULL DEFAULT '',
        n_proteins      INTEGER NOT NULL DEFAULT 0,
        n_samples       INTEGER NOT NULL DEFAULT 0,
        n_validated     INTEGER NOT NULL DEFAULT 0,
        n_discoveries   INTEGER NOT NULL DEFAULT 0,
        n_sig_drivers   INTEGER NOT NULL DEFAULT 0,
        mean_master_score REAL  NOT NULL DEFAULT 0.0,
        n_clusters      INTEGER NOT NULL DEFAULT 0,
        n_network_edges INTEGER NOT NULL DEFAULT 0,
        -- config snapshot (JSON)
        config_json     TEXT    NOT NULL DEFAULT '{}',
        -- methods used
        shap_method     TEXT    NOT NULL DEFAULT '',
        cluster_method  TEXT    NOT NULL DEFAULT '',
        fdr_method      TEXT    NOT NULL DEFAULT 'BH',
        -- heavy blobs (optional — skipped if too large)
        latent_blob     BLOB,
        pcorr_blob      BLOB,
        notes           TEXT    NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS proteins (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                  INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        protein_id              TEXT    NOT NULL,
        gene_symbol             TEXT    NOT NULL DEFAULT '',
        anomaly_class           TEXT    NOT NULL DEFAULT '',
        master_score            REAL    NOT NULL DEFAULT 0.0,
        shap_importance         REAL    NOT NULL DEFAULT 0.0,
        eigenvector_centrality  REAL    NOT NULL DEFAULT 0.0,
        betweenness_centrality  REAL    NOT NULL DEFAULT 0.0,
        latent_connectivity     INTEGER NOT NULL DEFAULT 0,
        reconstruction_error    REAL    NOT NULL DEFAULT 0.0,
        reconstruction_cv       REAL    NOT NULL DEFAULT 0.0,
        z_score                 REAL    NOT NULL DEFAULT 0.0,
        p_value                 REAL    NOT NULL DEFAULT 1.0,
        p_value_bh              REAL    NOT NULL DEFAULT 1.0,
        cluster_id              INTEGER NOT NULL DEFAULT -1,
        pseudotime              REAL    NOT NULL DEFAULT 0.0,
        trajectory_corr         REAL    NOT NULL DEFAULT 0.0,
        trajectory_pval_bh      REAL    NOT NULL DEFAULT 1.0,
        ml_confidence           REAL    NOT NULL DEFAULT 0.0
    );
    CREATE INDEX IF NOT EXISTS idx_proteins_run   ON proteins(run_id);
    CREATE INDEX IF NOT EXISTS idx_proteins_gene  ON proteins(gene_symbol);
    CREATE INDEX IF NOT EXISTS idx_proteins_class ON proteins(anomaly_class);

    CREATE TABLE IF NOT EXISTS enrichment (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        cluster_id      INTEGER NOT NULL DEFAULT -1,
        term            TEXT    NOT NULL DEFAULT '',
        gene_set        TEXT    NOT NULL DEFAULT '',
        p_value         REAL    NOT NULL DEFAULT 1.0,
        adj_p_value     REAL    NOT NULL DEFAULT 1.0,
        odds_ratio      REAL    NOT NULL DEFAULT 0.0,
        combined_score  REAL    NOT NULL DEFAULT 0.0,
        genes           TEXT    NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_enrichment_run ON enrichment(run_id);

    CREATE TABLE IF NOT EXISTS network_edges (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        source      TEXT    NOT NULL,
        target      TEXT    NOT NULL,
        source_gene TEXT    NOT NULL DEFAULT '',
        target_gene TEXT    NOT NULL DEFAULT '',
        weight      REAL    NOT NULL DEFAULT 0.0
    );
    CREATE INDEX IF NOT EXISTS idx_edges_run ON network_edges(run_id);

    CREATE TABLE IF NOT EXISTS stability (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        protein_id          TEXT    NOT NULL,
        recon_error_cv      REAL    NOT NULL DEFAULT 0.0,
        recon_error_mean    REAL    NOT NULL DEFAULT 0.0
    );
    CREATE INDEX IF NOT EXISTS idx_stability_run ON stability(run_id);
    """

    def __init__(self, db_path: str = "meta_analysis.db"):
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("Database opened: %s", db_path)

    def _init_schema(self):
        for stmt in self.DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self):
        self._conn.close()

    # ── SAVE RUN ──────────────────────────────────────────────────────────────

    def save_run(
        self,
        state: dict,
        df_final: pd.DataFrame,
        config,                          # PipelineConfig or dict
        name: str = "",
        dataset_name: str = "",
        store_blobs: bool = True,
        max_edge_rows: int = 10_000,
    ) -> int:
        """
        Persist a complete pipeline run.  Returns the new run_id.
        """
        # ── Config JSON ───────────────────────────────────────────────────────
        if hasattr(config, "__dict__"):
            cfg_dict = {k: v for k, v in config.__dict__.items()
                        if isinstance(v, (int, float, str, bool, list, type(None)))}
        else:
            cfg_dict = dict(config) if config else {}

        p_col     = "P_Value_BH" if "P_Value_BH" in df_final.columns else "P_Value"
        validated = (df_final.get("Anomaly_Class", pd.Series()) == "Validated_Signal").sum()
        discover  = (df_final.get("Anomaly_Class", pd.Series()) == "Biological_Discovery").sum()
        sig_drv   = (df_final[p_col] < 0.05).sum() if p_col in df_final.columns else 0

        G = state.get("G_initial")
        n_edges   = G.number_of_edges() if G else 0
        n_clust   = int(df_final["Cluster_ID"].nunique()) if "Cluster_ID" in df_final.columns else 0

        # ── Blobs ─────────────────────────────────────────────────────────────
        latent_blob = pcorr_blob = None
        if store_blobs:
            try:
                if "latent_space" in state:
                    latent_blob = _array_to_blob(state["latent_space"])
                if "pcorr_matrix" in state:
                    pcorr_blob = _array_to_blob(state["pcorr_matrix"])
            except Exception as exc:
                logger.warning("Could not serialise blobs: %s", exc)

        with self._tx() as cx:
            cur = cx.execute("""
                INSERT INTO runs (
                    name, created_at, dataset_name,
                    n_proteins, n_samples, n_validated, n_discoveries,
                    n_sig_drivers, mean_master_score,
                    n_clusters, n_network_edges,
                    config_json, shap_method, cluster_method, fdr_method,
                    latent_blob, pcorr_blob
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                name or f"Run {time.strftime('%Y-%m-%d %H:%M')}",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                dataset_name,
                len(df_final),
                state.get("df_log", pd.DataFrame()).shape[1],
                int(validated), int(discover), int(sig_drv),
                float(df_final["Master_Score"].mean()) if "Master_Score" in df_final.columns else 0.0,
                n_clust, n_edges,
                json.dumps(cfg_dict),
                cfg_dict.get("shap_method", "unknown"),
                cfg_dict.get("cluster_method", "unknown"),
                "Benjamini-Hochberg",
                latent_blob, pcorr_blob,
            ))
            run_id = cur.lastrowid

            # ── Proteins ──────────────────────────────────────────────────────
            rows = []
            for _, r in df_final.iterrows():
                rows.append((
                    run_id,
                    str(r.get("Protein_ID", "")),
                    str(r.get("Gene_Symbol", "")),
                    str(r.get("Anomaly_Class", "")),
                    float(r.get("Master_Score",            0.0)),
                    float(r.get("SHAP_Importance",         0.0)),
                    float(r.get("Eigenvector_Centrality",  0.0)),
                    float(r.get("Betweenness_Centrality",  0.0)),
                    int(  r.get("Latent_Connectivity",     0)),
                    float(r.get("Reconstruction_Error",    0.0)),
                    float(r.get("Reconstruction_Error_CV", 0.0)),
                    float(r.get("Z_Score",                 0.0)),
                    float(r.get("P_Value",                 1.0)),
                    float(r.get("P_Value_BH",              1.0)),
                    int(  r.get("Cluster_ID",             -1)),
                    float(r.get("Pseudotime",              0.0)),
                    float(r.get("Trajectory_Correlation",  0.0)),
                    float(r.get("Trajectory_PVal_BH",      1.0)),
                    float(r.get("ML_Confidence",           0.0)),
                ))
            cx.executemany("""
                INSERT INTO proteins (
                    run_id, protein_id, gene_symbol, anomaly_class,
                    master_score, shap_importance,
                    eigenvector_centrality, betweenness_centrality,
                    latent_connectivity, reconstruction_error, reconstruction_cv,
                    z_score, p_value, p_value_bh,
                    cluster_id, pseudotime,
                    trajectory_corr, trajectory_pval_bh, ml_confidence
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)

            # ── Enrichment ────────────────────────────────────────────────────
            enr_df = state.get("enrichment_df")
            if enr_df is not None and not enr_df.empty:
                enr_rows = []
                for _, r in enr_df.iterrows():
                    enr_rows.append((
                        run_id,
                        int(r.get("Cluster", -1)),
                        str(r.get("Term", "")),
                        str(r.get("Gene_set", "")),
                        float(r.get("P-value",          1.0)),
                        float(r.get("Adjusted P-value", 1.0)),
                        float(r.get("Odds Ratio",       0.0)),
                        float(r.get("Combined Score",   0.0)),
                        str(r.get("Genes", "")),
                    ))
                cx.executemany("""
                    INSERT INTO enrichment (
                        run_id, cluster_id, term, gene_set,
                        p_value, adj_p_value, odds_ratio, combined_score, genes
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                """, enr_rows)

            # ── Network edges (top N by weight) ───────────────────────────────
            if G and G.number_of_edges() > 0:
                gene_map = state.get("gene_map", {})
                edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True)
                edge_rows = []
                for u, v, d in edges[:max_edge_rows]:
                    if d.get("weight", 0) > 0.1:
                        edge_rows.append((
                            run_id, u, v,
                            gene_map.get(u, u), gene_map.get(v, v),
                            float(d.get("weight", 0)),
                        ))
                if edge_rows:
                    cx.executemany("""
                        INSERT INTO network_edges (run_id, source, target, source_gene, target_gene, weight)
                        VALUES (?,?,?,?,?,?)
                    """, edge_rows)

            # ── Stability ─────────────────────────────────────────────────────
            stab_df = state.get("stability_scores")
            if stab_df is not None and not stab_df.empty:
                stab_rows = [
                    (run_id, str(r["Protein_ID"]),
                     float(r.get("Recon_Error_CV", 0.0)),
                     float(r.get("Recon_Error_Mean", 0.0)))
                    for _, r in stab_df.iterrows()
                ]
                cx.executemany("""
                    INSERT INTO stability (run_id, protein_id, recon_error_cv, recon_error_mean)
                    VALUES (?,?,?,?)
                """, stab_rows)

        logger.info("Run saved: id=%d, name=%s, proteins=%d", run_id, name, len(df_final))
        return run_id

    # ── LIST RUNS ─────────────────────────────────────────────────────────────

    def list_runs(self) -> list[dict]:
        """Return summary of all runs, newest first."""
        cur = self._conn.execute("""
            SELECT run_id, name, created_at, dataset_name,
                   n_proteins, n_samples, n_validated, n_discoveries,
                   n_sig_drivers, mean_master_score,
                   n_clusters, n_network_edges,
                   shap_method, cluster_method, fdr_method, notes
            FROM runs ORDER BY run_id DESC
        """)
        return [dict(row) for row in cur.fetchall()]

    # ── LOAD RUN ──────────────────────────────────────────────────────────────

    def load_run(self, run_id: int, load_blobs: bool = False) -> dict:
        """
        Load a full run from the DB.
        Returns dict with keys: meta, df_proteins, df_enrichment, df_edges, df_stability,
        and optionally latent_space / pcorr_matrix arrays.
        """
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"run_id {run_id} not found")

        meta = dict(row)
        try:
            meta["config"] = json.loads(meta.get("config_json", "{}"))
        except json.JSONDecodeError:
            meta["config"] = {}

        # Proteins
        df_proteins = pd.read_sql_query(
            "SELECT * FROM proteins WHERE run_id=? ORDER BY master_score DESC",
            self._conn, params=(run_id,)
        )

        # Enrichment
        df_enrichment = pd.read_sql_query(
            "SELECT * FROM enrichment WHERE run_id=? ORDER BY adj_p_value ASC",
            self._conn, params=(run_id,)
        )

        # Edges
        df_edges = pd.read_sql_query(
            "SELECT * FROM network_edges WHERE run_id=? ORDER BY weight DESC",
            self._conn, params=(run_id,)
        )

        # Stability
        df_stability = pd.read_sql_query(
            "SELECT * FROM stability WHERE run_id=? ORDER BY recon_error_cv DESC",
            self._conn, params=(run_id,)
        )

        result = {
            "meta":         meta,
            "df_proteins":  df_proteins,
            "df_enrichment":df_enrichment,
            "df_edges":     df_edges,
            "df_stability": df_stability,
        }

        if load_blobs:
            if meta.get("latent_blob"):
                result["latent_space"] = _blob_to_array(meta["latent_blob"])
            if meta.get("pcorr_blob"):
                result["pcorr_matrix"] = _blob_to_array(meta["pcorr_blob"])

        return result

    # ── COMPARE RUNS ──────────────────────────────────────────────────────────

    def compare_runs(self, run_ids: list[int]) -> pd.DataFrame:
        """
        Return a wide DataFrame with one row per gene_symbol that appears in
        ANY of the selected runs, and columns for each metric × run.

        Columns look like: gene_symbol | master_score_run1 | master_score_run2 | …
        """
        if not run_ids:
            return pd.DataFrame()

        metrics = [
            "master_score", "shap_importance",
            "eigenvector_centrality", "betweenness_centrality",
            "reconstruction_error", "p_value_bh",
            "anomaly_class", "cluster_id",
        ]

        frames = []
        for rid in run_ids:
            df = pd.read_sql_query(
                f"SELECT gene_symbol, {', '.join(metrics)} "
                f"FROM proteins WHERE run_id=?",
                self._conn, params=(rid,)
            )
            df = df.rename(columns={m: f"{m}__run{rid}" for m in metrics})
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="gene_symbol", how="outer")

        # Compute rank shift: difference in master_score between first two runs
        score_cols = [c for c in merged.columns if c.startswith("master_score__")]
        if len(score_cols) >= 2:
            merged["score_delta"] = (
                merged[score_cols[0]].fillna(0) - merged[score_cols[1]].fillna(0)
            )

        return merged.sort_values(score_cols[0] if score_cols else "gene_symbol",
                                  ascending=False, na_position="last")

    # ── GET CONTEXT FOR CHATBOT ───────────────────────────────────────────────

    def get_chat_context(
        self,
        run_ids: list[int],
        top_n_proteins: int = 30,
        top_n_pathways: int = 15,
    ) -> str:
        """
        Build a concise text summary of one or more runs for the Ollama chatbot.
        Returns a multi-section string injected as system context.
        """
        sections = []

        for rid in run_ids:
            try:
                run = self.load_run(rid)
            except KeyError:
                continue

            m   = run["meta"]
            cfg = m.get("config", {})

            # ── Run header
            sections.append(f"""
=== RUN {rid}: {m['name']} ===
Dataset: {m['dataset_name']} | Analysed: {m['created_at']}
Proteins: {m['n_proteins']} total | Samples: {m['n_samples']}
Results: {m['n_validated']} validated, {m['n_discoveries']} discoveries, {m['n_sig_drivers']} significant drivers (BH p<0.05)
Mean master score: {m['mean_master_score']:.4f}
Network: {m['n_network_edges']} edges | Clusters: {m['n_clusters']}
Methods: SHAP={m['shap_method']}, Clustering={m['cluster_method']}, FDR={m['fdr_method']}
Config: latent_dim={cfg.get('latent_dim','?')}, iterations={cfg.get('iterations','?')}, epochs={cfg.get('epochs','?')}
""".strip())

            # ── Top proteins
            df_p = run["df_proteins"].head(top_n_proteins)
            if not df_p.empty:
                protein_lines = []
                for _, r in df_p.iterrows():
                    protein_lines.append(
                        f"  {r['gene_symbol']:12s}  score={r['master_score']:.3f}  "
                        f"SHAP={r['shap_importance']:.3f}  "
                        f"BH-p={r['p_value_bh']:.2e}  "
                        f"class={r['anomaly_class']}  cluster={r['cluster_id']}"
                    )
                sections.append("Top proteins (by master score):\n" + "\n".join(protein_lines))

            # ── Top pathways
            df_e = run["df_enrichment"].head(top_n_pathways)
            if not df_e.empty:
                enr_lines = []
                for _, r in df_e.iterrows():
                    enr_lines.append(
                        f"  Cluster {r['cluster_id']:3}  adj-p={r['adj_p_value']:.2e}  "
                        f"OR={r['odds_ratio']:.2f}  {r['term']}"
                    )
                sections.append("Top enriched pathways:\n" + "\n".join(enr_lines))

        if len(run_ids) >= 2:
            cdf = self.compare_runs(run_ids[:2])
            if "score_delta" in cdf.columns:
                top_up   = cdf.nlargest(10,  "score_delta")[["gene_symbol", "score_delta"]]
                top_down = cdf.nsmallest(10, "score_delta")[["gene_symbol", "score_delta"]]
                up_str   = ", ".join(f"{r['gene_symbol']}(+{r['score_delta']:.3f})" for _, r in top_up.iterrows())
                down_str = ", ".join(f"{r['gene_symbol']}({r['score_delta']:.3f})"  for _, r in top_down.iterrows())
                sections.append(
                    f"Comparison (Run {run_ids[0]} vs Run {run_ids[1]}):\n"
                    f"  Higher in Run {run_ids[0]}: {up_str}\n"
                    f"  Higher in Run {run_ids[1]}: {down_str}"
                )

        return "\n\n".join(sections)

    # ── UPDATE NOTES ──────────────────────────────────────────────────────────

    def update_notes(self, run_id: int, notes: str):
        with self._tx() as cx:
            cx.execute("UPDATE runs SET notes=? WHERE run_id=?", (notes, run_id))

    # ── DELETE RUN ────────────────────────────────────────────────────────────

    def delete_run(self, run_id: int):
        with self._tx() as cx:
            cx.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        logger.info("Deleted run_id=%d", run_id)

    # ── STATS QUERY HELPERS (used by chatbot) ─────────────────────────────────

    def query_proteins(
        self,
        run_id: int,
        gene: str = None,
        anomaly_class: str = None,
        cluster_id: int = None,
        min_score: float = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Flexible protein query — used by chatbot to answer specific questions."""
        clauses = ["run_id = ?"]
        params  = [run_id]
        if gene:
            clauses.append("gene_symbol LIKE ?")
            params.append(f"%{gene}%")
        if anomaly_class:
            clauses.append("anomaly_class = ?")
            params.append(anomaly_class)
        if cluster_id is not None:
            clauses.append("cluster_id = ?")
            params.append(cluster_id)
        if min_score is not None:
            clauses.append("master_score >= ?")
            params.append(min_score)

        sql = (f"SELECT * FROM proteins WHERE {' AND '.join(clauses)} "
               f"ORDER BY master_score DESC LIMIT {int(limit)}")
        return pd.read_sql_query(sql, self._conn, params=params)

    def run_summary_stats(self, run_id: int) -> dict:
        """Aggregate stats per class for a given run — used by chatbot."""
        rows = self._conn.execute("""
            SELECT anomaly_class,
                   COUNT(*)          AS n,
                   AVG(master_score) AS avg_score,
                   AVG(shap_importance) AS avg_shap,
                   AVG(p_value_bh)   AS avg_bh_p
            FROM proteins WHERE run_id=?
            GROUP BY anomaly_class
        """, (run_id,)).fetchall()
        return {r["anomaly_class"]: dict(r) for r in rows}
