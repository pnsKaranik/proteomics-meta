# proteomics-meta

[![ci](https://github.com/pnsKaranik/proteomics-meta/actions/workflows/ci.yml/badge.svg)](https://github.com/pnsKaranik/proteomics-meta/actions/workflows/ci.yml)

A sample-aware meta-analysis engine for quantitative proteomics. Given a
protein × sample abundance matrix, it ranks proteins by a consensus "master
score" that combines a VAE ensemble, co-expression network topology, feature
attribution and robust statistics — then serves the results through a Streamlit
dashboard with an optional local-LLM chatbot over the findings.

The pipeline adapts to cohort size: with a handful of samples it falls back to
shrinkage estimators and bootstrap p-values; with large cohorts it uses
graphical-lasso networks, Louvain communities and BH-FDR.

## Architecture

The codebase is a `src/` package with a strict dependency direction. The pure
statistical, network and I/O layers carry **no TensorFlow import**, so they load
fast and are unit-tested in CI without the heavy stack; the TF-dependent VAE core
sits on top of them.

```
proteomics_meta/
├── capabilities.py   # single place for optional-dependency detection + logging
├── config.py         # PipelineConfig, sample-aware mode routing
├── stats.py          # robust stats, bootstrap, consensus, differential expression
├── networks.py       # graphical-lasso / Spearman / partial-corr networks, clustering
├── entropy.py        # latent- and expression-space entropy features
├── external.py       # UniProt gene-name resolution, STRING interactions
├── engine.py         # β-VAE ensemble, SHAP/PIPS, reporting, orchestration (TensorFlow)
├── db.py             # SQLite result store (compressed numpy/parquet blobs)
├── chatbot.py        # Ollama client + gene-aware question answering
├── app.py            # Streamlit dashboard
├── cli.py            # batch runner (csv / tsv / parquet)
└── app_launcher.py   # `proteomics-meta-app` entry point
```

Dependency flow: `capabilities → {config, stats, networks, entropy, external} → engine → {app, cli}`. No cycles.

## Install

```bash
pip install -e .              # core engine + CLI
pip install -e ".[app]"       # + Streamlit dashboard
pip install -e ".[advanced]"  # + Louvain, SHAP, PHATE, UMAP, GSEApy
pip install -e ".[dev]"       # + pytest, ruff
```

TensorFlow 2.13 is pinned for the VAE. The advanced visualisation/enrichment
libraries are optional and degrade gracefully when absent.

## Use

```bash
# batch run on a matrix (proteins as rows, samples as columns, + a Protein.Group column)
proteomics-meta --file data/abundances.parquet --outdir results --iterations 3

# synthetic demo run (no input file)
proteomics-meta

# interactive dashboard
proteomics-meta-app
```

Or via Docker (bundles the app and a local Ollama model for the chatbot tab):

```bash
docker compose up --build   # dashboard on http://localhost:8501
```

## Pipeline

For each abundance matrix the engine: normalises and imputes; trains an ensemble
of β-VAEs and extracts a latent embedding; builds a co-expression network sized
to the sample count; clusters proteins; attributes importance with SHAP and
posterior inclusion probabilities; computes robust MAD-Z statistics with BH-FDR
(or bootstrap p-values in low-sample mode); and fuses these into a weighted
master score. Jackknife stability and bootstrap confidence intervals quantify how
reproducible each ranking is. Results, networks and figures are written to disk,
persisted in SQLite, and rendered in the dashboard.

## Sample-aware modes

| Mode | Samples | Network | Clustering | p-values |
|---|---|---|---|---|
| ultra_sparse | < 5 | graphical lasso | hierarchical (k=2) | bootstrap |
| low_sample | 5–14 | graphical lasso | GMM (k≤4) | bootstrap |
| moderate | 15–29 | graphical lasso | GMM (k≤10) | BH-FDR |
| full | ≥ 30 | partial correlation | Louvain / GMM | BH-FDR |

## Development

```bash
ruff check src tests
pytest
```

Tests cover the pure layers (mode routing, robust statistics, consensus scoring,
network construction and centrality, and the SQLite blob round-trips) and run
without TensorFlow, which keeps CI fast. Continuous integration runs ruff and
pytest on Python 3.10 and 3.11.
