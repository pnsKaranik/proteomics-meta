from __future__ import annotations

import numpy as np
import pandas as pd

from proteomics_meta.db import _array_to_blob, _blob_to_array, _blob_to_df, _df_to_blob
from proteomics_meta.networks import (
    build_network,
    build_spearman_network,
    calculate_network_centrality,
)


def _correlated_matrix(n_proteins=30, n_samples=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_proteins, n_samples))
    # make the first three proteins strongly correlated
    base[1] = base[0] + rng.normal(scale=0.01, size=n_samples)
    base[2] = base[0] + rng.normal(scale=0.01, size=n_samples)
    return base


def test_spearman_network_returns_graph_and_matrix():
    X = _correlated_matrix()
    proteins = [f"P{i}" for i in range(X.shape[0])]
    graph, corr = build_spearman_network(X, proteins, threshold=0.8)
    assert graph.number_of_nodes() >= 2
    assert corr.shape == (X.shape[0], X.shape[0])
    # the engineered trio should be connected
    assert graph.has_edge("P0", "P1") or graph.has_edge("P0", "P2")


def test_build_network_routes_low_sample_without_error():
    X = _correlated_matrix(n_proteins=20, n_samples=8)
    proteins = [f"P{i}" for i in range(X.shape[0])]
    graph, _ = build_network(X, proteins, mode="low_sample", threshold=0.1)
    assert graph.number_of_nodes() >= 0


def test_centrality_empty_graph_returns_frame():
    import networkx as nx

    df = calculate_network_centrality(nx.Graph(), ["P0", "P1"])
    assert isinstance(df, pd.DataFrame)


def test_array_blob_roundtrip():
    arr = np.random.default_rng(0).random((10, 4)).astype(np.float32)
    restored = _blob_to_array(_array_to_blob(arr))
    assert np.allclose(arr, restored)


def test_dataframe_blob_roundtrip():
    df = pd.DataFrame({"gene": ["TP53", "EGFR"], "score": [0.9, 0.3]})
    restored = _blob_to_df(_df_to_blob(df))
    pd.testing.assert_frame_equal(df, restored)
