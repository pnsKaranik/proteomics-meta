from __future__ import annotations

import numpy as np
import pandas as pd

from proteomics_meta.stats import (
    calculate_bootstrap_pvalues,
    calculate_consensus_score,
    calculate_robust_stats,
    compute_bootstrap_ci,
)


def test_robust_stats_shapes_and_range():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 200), [12.0, 15.0]])  # two clear outliers
    z, raw_p, bh_p = calculate_robust_stats(values)
    assert z.shape == raw_p.shape == bh_p.shape == values.shape
    assert np.all((bh_p >= 0) & (bh_p <= 1))
    # the injected high outliers should be the most significant (smallest p)
    assert raw_p[-1] < raw_p[:200].mean()


def test_robust_stats_constant_input_does_not_crash():
    z, raw_p, bh_p = calculate_robust_stats(np.ones(50))
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(bh_p))


def test_bootstrap_pvalues_bounded():
    rng = np.random.default_rng(1)
    p = calculate_bootstrap_pvalues(rng.normal(size=100), n_bootstrap=200)
    p = np.asarray(p)
    assert p.shape == (100,)
    assert np.all((p >= 0) & (p <= 1))


def test_bootstrap_ci_keys_and_ordering():
    rng = np.random.default_rng(2)
    out = compute_bootstrap_ci(rng.random(80), n_bootstrap=200, rng=rng)
    assert isinstance(out, dict)
    lower = np.asarray(out["ci_low"])
    upper = np.asarray(out["ci_high"])
    assert np.all(lower <= upper + 1e-9)


def test_consensus_score_monotonic_in_shap():
    df = pd.DataFrame(
        {
            "SHAP_Importance": [0.0, 0.5, 1.0],
            "Eigenvector_Centrality": [0.5, 0.5, 0.5],
            "P_Value_BH": [0.5, 0.5, 0.5],
        }
    )
    score = calculate_consensus_score(df)
    assert score.shape == (3,)
    assert score[0] <= score[1] <= score[2]
