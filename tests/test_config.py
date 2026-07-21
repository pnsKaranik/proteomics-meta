from __future__ import annotations

import pytest

from proteomics_meta.config import MODE_DESCRIPTIONS, PipelineConfig, get_pipeline_mode


@pytest.mark.parametrize(
    "n_samples,expected",
    [(0, "ultra_sparse"), (4, "ultra_sparse"), (5, "low_sample"), (14, "low_sample"),
     (15, "moderate"), (29, "moderate"), (30, "full"), (500, "full")],
)
def test_pipeline_mode_thresholds(n_samples, expected):
    assert get_pipeline_mode(n_samples) == expected


def test_every_mode_has_a_description():
    for mode in ("ultra_sparse", "low_sample", "moderate", "full"):
        assert mode in MODE_DESCRIPTIONS
        assert {"label", "network", "clustering", "viz", "color"} <= set(MODE_DESCRIPTIONS[mode])


def test_config_from_dict_ignores_unknown_keys():
    cfg = PipelineConfig.from_dict({"epochs": 5, "latent_dim": 3, "not_a_field": 99})
    assert cfg.epochs == 5
    assert cfg.latent_dim == 3
    assert not hasattr(cfg, "not_a_field")


def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.iterations == 3
    assert cfg.random_seed == 42
    assert cfg.gene_sets  # non-empty default factory
