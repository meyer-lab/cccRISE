"""Tests for the explorer app's dataset registry (no Shiny session needed)."""

import pytest

from cccrise.app import app, data
from cccrise.runners._cli import DEFAULT_GROUP_LABELS

#: IBD is registered only once its refit factor model exists (see `data.DATASETS`).
needs_ibd = pytest.mark.skipif(
    "ibd" not in data.DATASETS, reason="IBD factor model not yet refit"
)


@pytest.mark.parametrize("key", list(data.DATASETS))
def test_every_dataset_has_comparisons_and_a_valid_default(key):
    assert app.DEFAULT_COMPARISONS[key] in app.DATASET_COMPARISONS[key]


@needs_ibd
def test_ibd_comparisons_test_its_inflammation_levels():
    spec = data.DATASETS["ibd"]
    levels = {"Healthy", "CD_inflamed", "CD_uninflamed"}
    inflammation = [
        cfg
        for cfg in app.DATASET_COMPARISONS["ibd"].values()
        if cfg["group_column"] == spec.group_col
    ]

    assert len(inflammation) == 5  # all Crohn's, each pair of levels, 3-way
    for cfg in inflammation:
        assert set(cfg.get("levels", levels)) <= levels
        assert cfg.get("control_label", "Healthy") in levels


@pytest.mark.parametrize("key", list(data.DATASETS))
def test_every_dataset_finds_its_cached_rank_sweeps(key):
    sweeps = data.load_rank_selection(key)

    assert sweeps.rise is not None
    assert sweeps.cpd is not None


@needs_ibd
def test_ibd_group_labels_agree_with_the_runner_default():
    spec = data.DATASETS["ibd"]

    assert (spec.control_label, spec.other_label) == DEFAULT_GROUP_LABELS["ibd"]
