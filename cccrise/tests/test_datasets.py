"""Unit tests for the runner dataset registry and its CLI plumbing."""

import argparse

import pytest

from cccrise.runners._cli import add_dataset_argument, resolve_dataset
from cccrise.runners.datasets import DATASETS, IBD_CELL_CAP


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_argument(parser)
    return parser.parse_args(list(argv))


def test_registry_cap_is_the_registry_entry():
    for argv in (
        ("--dataset", "ibd"),
        ("--dataset", "ibd", "--max-cells-per-condition", "2000"),
    ):
        spec = resolve_dataset(_parse(*argv))

        assert spec is DATASETS["ibd"]
        assert spec.output_key == "ibd"
        assert spec.loader_args["max_cells_per_condition"] == IBD_CELL_CAP
        assert spec.rise_rank_filename() == "rise_rank_ibd.parquet"
        assert "(<= 2000 cells per condition, seed 0)" in spec.label


def test_another_cell_cap_reaches_the_loader_and_renames_every_output():
    spec = resolve_dataset(
        _parse("--dataset", "ibd", "--max-cells-per-condition", "500")
    )

    assert spec.key == "ibd"  # registry lookups (group labels) still resolve
    assert spec.loader_args["max_cells_per_condition"] == 500
    assert spec.loader_args["cell_cap_seed"] == 0
    assert spec.rise_rank_filename() == "rise_rank_ibd_cap500.parquet"
    assert spec.factors_filename == "ibd_ti_factors_cap500.nc"
    assert spec.factors_path() != DATASETS["ibd"].factors_path()
    assert spec.label.count("cells per condition") == 1


def test_a_non_default_cap_seed_is_part_of_the_output_name():
    spec = resolve_dataset(
        _parse(
            "--dataset",
            "ibd",
            "--max-cells-per-condition",
            "500",
            "--cell-cap-seed",
            "7",
        )
    )

    assert spec.output_key == "ibd_cap500s7"
    assert spec.loader_args["cell_cap_seed"] == 7


def test_cell_cap_is_refused_for_a_loader_without_one():
    with pytest.raises(ValueError, match="does not support a cell cap"):
        DATASETS["alad"].with_cell_cap(100)


def test_ibd_factor_model_is_committed():
    from cccrise.factors import IBD_FACTORS_PATH

    assert IBD_FACTORS_PATH == DATASETS["ibd"].factors_path()
    assert IBD_FACTORS_PATH.name == "ibd_ti_factors.nc"
    assert (
        IBD_FACTORS_PATH.exists()
        or IBD_FACTORS_PATH.with_name(IBD_FACTORS_PATH.name + ".xz").exists()
    )


def test_ibd_fits_on_cpu_at_any_cap():
    assert DATASETS["ibd"].backend == "cpu"
    assert DATASETS["ibd"].with_cell_cap(500).backend == "cpu"


def test_ibd_excludes_the_mesenchymal_lineage_and_records_it():
    from cccrise.import_data import IBD_EXCLUDED_BROAD_CELL_TYPES

    assert IBD_EXCLUDED_BROAD_CELL_TYPES == ("Mesenchymal",)
    for spec in (DATASETS["ibd"], DATASETS["ibd"].with_cell_cap(500)):
        assert spec.loader_args["excluded_broad_cell_types"] == ["Mesenchymal"]


def test_alad_loader_args_have_no_lineage_exclusion():
    assert "excluded_broad_cell_types" not in DATASETS["alad"].loader_args
