"""Unit tests for the runner dataset registry and its CLI plumbing."""

import argparse

import pytest

from cccrise.runners._cli import add_dataset_argument, resolve_dataset
from cccrise.runners.datasets import DATASETS


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_argument(parser)
    return parser.parse_args(list(argv))


def test_uncapped_dataset_is_the_registry_entry():
    spec = resolve_dataset(_parse("--dataset", "ibd"))

    assert spec is DATASETS["ibd"]
    assert spec.output_key == "ibd"
    assert "max_cells_per_condition" not in spec.loader_args


def test_cell_cap_reaches_the_loader_and_renames_every_output():
    spec = resolve_dataset(
        _parse("--dataset", "ibd", "--max-cells-per-condition", "2000")
    )

    assert spec.key == "ibd"  # registry lookups (group labels) still resolve
    assert spec.loader_args["max_cells_per_condition"] == 2000
    assert spec.loader_args["cell_cap_seed"] == 0
    assert spec.rise_rank_filename() == "rise_rank_ibd_cap2000.parquet"
    assert spec.factors_filename == "ibd_ti_factors_cap2000.nc"
    assert spec.factors_path() != DATASETS["ibd"].factors_path()


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


def test_ibd_cell_cap_names_the_committed_factor_model():
    from cccrise.factors import IBD_FACTORS_PATH
    from cccrise.runners.datasets import IBD_CELL_CAP

    capped = DATASETS["ibd"].with_cell_cap(IBD_CELL_CAP)

    assert capped.factors_path() == IBD_FACTORS_PATH
    # TODO: assert the file (or its .xz) exists again once the model is refit
    # without the mesenchymal lineage; the earlier fit is retired.
    assert IBD_FACTORS_PATH.name == "ibd_ti_factors_cap2000.nc"


def test_ibd_fits_on_cpu_capped_or_not():
    from cccrise.runners.datasets import IBD_CELL_CAP

    assert DATASETS["ibd"].backend == "cpu"
    assert DATASETS["ibd"].with_cell_cap(IBD_CELL_CAP).backend == "cpu"


def test_ibd_excludes_the_mesenchymal_lineage_and_records_it():
    from cccrise.import_data import IBD_EXCLUDED_BROAD_CELL_TYPES
    from cccrise.runners.datasets import IBD_CELL_CAP

    assert IBD_EXCLUDED_BROAD_CELL_TYPES == ("Mesenchymal",)
    for spec in (DATASETS["ibd"], DATASETS["ibd"].with_cell_cap(IBD_CELL_CAP)):
        assert spec.loader_args["excluded_broad_cell_types"] == ["Mesenchymal"]


def test_alad_loader_args_have_no_lineage_exclusion():
    assert "excluded_broad_cell_types" not in DATASETS["alad"].loader_args
