"""Tests for the runner scripts (dataset registry, provenance, pipelines)."""

import anndata
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.sparse import csr_array

from cellcommunicationpf2.ccc_rise import validate_cp_rank
from cellcommunicationpf2.factors import load_factors_to_anndata
from cellcommunicationpf2.import_data import (
    add_cond_idxs,
    import_ligand_receptor_pairs,
)
from cellcommunicationpf2.runners import provenance
from cellcommunicationpf2.runners._cli import parse_ranks
from cellcommunicationpf2.runners.datasets import DATASETS, DatasetSpec
from cellcommunicationpf2.runners.run_ccc_rise import run_pipeline
from cellcommunicationpf2.runners.run_cpd_rank import run_cpd_rank_selection
from cellcommunicationpf2.runners.run_rise_rank import run_rise_rank_selection

N_CELLS = 120
N_GENES = 30
N_COND = 4
RISE_RANK = 3
CP_RANK = 2


def _synthetic_adata(seed: int = 0) -> anndata.AnnData:
    """A tiny AnnData whose genes are real ligands/receptors, so the LR
    annotation matches and the communication tensor is non-empty."""
    lr = import_ligand_receptor_pairs()
    # Whole pairs only -- picking genes independently would leave every ligand
    # without its receptor and produce an empty communication tensor.
    simple = lr[~lr["ligand"].str.contains("&") & ~lr["receptor"].str.contains("&")]
    genes: list[str] = []
    for ligand, receptor in zip(simple["ligand"], simple["receptor"], strict=True):
        genes.extend(g for g in (ligand, receptor) if g not in genes)
        if len(genes) >= N_GENES:
            break

    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=3.0, size=(N_CELLS, len(genes))).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sample": [f"S{i % N_COND}" for i in range(N_CELLS)],
            "celltype": ["CT_A" if i % 2 else "CT_B" for i in range(N_CELLS)],
        },
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    adata = anndata.AnnData(
        X=csr_array(counts), obs=obs, var=pd.DataFrame(index=pd.Index(genes))
    )
    adata.var["means"] = np.asarray(counts.mean(axis=0)).ravel()
    return add_cond_idxs(adata, "sample")


@pytest.fixture
def synthetic_spec(tmp_path) -> DatasetSpec:
    """A DatasetSpec backed by synthetic data, usable by every runner.

    `source_file` points at a real stand-in file so the provenance path that
    checksums the source data is exercised.
    """
    source = tmp_path / "synthetic_source.h5ad"
    source.write_bytes(b"stand-in for the source data")
    return DatasetSpec(
        key="synthetic",
        label="Synthetic test dataset",
        loader=lambda filename=None, gene_threshold=0.0, normalize=True: (
            _synthetic_adata()
        ),
        condition_column="sample",
        factors_filename="synthetic_factors.nc",
        source_file=source,
        celltype_column="celltype",
    )


# --------------------------------------------------------------------------
# Registry and CLI helpers
# --------------------------------------------------------------------------


def test_registry_entries_are_consistent():
    for key, spec in DATASETS.items():
        assert spec.key == key
        assert spec.factors_filename.endswith(".nc")
        assert spec.rise_rank_filename() == f"rise_rank_{key}.csv"
        assert spec.cpd_rank_filename() == f"cpd_rank_{key}.csv"
        assert spec.loader_path.startswith("cellcommunicationpf2.import_data:")
        assert spec.loader_args["filename"] == str(spec.source_file)
        assert spec.loader_args["normalize"] is True


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("5,10,15", [5, 10, 15]),
        ("1:6", [1, 2, 3, 4, 5]),
        ("5:21:5", [5, 10, 15, 20]),
        ("1:4,10,4", [1, 2, 3, 4, 10]),
    ],
)
def test_parse_ranks(spec, expected):
    assert parse_ranks(spec) == expected


def test_validate_cp_rank():
    validate_cp_rank(15, (47, 15, 15, 513))
    with pytest.raises(ValueError, match="exceeds the smallest mode"):
        validate_cp_rank(16, (47, 15, 15, 513))


# --------------------------------------------------------------------------
# Provenance sidecars
# --------------------------------------------------------------------------


def test_write_and_read_metadata_roundtrip(tmp_path):
    output = tmp_path / "result.csv"
    output.write_text("a,b\n1,2\n")

    meta = provenance.build_metadata(
        "unit_test",
        {
            "dataset": "synthetic",
            "n_cells": np.int64(5),
            "ranks": np.arange(1, 4),
            "r2x_cp": np.float64(0.5),
            "outputs": {"csv": provenance.file_info(output)},
        },
    )
    sidecar = provenance.write_metadata(output, meta, header="header line")

    assert sidecar == tmp_path / "result.provenance.yaml"
    text = sidecar.read_text()
    assert text.startswith("# header line")

    loaded = yaml.safe_load(text)
    # numpy scalars/arrays must survive as plain YAML builtins
    assert loaded["n_cells"] == 5
    assert loaded["ranks"] == [1, 2, 3]
    assert loaded["r2x_cp"] == pytest.approx(0.5)
    assert loaded["outputs"]["csv"]["sha256"] == provenance.sha256_file(output)
    assert loaded["outputs"]["csv"]["bytes"] == output.stat().st_size
    assert set(loaded) >= {
        "dataset",
        "outputs",
        "git_sha",
        "package_versions",
        "generated",
        "runner",
    }
    assert set(loaded["package_versions"]) >= {"tensorly", "parafac2", "numpy"}


# --------------------------------------------------------------------------
# Runners, end to end on synthetic data
# --------------------------------------------------------------------------


def test_rise_rank_runner(synthetic_spec, tmp_path):
    output = tmp_path / "rise.csv"
    df, csv_path, sidecar = run_rise_rank_selection(
        synthetic_spec,
        ranks=[2, 3],
        n_folds=2,
        n_iter_max=10,
        output_path=output,
        selected_rank=3,
        notes="chosen for the test",
    )

    assert csv_path == output and output.exists()
    assert set(df["Rank"]) == {2, 3}
    assert len(df) == 4

    meta = yaml.safe_load(sidecar.read_text())
    assert meta["runner"] == "rise_rank_bicv"
    assert meta["method"] == "bicv"
    assert meta["ranks"] == [2, 3]
    assert meta["n_folds"] == 2
    assert meta["condition_key"] == "sample"
    assert meta["source_sha256"] == provenance.sha256_file(synthetic_spec.source_file)
    assert meta["resulting_dims"] == {
        "conditions": N_COND,
        "genes": N_GENES,
        "cells": N_CELLS,
    }
    assert meta["best_rank_by_mean_held_out_r2x"] in (2, 3)
    assert len(meta["per_rank"]) == 2
    assert meta["selection"] == {"selected_rank": 3, "notes": "chosen for the test"}


def test_cpd_rank_runner(synthetic_spec, tmp_path):
    output = tmp_path / "cpd.csv"
    df, csv_path, sidecar = run_cpd_rank_selection(
        synthetic_spec,
        rise_rank=RISE_RANK,
        ranks=[1, 2],
        n_masks=2,
        n_iter_max=10,
        output_path=output,
    )

    assert csv_path == output and output.exists()
    assert set(df["Rank"]) == {1, 2}

    meta = yaml.safe_load(sidecar.read_text())
    assert meta["runner"] == "cpd_rank_masked_imputation"
    assert meta["rise_rank"] == RISE_RANK
    assert meta["cp_ranks"] == [1, 2]
    shape = meta["interaction_tensor_shape"]
    assert shape[0] == N_COND and shape[1] == shape[2] == RISE_RANK
    assert meta["resulting_dims"]["lr_pairs"] == shape[3]


def test_cpd_rank_runner_fits_rise_once_for_the_whole_sweep(
    synthetic_spec, tmp_path, capsys
):
    """Every CP rank in a sweep must be scored against the same RISE factors."""
    run_cpd_rank_selection(
        synthetic_spec,
        rise_rank=RISE_RANK,
        ranks=[1, 2, 3],
        n_masks=2,
        n_iter_max=10,
        output_path=tmp_path / "cpd.csv",
    )
    assert capsys.readouterr().out.count("Loading data and fitting RISE") == 1


def test_cpd_rank_runner_rejects_overlarge_cp_rank(synthetic_spec, tmp_path):
    with pytest.raises(ValueError, match="exceeds the smallest mode"):
        run_cpd_rank_selection(
            synthetic_spec,
            rise_rank=RISE_RANK,
            ranks=[1, RISE_RANK + 5],
            n_masks=1,
            n_iter_max=10,
            output_path=tmp_path / "cpd.csv",
        )


def test_pipeline_runner_writes_readable_factors_and_provenance(
    synthetic_spec, tmp_path
):
    output = tmp_path / "factors.nc"
    _adata, ds, nc_path, sidecar = run_pipeline(
        synthetic_spec,
        rise_rank=RISE_RANK,
        cp_rank=CP_RANK,
        n_iter_max=20,
        random_state=0,
        output_path=output,
        rise_rank_notes="elbow of the BiCV curve",
        cpd_rank_notes="plateau of held-out entry R2X",
    )

    assert nc_path == output and output.exists()
    assert ds.sizes["cp_component"] == CP_RANK
    assert ds.sizes["rise_component"] == RISE_RANK
    assert ds.sizes["condition"] == N_COND

    # PARAFAC2 factors must be stored so consumers can re-project raw cells.
    for var in ("pf2_A", "pf2_B", "pf2_C", "pf2_weights"):
        assert var in ds

    # The .nc must round-trip into the AnnData the figure modules read.
    loaded = load_factors_to_anndata(output)
    assert loaded.uns["A"].shape == (N_COND, CP_RANK)
    assert loaded.uns["B"].shape == (RISE_RANK, CP_RANK)
    assert loaded.uns["D"].shape == (len(loaded.uns["lr_pairs"]), CP_RANK)

    meta = yaml.safe_load(sidecar.read_text())
    assert meta["runner"] == "ccc_rise_pipeline"
    assert meta["rise_rank"] == RISE_RANK
    assert meta["cp_rank"] == CP_RANK
    assert meta["random_state"] == 0
    assert 0.0 <= meta["r2x_cp"] <= 1.0
    assert meta["outputs"]["factors_nc"]["sha256"]
    assert meta["condition_key"] == "sample"
    assert meta["resulting_dims"] == {
        "conditions": N_COND,
        "genes": N_GENES,
        "cells": N_CELLS,
        "lr_pairs": len(loaded.uns["lr_pairs"]),
    }

    rs = meta["rank_selection"]
    assert rs["rise"]["rank"] == RISE_RANK
    assert rs["rise"]["method"] == "bicv"
    assert rs["rise"]["notes"] == "elbow of the BiCV curve"
    assert rs["cpd"]["rank"] == CP_RANK
    assert rs["cpd"]["notes"] == "plateau of held-out entry R2X"


def test_pipeline_rejects_overlarge_cp_rank_before_fitting(synthetic_spec, tmp_path):
    with pytest.raises(ValueError, match="exceeds the smallest mode"):
        run_pipeline(
            synthetic_spec,
            rise_rank=RISE_RANK,
            cp_rank=RISE_RANK + 1,
            n_iter_max=20,
            output_path=tmp_path / "never_written.nc",
        )
    assert not (tmp_path / "never_written.nc").exists()
