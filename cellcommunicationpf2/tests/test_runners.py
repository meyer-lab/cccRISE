"""Tests for the runner scripts (dataset registry, YAML provenance, pipelines)."""

import argparse

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
from cellcommunicationpf2.runners.datasets import DATASETS, DatasetSpec, get_dataset
from cellcommunicationpf2.runners.run_ccc_rise import run_pipeline
from cellcommunicationpf2.runners.run_cpd_rank import run_cpd_rank_selection
from cellcommunicationpf2.runners.run_rise_rank import run_rise_rank_selection

N_CELLS = 120
N_COND = 4
RISE_RANK = 3
CP_RANK = 2


def _synthetic_adata(seed: int = 0) -> anndata.AnnData:
    """A tiny AnnData whose genes are real ligands/receptors, so the LR
    annotation actually matches and the communication tensor is non-empty."""
    lr = import_ligand_receptor_pairs()
    # Whole pairs only -- picking genes independently would leave every ligand
    # without its receptor and produce an empty communication tensor.
    simple = lr[~lr["ligand"].str.contains("&") & ~lr["receptor"].str.contains("&")]
    genes: list[str] = []
    for ligand, receptor in zip(simple["ligand"], simple["receptor"], strict=True):
        genes.extend(g for g in (ligand, receptor) if g not in genes)
        if len(genes) >= 30:
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

    `source_file` points at a real (stand-in) file so the provenance path that
    checksums the raw input is exercised rather than skipped.
    """
    source = tmp_path / "synthetic_source.h5ad"
    source.write_bytes(b"stand-in for the raw .h5ad")
    return DatasetSpec(
        key="synthetic",
        label="Synthetic test dataset",
        loader=lambda gene_threshold=0.0, normalize=True, filename=None: (
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
        assert spec.rise_rank_filename() == f"rise_rank_bicv_{key}.csv"
        assert spec.cpd_rank_filename(7) == f"cpd_rank_imputation_{key}_rise7.csv"
        assert callable(spec.loader)
        assert spec.loader_path.startswith("cellcommunicationpf2.import_data:")
        assert spec.source_file is not None
        assert spec.loader_args["normalize"] is True
        assert spec.key != "covid"


def test_get_dataset_is_case_insensitive_and_errors_helpfully():
    assert get_dataset("ALAD").key == "alad"
    assert get_dataset("IBD").key == "ibd"
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("nope")


def test_registry_holds_only_the_active_datasets():
    """COVID is being removed from the project, so it is not a runner target."""
    assert set(DATASETS) == {"alad", "ibd"}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("5,10,15", [5, 10, 15]),
        ("1:6", [1, 2, 3, 4, 5]),
        ("5:21:5", [5, 10, 15, 20]),
        ("1:4,10, 4", [1, 2, 3, 4, 10]),
    ],
)
def test_parse_ranks(spec, expected):
    assert parse_ranks(spec) == expected


def test_parse_ranks_rejects_bad_input():
    with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
        parse_ranks("0:3")
    with pytest.raises(argparse.ArgumentTypeError, match="Bad rank slice"):
        parse_ranks("1:2:3:4")


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
            "tol": np.float32(1e-3),
            "r2x_cp": np.float64(0.5),
            "outputs": {"csv": provenance.file_info(output)},
        },
    )
    sidecar = provenance.write_metadata(output, meta, header="header line")

    # Issue #164 documents the sidecar as `<stem>.provenance.yaml`.
    assert sidecar == tmp_path / "result.provenance.yaml"
    text = sidecar.read_text()
    assert text.startswith("# header line")

    loaded = yaml.safe_load(text)
    # numpy scalars/arrays must survive as plain YAML builtins
    assert loaded["n_cells"] == 5
    assert loaded["ranks"] == [1, 2, 3]
    assert loaded["r2x_cp"] == pytest.approx(0.5)
    assert loaded["outputs"]["csv"]["sha256"]
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


def test_source_info_records_checksum_and_skips_on_request(tmp_path):
    src = tmp_path / "raw.h5ad"
    src.write_bytes(b"not really an h5ad")

    info = provenance.source_info(src)
    assert info["source_file"] == str(src)
    assert info["source_sha256"] == provenance.sha256_file(src)

    assert provenance.source_info(src, checksum=False)["source_sha256"] == "skipped"
    assert (
        provenance.source_info(tmp_path / "gone.h5ad")["source_sha256"]
        == "file not found"
    )
    assert provenance.source_info(None) == {
        "source_file": None,
        "source_sha256": None,
    }


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
    assert meta["resulting_dims"]["cells"] == N_CELLS
    assert meta["resulting_dims"]["conditions"] == N_COND
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
    assert meta["max_cp_rank"] == RISE_RANK
    shape = meta["interaction_tensor_shape"]
    assert shape[0] == N_COND and shape[1] == shape[2] == RISE_RANK
    # The tensor cache is an internal temp-file detail, not a recorded parameter.
    assert "tensor_cache" not in meta


def test_cpd_tensor_cache_is_internal_and_reused(synthetic_spec, tmp_path, capsys):
    """A second sweep with the same tensor-determining parameters must reuse the
    temp cache rather than refitting RISE."""
    for name in ("first", "second"):
        run_cpd_rank_selection(
            synthetic_spec,
            rise_rank=RISE_RANK,
            ranks=[1],
            n_masks=1,
            n_iter_max=10,
            output_path=tmp_path / f"{name}.csv",
        )
    out = capsys.readouterr().out
    assert out.count("Loading data and fitting RISE") == 1
    assert out.count("Reusing cached interaction tensor") == 1


def test_pipeline_runner_writes_readable_factors_and_provenance(
    synthetic_spec, tmp_path
):
    output = tmp_path / "factors.nc"
    _adata, ds, nc_path, sidecar = run_pipeline(
        synthetic_spec,
        rise_rank=RISE_RANK,
        cp_rank=CP_RANK,
        n_iter_max=20,
        tol=1e-3,
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
    # The three dimensions on which the stored ALAD factors failed to
    # reproduce (issue #164) are recorded explicitly.
    assert meta["resulting_dims"] == {
        "conditions": N_COND,
        "genes": 30,
        "cells": N_CELLS,
        "lr_pairs": len(loaded.uns["lr_pairs"]),
    }
    assert "cell_projections" not in ds  # per-cell storage belongs to shiny
    assert ":" in meta["loader"]  # module:function, per issue #164
    assert meta["source_sha256"] == provenance.sha256_file(synthetic_spec.source_file)
    assert meta["condition_key"] == "sample"

    rs = meta["rank_selection"]
    assert rs["rise"]["rank"] == RISE_RANK
    assert rs["rise"]["method"] == "bicv"
    assert rs["rise"]["notes"] == "elbow of the BiCV curve"
    assert rs["cpd"]["rank"] == CP_RANK
    assert rs["cpd"]["method"] == "masked_entry_imputation_cv"
    assert rs["cpd"]["notes"] == "plateau of held-out entry R2X"


def test_pipeline_cites_rank_selection_sidecars(synthetic_spec, tmp_path):
    """The pipeline's rank_selection block should be seeded from the sweep
    sidecars written by the rank-selection runners."""
    _, _, rise_sidecar = run_rise_rank_selection(
        synthetic_spec,
        ranks=[2, 3],
        n_folds=2,
        n_iter_max=10,
        output_path=tmp_path / "rise.csv",
        selected_rank=RISE_RANK,
        notes="from the sweep",
    )

    _, _, _, sidecar = run_pipeline(
        synthetic_spec,
        rise_rank=RISE_RANK,
        cp_rank=CP_RANK,
        n_iter_max=20,
        random_state=0,
        output_path=tmp_path / "factors.nc",
        rise_rank_sidecar=rise_sidecar,
    )

    rise_block = yaml.safe_load(sidecar.read_text())["rank_selection"]["rise"]
    assert rise_block["sweep_yaml"] == provenance.relpath(rise_sidecar)
    assert rise_block["sweep_selected_rank"] == RISE_RANK
    assert rise_block["notes"] == "from the sweep"
    assert rise_block["sweep_git_commit"] is not None


# --------------------------------------------------------------------------
# cp_rank guard (issue #164)
# --------------------------------------------------------------------------


def test_validate_cp_rank_allows_rank_up_to_smallest_mode():
    validate_cp_rank(15, (47, 15, 15, 513))
    validate_cp_rank(12, (12, 35, 35, 895))


def test_validate_cp_rank_rejects_the_published_alad_configuration():
    """rise 15 / cp 18 on ALAD is the configuration that triggers a 975 GiB
    allocation inside tensorly's SVD init (issue #164)."""
    with pytest.raises(ValueError, match=r"cp_rank=18 exceeds the smallest mode"):
        validate_cp_rank(18, (47, 15, 15, 513))


def test_pipeline_rejects_overlarge_cp_rank_before_fitting(synthetic_spec, tmp_path):
    """The guard must fire on the cheap pre-check, not after an hours-long fit."""
    with pytest.raises(ValueError, match="exceeds the smallest mode"):
        run_pipeline(
            synthetic_spec,
            rise_rank=RISE_RANK,
            cp_rank=RISE_RANK + 1,
            n_iter_max=20,
            output_path=tmp_path / "never_written.nc",
        )
    assert not (tmp_path / "never_written.nc").exists()


def test_cpd_sweep_skips_ranks_over_the_smallest_mode(synthetic_spec, tmp_path):
    """Ranks past the cap are dropped and recorded, not left to blow up mid-sweep."""
    df, _, sidecar = run_cpd_rank_selection(
        synthetic_spec,
        rise_rank=RISE_RANK,
        ranks=[2, RISE_RANK + 5],
        n_masks=1,
        n_iter_max=10,
        output_path=tmp_path / "cpd.csv",
    )

    assert set(df["Rank"]) == {2}
    meta = yaml.safe_load(sidecar.read_text())
    assert meta["cp_ranks"] == [2]
    assert meta["cp_ranks_skipped_over_max_mode"] == [RISE_RANK + 5]
    assert meta["max_cp_rank"] == RISE_RANK


def test_cpd_runner_fms_elbow_sweep(synthetic_spec, tmp_path):
    """--fms-runs adds the second criterion issue #164 asks for."""
    _, csv_path, sidecar = run_cpd_rank_selection(
        synthetic_spec,
        rise_rank=RISE_RANK,
        ranks=[1, 2],
        n_masks=1,
        n_iter_max=10,
        fms_runs=2,
        output_path=tmp_path / "cpd.csv",
    )

    fms_csv = csv_path.with_name(f"{csv_path.stem}_fms.csv")
    assert fms_csv.exists()
    fms_df = pd.read_csv(fms_csv)
    assert set(fms_df.columns) >= {"Run", "Component", "FMS", "R2X"}
    assert set(fms_df["Component"]) == {1, 2}

    meta = yaml.safe_load(sidecar.read_text())
    assert meta["method"] == "masked_entry_imputation_cv + fms_elbow"
    assert meta["fms_runs"] == 2
    assert meta["best_rank_by_mean_fms"] in (1, 2)
    assert meta["outputs"]["fms_csv"]["sha256"]


def test_pipeline_cites_the_fms_curve_as_rank_selection_evidence(
    synthetic_spec, tmp_path
):
    _, cpd_csv, cpd_sidecar = run_cpd_rank_selection(
        synthetic_spec,
        rise_rank=RISE_RANK,
        ranks=[1, 2],
        n_masks=1,
        n_iter_max=10,
        fms_runs=2,
        output_path=tmp_path / "cpd.csv",
        selected_rank=CP_RANK,
        notes="elbow of the FMS curve",
    )

    _, _, _, sidecar = run_pipeline(
        synthetic_spec,
        rise_rank=RISE_RANK,
        cp_rank=CP_RANK,
        n_iter_max=20,
        random_state=0,
        output_path=tmp_path / "factors.nc",
        cpd_rank_sidecar=cpd_sidecar,
    )

    meta = yaml.safe_load(sidecar.read_text())
    cpd_block = meta["rank_selection"]["cpd"]
    assert cpd_block["sweep_csv"] == provenance.relpath(cpd_csv)
    assert cpd_block["fms_csv"] == provenance.relpath(
        cpd_csv.with_name(f"{cpd_csv.stem}_fms.csv")
    )
    assert cpd_block["sweep_selected_rank"] == CP_RANK
    assert cpd_block["notes"] == "elbow of the FMS curve"
    # Both evidence files are listed at the top level, per issue #164.
    assert set(meta["rank_selection_evidence"]) == {
        cpd_block["sweep_csv"],
        cpd_block["fms_csv"],
    }
