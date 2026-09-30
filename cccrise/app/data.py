"""Dataset registry and cached loading for the CCC-RISE explorer app."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import anndata
import pandas as pd
import yaml

from cccrise.factors import (
    ALAD_FACTORS_PATH,
    IBD_FACTORS_PATH,
    load_factors_netcdf,
    load_factors_to_anndata,
)
from cccrise.figures.common import get_alad_data, get_ibd_data
from cccrise.runners.datasets import DATA_DIR, IBD_CELL_CAP
from cccrise.runners.datasets import DATASETS as RUNNER_DATASETS
from cccrise.runners.datasets import DatasetSpec as RunnerSpec
from cccrise.runners.provenance import sidecar_path


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing how to load and interpret one dataset."""

    key: str
    label: str
    factors_path: Path
    raw_loader: Callable[[], anndata.AnnData]
    celltype_col: str
    condition_col: str
    group_col: str | None
    #: Value of `group_col` treated as the reference class for the
    #: case/control condition association, and the name given to every other
    #: value. None if no case/control grouping is defined for this dataset.
    control_label: str | None = None
    other_label: str | None = None
    #: Runner registry entry the factors were fit with; names the cached
    #: rank-selection sweeps and the patient column.
    runner: RunnerSpec | None = None


DATASETS: dict[str, DatasetSpec] = {
    "alad": DatasetSpec(
        key="alad",
        label="ALAD Lung Transplant",
        factors_path=ALAD_FACTORS_PATH,
        raw_loader=lambda: get_alad_data(project_cells=True, do_embedding=False),
        celltype_col="broad_cell_type",
        condition_col="dsco_id",
        group_col="ALADstatus",
        control_label="control",
        other_label="alad",
        runner=RUNNER_DATASETS["alad"],
    ),
    "ibd": DatasetSpec(
        key="ibd",
        label="IBDverse Crohn's Disease Terminal Ileum",
        factors_path=IBD_FACTORS_PATH,
        # No embedding: PaCMAP coordinates come from the factors file.
        raw_loader=lambda: get_ibd_data(project_cells=True, do_embedding=False),
        celltype_col="broad_cell_type",
        condition_col="sanger_sample_id",
        group_col="inflammation_status",
        control_label="Healthy",
        other_label="Crohn's",
        runner=RUNNER_DATASETS["ibd"].with_cell_cap(IBD_CELL_CAP),
    ),
}


@dataclass(frozen=True)
class LoadedDataset:
    adata: anndata.AnnData
    spec: DatasetSpec
    #: Per-cell sender/receiver weights (`obsm["sc_B"]`/`obsm["rc_C"]`) are
    #: stored -- drives the Component Explorer's per-cell views.
    has_projections: bool
    #: 2D PaCMAP coordinates (`obsm["PaCMAP"]`) are stored -- drives the Cell
    #: Projections tab specifically. `add_cell_projections` always writes
    #: these together, but a factors file built another way (e.g. a fresh
    #: factorization run through a different pipeline) could have one
    #: without the other, so they're tracked separately rather than assuming
    #: PaCMAP whenever per-cell weights are present.
    has_pacmap: bool


@lru_cache(maxsize=4)
def load_dataset(key: str) -> LoadedDataset:
    """Load a dataset's factor matrices, plus per-cell projections/PaCMAP if
    the `.nc` file was augmented with `add_cell_projections`. This reads
    only the (small) factors file, not the raw single-cell data."""
    spec = DATASETS[key]
    adata = load_factors_to_anndata(spec.factors_path)
    has_projections = "sc_B" in adata.obsm
    has_pacmap = "PaCMAP" in adata.obsm
    return LoadedDataset(
        adata=adata, spec=spec, has_projections=has_projections, has_pacmap=has_pacmap
    )


@lru_cache(maxsize=4)
def load_rise_factors(key: str) -> anndata.AnnData:
    """RISE (PARAFAC2) factors in the layout `scrise.plotting` expects: one
    obs row per condition and one var per gene."""
    spec = DATASETS[key]
    ds = load_factors_netcdf(spec.factors_path)
    conditions = ds.coords["condition"].values.astype(str)
    out = anndata.AnnData(
        obs=pd.DataFrame({spec.condition_col: conditions}, index=conditions),
        var=pd.DataFrame(index=ds.coords["gene"].values.astype(str)),
    )
    out.uns["Pf2_A"] = ds["pf2_A"].values
    out.uns["Pf2_B"] = ds["pf2_B"].values
    out.varm["Pf2_C"] = ds["pf2_C"].values
    return out


@dataclass(frozen=True)
class RawDataset:
    adata: anndata.AnnData | None
    error: str | None = None


@lru_cache(maxsize=4)
def load_raw_dataset(key: str) -> RawDataset:
    """Load raw single-cell expression for a dataset, cached for the life of
    the process. Needed for gene-level interaction heatmaps (actual
    expression values), and for condition associations on `obs` columns the
    factors file doesn't store (see `app.association_source`); not for the
    per-cell projections/PaCMAP already stored in the factors file. Falls
    back gracefully if the raw file isn't available locally.
    """
    spec = DATASETS[key]
    try:
        return RawDataset(adata=spec.raw_loader())
    except (FileNotFoundError, OSError) as exc:
        return RawDataset(
            adata=None,
            error=(
                "Raw single-cell expression data for this dataset was not found "
                f"locally, so interaction heatmaps are unavailable ({exc})."
            ),
        )


@dataclass(frozen=True)
class RankSelection:
    """Cached rank-selection sweeps from `cccrise-riserank`/`cccrise-cpdrank`.
    Either sweep is None if it hasn't been run for this dataset."""

    rise: pd.DataFrame | None
    cpd: pd.DataFrame | None
    #: RISE rank the CPD sweep's interaction tensor was built at.
    cpd_rise_rank: int | None


def _read_sweep(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else None


@lru_cache(maxsize=4)
def load_rank_selection(key: str) -> RankSelection:
    runner_spec = DATASETS[key].runner
    if runner_spec is None:
        return RankSelection(rise=None, cpd=None, cpd_rise_rank=None)
    cpd = _read_sweep(runner_spec.cpd_rank_filename())
    cpd_rise_rank = None
    if cpd is not None:
        sidecar = sidecar_path(DATA_DIR / runner_spec.cpd_rank_filename())
        if sidecar.exists():
            cpd_rise_rank = yaml.safe_load(sidecar.read_text()).get("rise_rank")
    return RankSelection(
        rise=_read_sweep(runner_spec.rise_rank_filename()),
        cpd=cpd,
        cpd_rise_rank=cpd_rise_rank,
    )
