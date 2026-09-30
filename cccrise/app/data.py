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
    COVID19_FACTORS_PATH,
    load_factors_netcdf,
    load_factors_to_anndata,
)
from cccrise.figures.common import get_alad_data, get_covid_data
from cccrise.runners.datasets import DATA_DIR
from cccrise.runners.datasets import DATASETS as RUNNER_DATASETS
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
    #: Value of `group_col` treated as the reference class for the logistic
    #: regression weight summary, and the name given to every other value.
    #: None if no case/control grouping is defined for this dataset.
    control_label: str | None = None
    other_label: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "covid": DatasetSpec(
        key="covid",
        label="COVID-19 BALF",
        factors_path=COVID19_FACTORS_PATH,
        # No embedding: PaCMAP coordinates come from the factors file when
        # it stores them.
        raw_loader=lambda: get_covid_data(project_cells=True, do_embedding=False),
        celltype_col="celltype",
        condition_col="sample",
        group_col="condition",
        control_label="Control",
        other_label="COVID-19",
    ),
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
    ),
}


@dataclass(frozen=True)
class LoadedDataset:
    adata: anndata.AnnData
    spec: DatasetSpec
    #: Per-cell sender/receiver weights (`obsm["sc_B"]`/`obsm["rc_C"]`) are
    #: available -- recomputed from the raw data, so absent without it.
    #: Drives the Component Explorer's per-cell views.
    has_projections: bool
    #: 2D PaCMAP coordinates (`obsm["PaCMAP"]`) are available -- drives the
    #: Cell Projections tab. Stored in the factors file only when the fit ran
    #: PaCMAP, so tracked separately from `has_projections`.
    has_pacmap: bool


@lru_cache(maxsize=4)
def load_dataset(key: str) -> LoadedDataset:
    """Load a dataset's factor matrices, plus per-cell projections, sender/
    receiver weights and PaCMAP, which need the raw single-cell data
    (`load_raw_dataset`) to recompute; without it, only the factor matrices."""
    spec = DATASETS[key]
    raw = load_raw_dataset(key).adata
    adata = raw if raw is not None else load_factors_to_anndata(spec.factors_path)
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
    expression values), the boot-time logistic regression fit (which
    needs real per-cell counts for its read-depth correction), and the
    per-cell projections `load_dataset` recomputes. Falls back gracefully if
    the raw file isn't available locally.

    The app's boot-time logreg fit (see `app.py`) calls this for every
    dataset up front, so in practice it's rarely "lazy" in a running
    server -- by the time a session opens an interaction heatmap, the raw
    data is normally already loaded and cached from that.
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
    return pd.read_parquet(path) if path.exists() else None


@lru_cache(maxsize=4)
def load_rank_selection(key: str) -> RankSelection:
    runner_spec = RUNNER_DATASETS.get(key)
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
