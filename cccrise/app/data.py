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
        # No embedding: the app only ever reads raw cells for gene-level
        # heatmaps and the boot-time logreg fit, neither of which touches
        # PaCMAP coordinates (those come from the factors file instead, via
        # `load_dataset`, when present).
        raw_loader=lambda: get_covid_data(project_cells=True, do_embedding=False),
        celltype_col="celltype",
        condition_col="sample",
        group_col="condition",
        control_label="Control",
        other_label="COVID-19",
    ),
    # Registering IBD here: its cap-2000 factor model is committed
    # xz-compressed; run `xz -dk cccrise/data/ibd_ti_factors_cap2000.nc.xz`
    # first and point `factors_path` at the decompressed `.nc`.
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


@dataclass(frozen=True)
class RawDataset:
    adata: anndata.AnnData | None
    error: str | None = None


@lru_cache(maxsize=4)
def load_raw_dataset(key: str) -> RawDataset:
    """Load raw single-cell expression for a dataset, cached for the life of
    the process. Needed for gene-level interaction heatmaps (actual
    expression values) and the boot-time logistic regression fit (which
    needs real per-cell counts for its read-depth correction), not for the
    per-cell projections/PaCMAP already stored in the factors file. Falls
    back gracefully if the raw file isn't available locally.

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
    return pd.read_csv(path) if path.exists() else None


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
