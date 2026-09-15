"""Dataset registry and cached loading for the CCC-RISE explorer app."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import anndata

from cccrise.factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    load_factors_to_anndata,
)
from cccrise.figures.common import get_alad_data, get_covid_data


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing how to load and interpret one dataset."""

    key: str
    label: str
    factors_path: Path
    raw_loader: object  # Callable[[], anndata.AnnData]
    celltype_col: str
    condition_col: str
    group_col: str | None


DATASETS: dict[str, DatasetSpec] = {
    "covid": DatasetSpec(
        key="covid",
        label="COVID-19 BALF",
        factors_path=COVID19_FACTORS_PATH,
        raw_loader=lambda: get_covid_data(project_cells=True, do_embedding=True),
        celltype_col="celltype",
        condition_col="sample",
        group_col="condition",
    ),
    "alad": DatasetSpec(
        key="alad",
        label="ALAD Lung Transplant",
        factors_path=ALAD_FACTORS_PATH,
        raw_loader=lambda: get_alad_data(project_cells=True, do_embedding=True),
        celltype_col="broad_cell_type",
        condition_col="dsco_id",
        group_col="ALADstatus",
    ),
}


@dataclass(frozen=True)
class LoadedDataset:
    adata: anndata.AnnData
    spec: DatasetSpec
    has_projections: bool


@lru_cache(maxsize=4)
def load_dataset(key: str) -> LoadedDataset:
    """Load a dataset's factor matrices, plus per-cell projections/PaCMAP if
    the `.nc` file was augmented with `add_cell_projections`. This reads
    only the (small) factors file, not the raw single-cell data."""
    spec = DATASETS[key]
    adata = load_factors_to_anndata(spec.factors_path)
    has_projections = "sc_B" in adata.obsm
    return LoadedDataset(adata=adata, spec=spec, has_projections=has_projections)


@dataclass(frozen=True)
class RawDataset:
    adata: anndata.AnnData | None
    error: str | None = None


@lru_cache(maxsize=4)
def load_raw_dataset(key: str) -> RawDataset:
    """Lazily load raw single-cell expression for a dataset. Only needed for
    gene-level interaction heatmaps (actual expression values), not for the
    per-cell projections/PaCMAP already stored in the factors file. Falls
    back gracefully if the raw file isn't available locally."""
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
