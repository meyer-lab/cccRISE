"""
Dataset registry shared by every runner.

A runner is told which dataset to work on by a short label (``alad``, ``ibd``);
the loader, the source file, the condition column and the canonical output
filenames are looked up here. That keeps "which dataset?" a single CLI argument
and keeps result files named consistently across runners and figures.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata

from ..import_data import IBD_TI_IVCSC, import_alad, import_ibdverse

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything a runner needs to know about one dataset."""

    key: str
    label: str
    loader: Callable[..., anndata.AnnData]
    condition_column: str
    factors_filename: str
    #: Source file the loader reads, recorded with its sha256 in every sidecar.
    source_file: Path
    gene_threshold: float = 0.001
    celltype_column: str | None = None
    group_column: str | None = None
    notes: str = ""

    @property
    def loader_path(self) -> str:
        """``module:function`` name of the loader, recorded in the sidecar."""
        module = getattr(self.loader, "__module__", "?")
        name = getattr(self.loader, "__qualname__", repr(self.loader))
        return f"{module}:{name}"

    @property
    def loader_args(self) -> dict[str, Any]:
        """The keyword arguments `load` passes, recorded in the sidecar."""
        return {
            "filename": str(self.source_file),
            "gene_threshold": self.gene_threshold,
            "normalize": True,
        }

    def load(self) -> anndata.AnnData:
        """Load the dataset, normalized."""
        return self.loader(**self.loader_args)

    def factors_path(self) -> Path:
        return DATA_DIR / self.factors_filename

    def rise_rank_filename(self) -> str:
        """CSV name for the RISE rank sweep."""
        return f"rise_rank_{self.key}.csv"

    def cpd_rank_filename(self) -> str:
        """CSV name for the CP rank sweep.

        The method and the RISE rank the sweep was conditioned on are recorded
        in the sidecar rather than in the filename, so a re-run at different
        settings replaces the artifact instead of accumulating variants.
        """
        return f"cpd_rank_{self.key}.csv"


DATASETS: dict[str, DatasetSpec] = {
    "alad": DatasetSpec(
        key="alad",
        label="ALAD Lung Transplant BAL",
        loader=import_alad,
        condition_column="dsco_id",
        factors_filename="bal_alad_factors.nc",
        source_file=Path("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"),
        celltype_column="broad_cell_type",
        group_column="ALADstatus",
    ),
    "ibd": DatasetSpec(
        key="ibd",
        label="IBDverse Terminal Ileum",
        loader=import_ibdverse,
        condition_column="sanger_sample_id",
        factors_filename="ibd_ti_factors.nc",
        source_file=IBD_TI_IVCSC,
        notes=(
            "Cell-type and disease-status `.obs` columns are not yet pinned "
            "down for this dataset; fill them in once the IBDverse annotation "
            "columns are settled."
        ),
    ),
}

DATASET_KEYS = tuple(DATASETS)
