"""
Dataset registry shared by every runner.

A runner is told *which* dataset to work on by a short label (``alad``,
``ibd``); everything else it needs -- the loader, the raw source file, the
column of `.obs` that defines a condition, and the canonical output filenames
-- is looked up here. That keeps "which dataset?" a one-word CLI argument and
keeps the naming of result files consistent between the pipeline runner, the
rank-selection runners, and the figure modules.
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
    #: Raw ``.h5ad`` the loader reads. Recorded (with its sha256) in every
    #: provenance sidecar -- this is the field that would have caught the ALAD
    #: source drift in issue #164.
    source_file: Path | None = None
    gene_threshold: float = 0.001
    celltype_column: str | None = None
    group_column: str | None = None
    notes: str = ""

    @property
    def loader_path(self) -> str:
        """``module:function`` name of the loader, in the form issue #164 documents.

        Falls back gracefully for loaders that aren't plain functions (a lambda
        or partial in a test fixture, say).
        """
        module = getattr(self.loader, "__module__", "?")
        name = getattr(self.loader, "__qualname__", repr(self.loader))
        return f"{module}:{name}"

    @property
    def loader_args(self) -> dict[str, Any]:
        """The keyword arguments `load` passes, recorded in the sidecar."""
        args: dict[str, Any] = {
            "gene_threshold": self.gene_threshold,
            "normalize": True,
        }
        if self.source_file is not None:
            args["filename"] = str(self.source_file)
        return args

    def load(self, gene_threshold: float | None = None) -> anndata.AnnData:
        """Load the dataset, normalized, at the given (or default) gene threshold."""
        threshold = self.gene_threshold if gene_threshold is None else gene_threshold
        kwargs: dict[str, Any] = {"gene_threshold": threshold, "normalize": True}
        if self.source_file is not None:
            kwargs["filename"] = str(self.source_file)
        return self.loader(**kwargs)

    def factors_path(self, output_dir: Path | None = None) -> Path:
        return (output_dir or DATA_DIR) / self.factors_filename

    def rise_rank_filename(self) -> str:
        """Canonical CSV name for the RISE BiCV sweep (matches figureRISErank)."""
        return f"rise_rank_bicv_{self.key}.csv"

    def cpd_rank_filename(self, rise_rank: int) -> str:
        """Canonical CSV name for the CPD imputation sweep (matches figureCPDrank)."""
        return f"cpd_rank_imputation_{self.key}_rise{rise_rank}.csv"


DATASETS: dict[str, DatasetSpec] = {
    "alad": DatasetSpec(
        key="alad",
        label="ALAD Lung Transplant BAL",
        loader=import_alad,
        condition_column="dsco_id",
        factors_filename="bal_alad_factors.nc",
        source_file=Path("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"),
        gene_threshold=0.001,
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
        gene_threshold=0.001,
        notes=(
            "Cell-type and disease-status `.obs` columns are not yet pinned down "
            "for this dataset; fill them in here once the IBDverse annotation "
            "columns are settled."
        ),
    ),
}

DATASET_KEYS = tuple(DATASETS)


def get_dataset(key: str) -> DatasetSpec:
    """Look up a dataset spec by label, with a helpful error for typos."""
    try:
        return DATASETS[key.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {key!r}. Available: {', '.join(DATASET_KEYS)}."
        ) from None
