"""
Dataset registry shared by every runner.

A runner is told which dataset to work on by a short label (``alad``, ``ibd``).
The loader, the source file, the condition column, the canonical output
filenames and the selected ranks are looked up here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata

from ..import_data import IBD_FULL_COHORT_IVCSC, import_alad, import_ibdverse

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    loader: Callable[..., anndata.AnnData]
    condition_column: str
    factors_filename: str
    #: Source file the loader reads
    source_file: Path
    gene_threshold: float = 0.001
    celltype_column: str | None = None
    group_column: str | None = None
    rise_rank: int | None = None
    cp_rank: int | None = None
    rank_notes: str = ""
    notes: str = ""

    @property
    def loader_args(self) -> dict[str, Any]:
        """The keyword arguments `load` passes."""
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
        return f"rise_rank_{self.key}.csv"

    def cpd_rank_filename(self) -> str:
        return f"cpd_rank_{self.key}.csv"

    def stability_filename(self) -> str:
        return f"component_stability_{self.key}.csv"

    def anova_bootstrap_filename(self) -> str:
        return f"anova_bootstrap_{self.key}.csv"

    def significance_lr_filename(self) -> str:
        return f"significance_lr_{self.key}.csv"

    def significance_differential_filename(self) -> str:
        return f"significance_differential_{self.key}.parquet"


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
        rise_rank=19,
        cp_rank=3,
        rank_notes=(
            "RISE 19 is the argmax of the 10-repeat BiCV sweep. Ranks 14-22 "
            "sit within one SEM of it, so it is the best-supported point in a "
            "flat interval rather than a resolved optimum. CP 3 is the last "
            "rank before CORCONDIA flips sign (97.2% at 3, -238% at 4) in the "
            "sweep conditioned on RISE 19."
        ),
    ),
    "ibd": DatasetSpec(
        key="ibd",
        label="IBDverse Crohn's Disease Terminal Ileum",
        loader=import_ibdverse,
        condition_column="sanger_sample_id",
        factors_filename="ibd_ti_factors.nc",
        source_file=IBD_FULL_COHORT_IVCSC,
        celltype_column="broad_cell_type",
        group_column="inflammation_status",
        notes=(
            "Krzak et al. Nat Genet 2026 terminal-ileum cohort: 343 samples, "
            "232 healthy and 111 Crohn's (64 inflamed / 47 uninflamed). "
            "`celltype` holds 57 annotated types matching the published DGE "
            "tables; `broad_cell_type` holds 8 lineages."
        ),
    ),
}

DATASET_KEYS = tuple(DATASETS)
