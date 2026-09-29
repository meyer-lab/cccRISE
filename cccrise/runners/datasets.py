"""
Dataset registry shared by every runner.

A runner is told which dataset to work on by a short label (``alad``, ``ibd``);
the loader, the source file, the condition column and the canonical output
filenames are looked up here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata

from ..import_data import IBD_FULL_COHORT_IVCSC, import_alad, import_ibdverse

DATA_DIR = Path(__file__).parent.parent / "data"

#: `communication_score` value every runner defaults to; its results are the
#: ones manuscript figures and interpretation read, so they stay in `DATA_DIR`
#: instead of the per-method subdirectory `scoring_output_dir` uses for the rest.
DEFAULT_COMMUNICATION_SCORE = "expression_product"
#: Every `communication_score` a runner accepts; see `ccc.combine_communication_score`.
COMMUNICATION_SCORES = (
    "expression_product",
    "expression_mean",
    "expression_gmean",
    "expression_hill",
)

ALT_SCORE_METHODS_DIR = "alt_score_methods"


def scoring_output_dir(communication_score: str) -> Path:
    """Where a runner should default its output for `communication_score`.

    The default score's results are canonical and live in `DATA_DIR`; every
    other score's results land in their own subdirectory under
    `DATA_DIR/alt_score_methods/` so a scoring-method sweep can't collide with
    the canonical files or with each other.
    """
    if communication_score == DEFAULT_COMMUNICATION_SCORE:
        return DATA_DIR
    return DATA_DIR / ALT_SCORE_METHODS_DIR / communication_score


#: The one provenance file `run_alt_comm_score` leaves in each
#: `comparison_output_dir`, consolidated from its outputs' sidecars.
COMPARISON_PROVENANCE_FILENAME = "provenance.yaml"


def comparison_output_dir(communication_score: str) -> Path:
    """Where `run_alt_comm_score` writes one method's results.

    Unlike `scoring_output_dir`, this never collapses to `DATA_DIR` for the
    default score: comparing scoring methods fairly means running all of
    them -- default included -- at the same (comparison-scale) settings, and
    that copy of the default's results must not collide with its
    comprehensive, canonical copy that `scoring_output_dir` resolves to
    `DATA_DIR` for and that manuscript figures read.
    """
    return DATA_DIR / ALT_SCORE_METHODS_DIR / communication_score


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
    #: Condition-level patient identity, for conditions the same patient
    #: contributes more than one sample of (see `validation`'s
    #: `patient_column`). None where `condition_column` already identifies
    #: the patient one-to-one.
    patient_column: str | None = None
    #: Condition-level outcome the ANOVA runners test every level of by
    #: default. None falls back to `group_column` and its registered labels.
    outcome_column: str | None = None
    #: RISE backend every runner fits on ('cpu', 'cupy', 'mlx'); None
    #: auto-detects, for cohorts too large for the GPU.
    backend: str | None = None
    #: Storage precision of per-cell arrays in written factors files
    #: (`factors.save_ccc_rise_fit`); "float16" halves them for large
    #: cohorts, at the cost of an only-approximate RISE reference.
    cell_dtype: str = "float32"
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

    def anova_point_estimate_filename(self) -> str:
        return f"anova_point_estimate_{self.key}.csv"

    def significance_lr_filename(self) -> str:
        return f"significance_lr_{self.key}.csv"

    def significance_differential_filename(self) -> str:
        return f"significance_differential_{self.key}.parquet"

    def cell_obs_columns(self) -> list[str]:
        """`.obs` columns to persist per-cell in the factors file (for the
        Shiny explorer's per-cell projections)."""
        return [
            col
            for col in (
                "celltype",
                self.celltype_column,
                self.condition_column,
                self.group_column,
            )
            if col
        ]

    def nichenet_scores_filename(self) -> str:
        return f"nichenet_scores_{self.key}.csv"


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
        patient_column="patient_id",
        outcome_column="6monthcondition",
        backend="cupy",
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
