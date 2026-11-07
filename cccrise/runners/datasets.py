"""
Dataset registry shared by every runner.

A runner is told which dataset to work on by a short label (``alad``, ``ibd``);
the loader, the source file, the condition column and the canonical output
filenames are looked up here.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import anndata

from ..import_data import (
    IBD_EXCLUDED_BROAD_CELL_TYPES,
    IBD_FULL_COHORT_IVCSC,
    import_alad,
    import_ibdverse,
)

DATA_DIR = Path(__file__).parent.parent / "data"
#: Core factorizations written by `run_ccc_rise`.
FACTORS_DIR = DATA_DIR / "factors"
#: RISE and CP rank-selection sweeps (`run_rise_rank`, `run_cpd_rank`).
RANK_SELECTION_DIR = DATA_DIR / "rank_selection_results"
#: Validation experiments: stability, significance, NicheNet, and the
#: alternate communication-score comparison (`ALT_SCORE_METHODS_DIR`).
VALIDATION_DIR = DATA_DIR / "validation_results"
#: One subdirectory per communication score (`comparison_output_dir`).
ALT_SCORE_METHODS_DIR = VALIDATION_DIR / "alt_score_methods"
#: Annotation-pooled comparators (`run_comparators`): each method's rank
#: selection and fit, per dataset and labelling (`comparator_filename`).
COMPARATORS_DIR = DATA_DIR / "comparators"
COMPARATOR_RANKS_DIR = COMPARATORS_DIR / "rank_selection"
COMPARATOR_FITS_DIR = COMPARATORS_DIR / "fits"
COMPARATOR_SCORES_DIR = COMPARATORS_DIR / "scores"
#: Figure 4's IBD program tables (`run_ibd_programs`)
IBD_PROGRAMS_DIR = DATA_DIR / "ibd_programs"
#: Upstream NICHES against cccRISE's populations (`run_niches`)
NICHES_DIR = DATA_DIR / "niches"
#: Figure 2's and Figure 3's ALAD program tables (`run_alad_programs`)
ALAD_PROGRAMS_DIR = DATA_DIR / "alad_programs"


def comparator_filename(method: str, what: str, dataset: str, labelling: str) -> str:
    """A comparator table's name, ``<method>_<what>_<dataset>_<labelling>
    .parquet`` (e.g. ``tc2c_elbow_alad_coarse.parquet``); `what` ``"fit"``
    names the fit, ``<method>_<dataset>_<labelling>.nc``, and ``"starts"``
    every start's factors, ``<method>_starts_<dataset>_<labelling>.nc``."""
    if what == "fit":
        return f"{method}_{dataset}_{labelling}.nc"
    suffix = "nc" if what == "starts" else "parquet"
    return f"{method}_{what}_{dataset}_{labelling}.{suffix}"


def scores_filename(what: str, dataset: str) -> str:
    """A comparator scores table's name, ``<what>_<dataset>.parquet`` (e.g.
    ``containment_alad.parquet``): one table per analysis, every method and
    labelling in it."""
    return f"{what}_{dataset}.parquet"


#: `communication_score` value every runner defaults to; its results are the
#: ones manuscript figures and interpretation read, so they stay in their
#: analysis's directory instead of the per-method subdirectory
#: `scoring_output_dir` uses for the rest.
DEFAULT_COMMUNICATION_SCORE = "expression_product"
#: Every `communication_score` a runner accepts; see `ccc.combine_communication_score`.
COMMUNICATION_SCORES = (
    "expression_product",
    "expression_mean",
    "expression_gmean",
    "expression_hill",
)


def scoring_output_dir(communication_score: str, default_dir: Path) -> Path:
    """Where a runner should default its output for `communication_score`.

    The default score's results are canonical and live in `default_dir`
    (the runner's analysis directory, e.g. `FACTORS_DIR`); every other
    score's results land in its `comparison_output_dir`, so a scoring-method
    sweep can't collide with the canonical files or with each other.
    """
    if communication_score == DEFAULT_COMMUNICATION_SCORE:
        return default_dir
    return comparison_output_dir(communication_score)


def comparison_output_dir(communication_score: str) -> Path:
    """Where `run_alt_comm_score` writes one method's results.

    Unlike `scoring_output_dir`, this never collapses to the canonical
    directory for the default score: comparing scoring methods fairly means
    running all of them -- default included -- on one shared RISE fit, and
    that copy of the default's results must not collide with its canonical
    copy that manuscript figures read.
    """
    return ALT_SCORE_METHODS_DIR / communication_score


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    #: The cohort's name; `label` adds the cell cap, if any.
    cohort_label: str
    loader: Callable[..., anndata.AnnData]
    condition_column: str
    celltype_column: str
    factors_filename: str
    #: Source file the loader reads
    source_file: Path
    gene_threshold: float = 0.001
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
    #: auto-detects, which picks a GPU whenever one is present.
    backend: str | None = None
    notes: str = ""
    #: Per-condition cell cap passed to the loader (see
    #: `import_data.cap_cells_per_condition`). None loads every cell.
    max_cells_per_condition: int | None = None
    cell_cap_seed: int = 0
    #: Appended to `key` in every output filename (see `output_key`);
    #: `with_cell_cap` sets it for a cap other than the registry's.
    name_suffix: str = ""
    #: Broad lineages the loader removes before anything else (see
    #: `import_data.IBD_EXCLUDED_BROAD_CELL_TYPES`). Empty keeps every lineage.
    excluded_broad_cell_types: tuple[str, ...] = ()
    #: The finer annotation, beside `celltype_column`'s coarse one; together
    #: they are the labellings the comparators pool cells by (`labellings`).
    fine_celltype_column: str | None = None

    @property
    def loader_args(self) -> dict[str, Any]:
        """The keyword arguments `load` passes."""
        args: dict[str, Any] = {
            "filename": str(self.source_file),
            "gene_threshold": self.gene_threshold,
            "normalize": True,
        }
        if self.max_cells_per_condition is not None:
            args["max_cells_per_condition"] = self.max_cells_per_condition
            args["cell_cap_seed"] = self.cell_cap_seed
        if self.excluded_broad_cell_types:
            args["excluded_broad_cell_types"] = list(self.excluded_broad_cell_types)
        return args

    @property
    def labellings(self) -> dict[str, str]:
        """Cell annotation column of each labelling: ``coarse`` and, when the
        dataset has one, ``fine``."""
        fine = {"fine": self.fine_celltype_column} if self.fine_celltype_column else {}
        return {"coarse": self.celltype_column, **fine}

    @property
    def label(self) -> str:
        """`cohort_label`, with the cell cap when there is one."""
        if self.max_cells_per_condition is None:
            return self.cohort_label
        return (
            f"{self.cohort_label} (<= {self.max_cells_per_condition} cells per "
            f"condition, seed {self.cell_cap_seed})"
        )

    @property
    def output_key(self) -> str:
        """`key` plus `name_suffix`. Every output filename uses it, so a run at
        a cap other than the registry's never overwrites the registry
        dataset's results; `key` itself stays the registry key, for lookups
        such as `_cli.DEFAULT_GROUP_LABELS`."""
        return self.key + self.name_suffix

    def with_cell_cap(self, max_cells: int, seed: int = 0) -> "DatasetSpec":
        """This dataset capped at `max_cells` cells per condition, its outputs
        suffixed ``_cap<N>`` (and ``s<seed>`` for a non-zero seed) unless that
        is already its cap."""
        if "max_cells_per_condition" not in inspect.signature(self.loader).parameters:
            raise ValueError(
                f"Dataset {self.key!r}'s loader does not support a cell cap."
            )
        if (max_cells, seed) == (self.max_cells_per_condition, self.cell_cap_seed):
            return self
        suffix = f"_cap{max_cells}" + (f"s{seed}" if seed else "")
        return replace(
            self,
            max_cells_per_condition=max_cells,
            cell_cap_seed=seed,
            name_suffix=suffix,
            factors_filename=self.factors_filename.replace(
                f"{self.name_suffix}.nc", f"{suffix}.nc"
            ),
        )

    def load(self) -> anndata.AnnData:
        """Load the dataset, normalized."""
        return self.loader(**self.loader_args)

    def factors_path(self) -> Path:
        return FACTORS_DIR / self.factors_filename

    def rise_rank_filename(self) -> str:
        return f"rise_rank_{self.output_key}.parquet"

    def cpd_rank_filename(self) -> str:
        return f"cpd_rank_{self.output_key}.parquet"

    def cpd_imputation_filename(self) -> str:
        return f"cpd_imputation_{self.output_key}.parquet"

    def cpd_start_fms_filename(self) -> str:
        return f"cpd_start_fms_{self.output_key}.parquet"

    def cpd_ridge_filename(self) -> str:
        return f"cpd_ridge_{self.output_key}.parquet"

    def stability_filename(self) -> str:
        return f"component_stability_{self.output_key}.parquet"

    def anova_point_estimate_filename(self) -> str:
        return f"anova_point_estimate_{self.output_key}.parquet"

    def significance_lr_filename(self) -> str:
        return f"significance_lr_{self.output_key}.parquet"

    def significance_entries_filename(self) -> str:
        return f"significance_entries_{self.output_key}.nc"

    def nichenet_scores_filename(self) -> str:
        return f"nichenet_scores_{self.output_key}.parquet"

    def nichenet_cell_null_filename(self) -> str:
        return f"nichenet_cell_null_{self.output_key}.parquet"


#: Per-sample cell cap of the registry's IBD cohort, whose results every IBD
#: figure and the app read; the uncapped cohort is too large to fit whole.
IBD_CELL_CAP = 2000

DATASETS: dict[str, DatasetSpec] = {
    "alad": DatasetSpec(
        key="alad",
        cohort_label="ALAD Lung Transplant BAL",
        loader=import_alad,
        condition_column="dsco_id",
        factors_filename="bal_alad_factors.nc",
        source_file=Path("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"),
        celltype_column="broad_cell_type",
        fine_celltype_column="celltype",
        group_column="ALADstatus",
        patient_column="patient_id",
        outcome_column="6monthcondition",
        backend="cupy",
    ),
    "ibd": DatasetSpec(
        key="ibd",
        cohort_label="IBDverse Crohn's Disease Terminal Ileum",
        loader=import_ibdverse,
        condition_column="sanger_sample_id",
        factors_filename="ibd_ti_factors.nc",
        source_file=IBD_FULL_COHORT_IVCSC,
        celltype_column="broad_cell_type",
        fine_celltype_column="celltype",
        group_column="inflammation_status",
        # Too large for the GPU: auto-detect would pick CuPy, whose managed-memory
        # products mismatched SciPy's on the full matrix. MKL speeds up the CPU
        # products when `MKL_RT` points at libmkl_rt.
        backend="cpu",
        max_cells_per_condition=IBD_CELL_CAP,
        excluded_broad_cell_types=IBD_EXCLUDED_BROAD_CELL_TYPES,
        notes=(
            "Krzak et al. Nat Genet 2026 terminal-ileum cohort: 343 samples, "
            "232 healthy and 111 Crohn's (64 inflamed / 47 uninflamed). "
            "`celltype` holds 57 annotated types matching the published DGE "
            "tables; `broad_cell_type` holds 8 lineages, of which the "
            "mesenchymal one (fibroblasts, endothelial cells, pericytes) is "
            "excluded as a biopsy-sampling artifact "
            "(`import_data.IBD_EXCLUDED_BROAD_CELL_TYPES`)."
        ),
    ),
}

DATASET_KEYS = tuple(DATASETS)
