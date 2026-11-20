"""Number of donors, terminal ileum biopsies, and cells per inflammation
status, shown in the IBD cohort schematic."""

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS
from ... import common
from .. import _common as S

SCHEMATIC = True  # an illustration, but its sample sizes are source data
AXES = ("full",)


def plot(axs):
    (ax,) = axs
    samples, counts = S.samples(), S.celltype_counts()
    groups = samples[S.SPEC.group_column]
    sizes = (
        samples.groupby(S.SPEC.group_column)
        .agg(
            donors=("individual_id", "nunique"),
            biopsies=("n_cells", "size"),
            cells=("n_cells", "sum"),
        )
        .reindex(list(S.GROUP_LABELS))
        .rename(index=S.GROUP_LABELS)
    )
    common.source_data(
        "cohort_sample_sizes",
        sizes.rename_axis("Inflammation status"),
        "Number of donors, terminal ileum biopsies and cells (after capping each "
        "biopsy at 2,000 cells and excluding the mesenchymal lineage) per "
        "inflammation status.",
    )
    n_lr = load_factors_netcdf(DATASETS[S.DATASET].factors_path()).sizes["lr_pair"]
    n = sizes["biopsies"]
    common.placeholder(
        ax,
        f"IBDverse terminal ileum: {len(samples)} biopsies, one per donor\n"
        f"({n['Healthy']} healthy, {(groups != 'Healthy').sum()} Crohn's disease: "
        f"{n['CD uninflamed']} uninflamed, {n['CD inflamed']} inflamed)\n"
        f"{sizes['cells'].sum():,} cells, {counts['celltype'].nunique()} cell types "
        f"in {counts['lineage'].nunique()} lineages, {n_lr:,} LR pairs",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
