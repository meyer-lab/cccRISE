"""
Visual check that the EMBO source data reproduce the figures (#227): a sample
of panels, every figure's, redrawn from the zips `cccrise-sourcedata` wrote
instead of from the analyses, one PNG per panel letter, to compare by eye
with the built figures:

    uv run cccrise-fbuild <figure>      # each figure, then
    uv run cccrise-sourcedata           # its zips, then
    uv run python -m cccrise.tests.source_data_replay             # every figure
    uv run python -m cccrise.tests.source_data_replay "Figure 3"  # some

writes `output/source_data_replay/<figure label>/<panel>.png`. Not collected
by pytest (no `test_` prefix), so it never runs with the test suite or CI.

Each panel has a data loading section, which reads its CSVs, and a
visualization section, which draws them: mostly the panel's own drawing code
with its data loaders patched to return the CSVs, so the replay draws exactly
as the figure does, else (panels whose loaders return more than the source
data holds) just the marks carrying data, at minimal styling. While a panel
draws, the analysis loaders raise, so nothing is drawn from the analyses.
Elements drawn from inputs that aren't source data (e.g. a fit's chosen rank)
are left out where noted.
"""

import ast
import contextlib
import re
import sys
import zipfile
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cccrise import factors
from cccrise.figures import common, source_data, style
from cccrise.figures.figure2 import _common as F2
from cccrise.figures.figure2.figureS_alad_cohort import _stability
from cccrise.figures.figure2.figureS_cmp5_identity import _common as V
from cccrise.figures.figure3 import _common as F3
from cccrise.figures.figure3 import (
    coordinated_changes,
    lr_permutation,
    lr_volcano,
    sender_receiver_scatter,
)
from cccrise.figures.figure3.figureS_alad_associations import (
    _anova_stability,
)
from cccrise.figures.figure3.figureS_ccc_scoring import _common as CCC
from cccrise.figures.figure3.figureS_ccc_scoring import interpretation
from cccrise.figures.figure4 import _common as F4
from cccrise.figures.figure4 import (
    hormone_angle_hist,
    hormone_angle_scatter,
)
from cccrise.figures.figure4.figureS_ibd_associations import (
    status_loadings,
)
from cccrise.figures.figure5 import (
    _coarse_fine,
    _depth,
    _missed,
    _niches,
    _recovery,
    span_containment,
)
from cccrise.figures.figure5.figureS_comparator_rank_selection import (
    _panels as comparator_ranks,
)
from cccrise.runners.datasets import DATASETS

ZIP_DIR = source_data.OUTPUT_DIR
OUT_DIR = common.OUTPUT_DIR / "source_data_replay"
FIGURES = "cccrise.figures"
#: The analysis loaders, which raise while a panel draws unless patched
ANALYSIS_LOADERS = (
    (common, "load_runner_output"),
    (common, "load_factors_to_anndata"),
    (common, "get_cell_projections"),
    (common, "alad_programs"),
    (common, "load_cpd_rank"),
    (common, "load_cpd_imputation"),
    (common, "load_minima"),
    (factors, "load_factors_netcdf"),
)


class Zip:
    """A figure's source data zip, its CSVs read as the analyses' tables:
    CCC-RISE named cccRISE again (`common.source_data` renames it)."""

    def __init__(self, label: str):
        self.zip = zipfile.ZipFile(ZIP_DIR / f"{label}.zip")

    def __call__(self, folder: str, name: str) -> pd.DataFrame:
        df = pd.read_csv(self.zip.open(f"{folder}/{name}.csv"))
        for c in df.select_dtypes(["object", "str"]):
            df[c] = df[c].replace("CCC-RISE", "cccRISE")
        return df


# =============================================================================
# Drawing
# =============================================================================


def module(name: str, patches: dict, panels: dict[str, int | None]) -> dict:
    """Panel module `name` (under `cccrise.figures`) drawn alone with the
    `patches` ({(object, attribute): value}) in place, as {panel: (figure,
    its axes index, or None for the whole module)} for `panels`."""
    with analysis_blocked(patches):
        fig = common.standalone_fixed(f"{FIGURES}.{name}")
    return {p: (fig, i) for p, i in panels.items()}


def axes(kind: str = "square", n: int = 1) -> tuple[plt.Figure, list[plt.Axes]]:
    """A figure of `n` side-by-side `style.AXIS_SIZE[kind]` axes, with room
    for labels."""
    w, h = style.AXIS_SIZE[kind]
    margin, gap = 0.7, 0.5
    fig = plt.figure(figsize=(n * w + (n - 1) * gap + 2 * margin, h + 2 * margin))
    width, height = fig.get_size_inches()
    return fig, [
        fig.add_axes(
            (
                (margin + i * (w + gap)) / width,
                margin / height,
                w / width,
                h / height,
            )
        )
        for i in range(n)
    ]


@contextlib.contextmanager
def analysis_blocked(patches: dict):
    def blocked(name):
        def load(*_, **__):
            raise RuntimeError(f"{name} read analysis data; patch it to the CSVs")

        return load

    with contextlib.ExitStack() as stack:
        for obj, attr in ANALYSIS_LOADERS:
            stack.enter_context(patch.object(obj, attr, blocked(attr)))
        for (obj, attr), value in patches.items():
            stack.enter_context(patch.object(obj, attr, value))
        yield


def manual(draw: Callable, kind: str = "square", n: int = 1) -> tuple:
    """A figure of fresh axes drawn by `draw(axs)`, analysis blocked."""
    fig, axs = axes(kind, n)
    with analysis_blocked({}):
        draw(axs)
    return fig, None


def save(path, fig: plt.Figure, index: int | None) -> None:
    """`fig`, or only its `index`th axes (with what it owns: labels,
    colorbars inset in it)."""
    if index is None:
        fig.savefig(path, bbox_inches="tight", dpi=300)
        return
    ax = fig.axes[index]
    box = ax.get_tightbbox()
    assert box is not None  # a drawn axes has extents
    box = box.padded(4)
    fig.savefig(
        path, bbox_inches=box.transformed(fig.dpi_scale_trans.inverted()), dpi=300
    )


def seeds(table: pd.DataFrame) -> pd.DataFrame:
    """A minima table's `Seeds` as tuples again."""
    return table.assign(Seeds=table["Seeds"].map(ast.literal_eval))


def heatmap(ax, values: pd.DataFrame, xlabel: str) -> None:
    image = common.plot_factor_heatmap(ax, values.to_numpy(dtype=float), xlabel)
    n = values.shape[1]
    step = 1 if n <= 8 else 2 if n <= 20 else 10  # 17 or 60 crowd the axis
    ax.set_xticks(range(0, n, step), range(1, n + 1, step))
    ax.set_yticks([])
    common.colorbar(ax, image, "Loading / max |loading|")


def pair_heatmap(ax, t: pd.DataFrame) -> None:
    """A `_split.pair_heatmap` as a plain heatmap: each column's carrying
    pairs at their exported |loading| (blank where a column doesn't carry the
    pair), columns in exported order."""
    t = t.assign(
        Name=t["Method"].str.replace("cccRISE", "CCC-RISE")
        + " "
        + t["Labelling"].fillna("").str[:1].str.upper()
        + " "
        + t["Column"].astype(str)
    )
    value = t.columns[4]
    grid = t.pivot_table(index="LR pair", columns="Name", values=value, sort=False)
    image = ax.imshow(
        grid.to_numpy(),
        cmap=style.CMAPS["weight"],
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_yticks(range(len(grid)), grid.index, fontsize=style.FONT_SIZE_DENSE - 1)
    ax.set_xticks(
        range(grid.shape[1]), grid.columns, rotation=90, fontsize=style.FONT_SIZE_DENSE
    )
    common.colorbar(ax, image, value)


# =============================================================================
# Panels: {figure label: {panel: draw(zip) -> {panel: (figure, axes index)}}}
# =============================================================================


def figure2(z: Zip) -> dict:
    cells = z("2L,N", "cell_weights_groups")
    k = F2.GRANULARITY_COMPONENT
    sender, receiver = (
        cells[f"Cmp {k} {r} weight"].to_numpy() for r in ("sender", "receiver")
    )
    cell_patches = {
        (F2, "component_weights"): lambda: (sender, receiver),
        (F2, "cell_table"): lambda: cells,
    }
    celltypes = SimpleNamespace(
        obs=pd.DataFrame(
            {DATASETS[F2.GRANULARITY_DATASET].celltype_column: cells["Cell type"]}
        )
    )
    receivers = z("2G", "cpd_receiver_eigen-state_loadings").set_index(
        "Receiver eigen-state"
    )

    def receiver_matrix(axs):
        heatmap(axs[0], receivers, "CPD component")
        axs[0].set_ylabel("Receiver eigen-state")

    return (
        {
            "2D": manual(
                lambda axs: common.plot_minima_table(
                    axs[0],
                    seeds(z("2D", "cpd_minima_by_rank")),
                    common.selected_ranks("alad")[1],
                )
            )
        }
        | {"2G": manual(receiver_matrix, "slim")}
        | module("figure2.sender_receiver_strata", cell_patches, {"2L": None})
        | module(
            "figure2.celltype_composition",
            cell_patches
            | {
                (common, "get_cell_projections"): lambda dataset: celltypes,
                (F2, "cell_groups"): lambda: cells["Group"],
            },
            {"2N": None},
        )
    )


def figure3(z: Zip) -> dict:
    # 3E, colored on the lymphocyte blow-up's scale (3C-D's data)
    pacmap = z("3C-D", "pacmap_celltype_eigenstate")
    e = F3.EIGENSTATE
    limit = float(
        np.percentile(
            np.abs(pacmap.loc[pacmap["In blow-up"], f"Eigen-state {e} projection"]),
            common.SATURATION_PERCENTILE,
        )
    )
    cells = z("3E", f"weight_component{F3.COMPONENT}_cells")
    n, k = len(cells), F3.COMPONENT
    projections = np.zeros((n, e))
    projections[:, e - 1] = cells[f"Eigen-state {e} projection"]
    weights = {"sc_B": np.zeros((n, k)), "rc_C": np.zeros((n, k))}
    weights["sc_B"][:, k - 1] = cells["Sender weight"]
    weights["rc_C"][:, k - 1] = cells["Receiver weight"]
    adata = SimpleNamespace(
        obsm={**weights, "projections": projections},
        obs_names=cells["Cell"],
        obs={F3.SPEC.condition_column: cells["Sample"]},
    )
    # 3H, its co-variation panel (G, the association histogram, beside it)
    anova = z("3G", "lr_pair_outcome_anova")
    top_column = f"Cmp {k} top {coordinated_changes.N_TOP}"
    baselines = z("3H", "baseline_rank_correlations")
    labels = {label: key for key, (label, _) in coordinated_changes.BASELINES.items()}
    result = {
        "lr_eta": anova["EtaSquared"].to_numpy(),
        "component_eta": float(z("3A", "cpd_outcome_anova")["EtaSquared"].iloc[0]),
        "top": np.flatnonzero(anova[top_column]),
        "observed": z("3H", "top_pair_rank_correlations")[
            "Spearman correlation"
        ].to_numpy(),
        "baselines": {
            labels[str(b)]: g.pivot(
                index="Draw", columns="Pair of pairs", values="Spearman correlation"
            ).to_numpy()
            for b, g in baselines.groupby("Baseline", sort=False)
        },
        "tests": z("3H", "covariation_tests"),
    }
    # 3I-J: every component's top pairs' tests, in loading order
    tests = z("3I-J", "top_pair_permutation_tests").sort_values(
        ["Component", "Loading rank"]
    )
    ratios = z("3J", "top_pair_log2_ratios")
    case, control = (
        f"{lr_permutation.CASE_DISPLAY} mean",
        f"{lr_permutation.CONTROL_DISPLAY} mean",
    )
    analysis = {
        int(g["Component"].iloc[0]): {
            "observed": g["Observed difference"].to_numpy(),
            "p": g["Permutation p"].to_numpy(),
            "order": np.arange(len(g)),
            "means": (g[case].to_numpy(), g[control].to_numpy()),
            "log2": ratios.loc[
                ratios["Component"] == g["Component"].iloc[0], "log2 mean ratio"
            ].to_numpy(),
            "names": g["LR pair"].to_numpy(),
        }
        for _, g in tests.groupby("Component")
    }
    null = z("3I", "top_pair_permutation_null").pivot(
        index="Permutation", columns="LR pair", values="Null difference"
    )
    component = analysis[k] | {
        "null": null[analysis[k]["names"][: lr_permutation.N_TOP]].to_numpy()
    }

    def nulls(axs):
        lr_permutation.plot_pair_nulls(component, lr_permutation.N_TOP, axs[0])

    return (
        module(
            "figure3.sender_receiver_scatter",
            {
                (common, "get_cell_projections"): lambda dataset: adata,
                (sender_receiver_scatter, "projection_limit"): lambda: limit,
            },
            {"3E": None},
        )
        | module(
            "figure3.coordinated_changes",
            {
                (coordinated_changes, "coordinated_analysis"): lambda: result,
                (coordinated_changes, "_source_data"): lambda result: None,
            },
            {"3H": 1},
        )
        | with_patches(
            {(lr_permutation, "LR_NAMES"): component["names"]},
            lambda: {"3I": manual(nulls)},
        )
        | module(
            "figure3.lr_volcano",
            {
                (lr_volcano, "permutation_analysis"): lambda: analysis,
                (lr_volcano, "save_permutation_tests"): lambda result: None,
                (lr_volcano, "mean_log2_ratio"): lambda r: r["log2"],
                (lr_volcano, "LR_NAMES"): analysis[k]["names"],
            },
            {"3J": None},
        )
    )


def with_patches(patches: dict, draw: Callable):
    with contextlib.ExitStack() as stack:
        for (obj, attr), value in patches.items():
            stack.enter_context(patch.object(obj, attr, value))
        return draw()


def figure4(z: Zip) -> dict:
    lr = z("4D", "cpd_lr_pair_loadings").set_index("LR pair")
    shown = lr.loc[lr.pop("Shown")].to_numpy(dtype=float)

    def lr_matrix(axs):
        heatmap(
            axs[0], pd.DataFrame(shown[common.dominant_order(shown)]), "CPD component"
        )
        axs[0].set_ylabel(f"LR pairs (n={len(shown)})")

    angles = z("4I-J", "hormone_angle_cells")
    expression = z("4K", "hormone_extreme_marker_expression_per_cell")
    return (
        {"4D": manual(lr_matrix)}
        | module(
            "figure4.hormone_angle_scatter",
            {(hormone_angle_scatter, "angle_cells"): lambda: angles},
            {"4I": None},
        )
        | module(
            "figure4.hormone_angle_hist",
            {(hormone_angle_hist, "angle_cells"): lambda: angles},
            {"4J": None},
        )
        | module(
            "figure4.hormone_extreme_markers",
            {
                (F4, "programs"): lambda table: expression.drop(columns="Class"),
                (F4, "hormone_extremes"): lambda cells: expression["Class"],
            },
            {"4K": None},
        )
    )


def figure5(z: Zip) -> dict:
    containment = z("5B", "component_containment_in_comparator_span").rename(
        columns={"Fit method": "In"}
    )
    cos2 = z("5E", "lr_cos2_between_fits_alad")

    def flows(axs):
        """Each block's left fit's columns over its right fit's, joined where
        cos^2 >= `FLOW_FROM`, line width growing with cos^2 (columns in
        exported order, not `_coarse_fine`'s best-match placement)."""
        ax = axs[0]
        for i, (_, t) in enumerate(cos2.groupby(["From fit", "To fit"], sort=False)):
            y = -i * 2
            left, right = t["From fit"].iloc[0], t["To fit"].iloc[0]
            tops, bottoms = (
                list(dict.fromkeys(t[c])) for c in ("From column", "To column")
            )
            x = {c: (j + 0.5) / len(tops) for j, c in enumerate(tops)}
            xb = {c: (j + 0.5) / len(bottoms) for j, c in enumerate(bottoms)}
            for r in t[t["LR cos2"] >= _coarse_fine.FLOW_FROM].itertuples():
                ax.plot(
                    [x[r[2]], xb[r[4]]], [y, y - 1], color="0.5", lw=3 * r[5], zorder=1
                )
            for row, at in ((x, y), (xb, y - 1)):
                ax.scatter(
                    list(row.values()), [at] * len(row), color="k", s=8, zorder=2
                )
                for name, xi in row.items():
                    ax.text(
                        xi,
                        at + (0.12 if at == y else -0.12),
                        name,
                        ha="center",
                        va="bottom" if at == y else "top",
                        fontsize=style.FONT_SIZE_DENSE - 2,
                    )
            ax.text(
                -0.02, y, left, ha="right", va="center", fontsize=style.FONT_SIZE_DENSE
            )
            ax.text(
                -0.02,
                y - 1,
                right,
                ha="right",
                va="center",
                fontsize=style.FONT_SIZE_DENSE,
            )
        ax.set_axis_off()

    return (
        module(
            "figure5.span_containment",
            {(span_containment, "rows"): lambda _: containment},
            {"5B": None},
        )
        | {
            "5D": manual(
                lambda axs: pair_heatmap(
                    axs[0], z("5D", "component_pairs_in_related_columns_alad")
                ),
                "fig5_columns",
            )
        }
        | {"5E": manual(flows, "fig5_labels")}
    )


def figure_ev1(z: Zip) -> dict:
    loadings = z("EV1A", "cpd_eigenstate_loadings")
    k = F2.GRANULARITY_COMPONENT
    uns = {}
    for mode, column in (("B", "Sender loading (B)"), ("C", "Receiver loading (C)")):
        uns[mode] = np.zeros((len(loadings), k))
        uns[mode][:, k - 1] = loadings[column]
    de = z("EV1C", "de_sender_vs_receiver").set_index("Gene")
    volcano = (
        de.rename(columns={"q (BH)": "q", "Wilcoxon score": "score"}),
        de["Detected fraction difference"],
        de["-log10 q (plotted, floored at q = 1e-30)"],
    )
    return module(
        "figure2.figureS_cmp5_identity.cmp_eigenstate_loadings",
        {(common, "load_factors_to_anndata"): lambda path: SimpleNamespace(uns=uns)},
        {"EV1A": None},
    ) | module(
        "figure2.figureS_cmp5_identity.sender_vs_receiver",
        {
            (V, "volcano_data"): lambda *_: volcano,
            (V, "save_volcano"): lambda *_: None,
        },
        {"EV1C": None},
    )


def figure_ev2(z: Zip) -> dict:
    table = z("EV2B", "anova_component_outcome")
    # Not source data: the test's alpha and outcome, as the figure labels them
    metadata = {"alpha": common.SIG_ALPHA, "group_column": "6monthcondition"}
    patches = {
        (_anova_stability, "replicates"): lambda: z(
            "EV2E-F", "anova_jackknife_replicates"
        ),
        (_anova_stability, "reference"): lambda: z("EV2E-F", "anova_full_cohort"),
        (_anova_stability, "metadata"): lambda: metadata,
    }
    return (
        {
            "EV2B": manual(
                lambda axs: common.plot_association_heatmap(table, axs[0]), "rect"
            )
        }
        | module(
            "figure3.figureS_alad_associations.effect_size_distributions",
            patches,
            {"EV2E": None},
        )
        | module(
            "figure3.figureS_alad_associations.significant_fraction",
            patches,
            {"EV2F": None},
        )
    )


def figure_ev3(z: Zip) -> dict:
    codes = {CCC.SCORE_LABELS[m]: m for m in CCC.ALL_SCORES}
    match = (
        z("EV3A", "factor_match_scores")
        .pivot(index="Component", columns="Method", values="MatchScore")
        .rename(columns=codes)[CCC.ALT_SCORES]
    )

    def match_bars(axs):
        CCC.grouped_bar(axs[0], match, "Match score")
        axs[0].set_ylim(0, 1.05)

    celltypes = z("EV3D", "loading_celltype_by_score")
    components = sorted(celltypes["Component"].unique())
    labels = [CCC.SCORE_LABELS[m] for m in CCC.ALL_SCORES]

    def celltype_grid(axs):
        fig = axs[0].figure
        fig.clear()
        grid = fig.subplots(len(components), 2, squeeze=False)
        for row, k in zip(grid, components, strict=True):
            for ax, role in zip(row, ("Sender", "Receiver"), strict=True):
                t = celltypes[
                    (celltypes["Component"] == k) & (celltypes["Role"] == role)
                ]
                table = t.pivot_table(
                    index="Cell type",
                    columns="Method",
                    values="Normalized loading",
                    sort=False,
                )[labels]
                interpretation._draw_heatmap(ax, table)
            row[0].set_ylabel(f"Component {k}", fontweight="bold")
        grid[0][0].set_title("Sender")
        grid[0][1].set_title("Receiver")

    def tall(axs):
        axs[0].figure.set_size_inches(5, 2 * len(components))
        celltype_grid(axs)

    return {"EV3A": manual(match_bars, "rect"), "EV3D": manual(tall)}


def figure_ev4(z: Zip) -> dict:
    receptors = z("EV4F,I", "receptor_expression_on_map")
    magnitude = z("EV4J", "hormone_magnitude_distribution")
    shares = z("EV4L", "hormone_extreme_share_by_biopsy").set_index("Biopsy")
    return (
        module(
            "figure4.figureS_ibd_hormone_axes.receptor_maps",
            {(F4, "programs"): lambda table: receptors},
            {"EV4F": 0, "EV4I": 1},
        )
        | module(
            "figure4.figureS_ibd_hormone_axes.magnitude_hist",
            {
                (F4, "programs"): lambda table: None,
                (F4, "hormone_angles"): lambda cells: magnitude,
            },
            {"EV4J": None},
        )
        | module(
            "figure4.figureS_ibd_hormone_axes.extreme_share",
            {(F4, "class_shares"): lambda _: shares},
            {"EV4L": None},
        )
    )


def figure_ev5(z: Zip) -> dict:
    best = z("EV5A", "program_best_column_cos2").rename(
        columns={"Best-match cos2": "Cos2"}
    )
    depth = z("EV5C", "depth_tracking_by_component")
    labels = {"ALAD": "alad", "IBD": "ibd"}
    groups = {
        dataset: {
            (m, lab): g["|rho|"].to_numpy()
            for (m, lab), g in depth[depth["Cohort"] == cohort].groupby(
                ["Method", "Labelling"], sort=False
            )
        }
        for cohort, dataset in labels.items()
    }
    return module(
        "figure5.figureS_comparator_split.ibd_best",
        {(_recovery, "best_columns"): lambda dataset: best.drop(columns="Dataset")},
        {"EV5A": None},
    ) | module(
        "figure5.figureS_comparator_split.depth",
        {(_depth, "_values"): lambda dataset: groups[dataset]},
        {"EV5C": None},
    )


def figure_s1(z: Zip) -> dict:
    conditions = z("S1B", "rise_condition_factor").set_index("Sample")

    def condition_factor(axs):
        heatmap(axs[0], conditions, "RISE component")
        axs[0].set_ylabel(f"Conditions (n={len(conditions)})")

    collinearity = z("S1D", "cpd_lr_pair_max_cosine_by_rank").merge(
        z("S1D", "cpd_max_cosine_by_rank"), on=["Rank", "Start", "Seed"]
    )
    # Each component's jackknife rows, and each seed refit as one row at its
    # aggregate FMS (the per-component seed rows aren't drawn or exported)
    jackknife = (
        z("S1E", "jackknife_fms_rise_components")
        .rename(columns={"Model": "Stage"})
        .assign(Axis=_stability.JACKKNIFE_AXIS)
    )
    aggregate = z("S1E", "aggregate_fms_rise_by_replicate")
    seed_rows = (
        aggregate[aggregate["Axis"] != _stability.JACKKNIFE_AXIS]
        .rename(columns={"Aggregate FMS": "FMS"})
        .assign(Stage=jackknife["Stage"].iloc[0], Component=1)
    )
    stability = pd.concat([jackknife, seed_rows], ignore_index=True).rename(
        columns={
            "Left-out sample": "LeftOut",
            "Matched replicate component": "MatchedComponent",
        }
    )
    return (
        {"S1B": manual(condition_factor)}
        | module(
            "figure2.figureS_alad_cohort.cp_collinearity",
            {(common, "load_cpd_rank"): lambda dataset: collinearity},
            {"S1D": None},
        )
        | module(
            "figure2.figureS_alad_cohort.rise_fms",
            {(_stability, "stability"): lambda: stability},
            {"S1E": None},
        )
    )


def figure_s2(z: Zip) -> dict:
    sweep = z("S2B-C", "bicv_r2x_by_rise_rank")
    lr = z("S2E,G", "cpd_lr_pair_max_cosine_by_rank")
    every_mode = lr.merge(
        z("S2G", "cpd_max_cosine_by_rank"), on=["Rank", "Start", "Seed"]
    )
    minima = seeds(z("S2F", "cpd_minima_by_rank"))
    genes = z("S2I", "rise_gene_factor_top_genes").set_index("Gene")

    def gene_factor(axs):
        heatmap(axs[0], genes, "RISE component")
        axs[0].set_ylabel(f"Top genes (n={len(genes)})")

    runner = {(common, "load_runner_output"): lambda *_, **__: sweep}
    return (
        module(
            "figure4.figureS_ibd_rank_selection.rise_rank_selection",
            runner,
            {"S2B": None},
        )
        | module("figure4.figureS_ibd_rank_selection.bicv_test", runner, {"S2C": None})
        | module(
            "figure4.figureS_ibd_rank_selection.cp_collinearity",
            {
                (common, "load_cpd_rank"): lambda dataset: lr,
                (common, "load_minima"): lambda dataset: minima,
            },
            {"S2E": None},
        )
        | module(
            "figure4.figureS_ibd_rank_selection.mode_collinearity",
            {(common, "load_cpd_rank"): lambda dataset: every_mode},
            {"S2G": None},
        )
        | {"S2I": manual(gene_factor)}
    )


def figure_s3(z: Zip) -> dict:
    table = z("S3A", "anova_component_inflammation")
    return {
        "S3A": manual(
            lambda axs: common.plot_association_heatmap(table, axs[0]), "rect"
        )
    } | module(
        "figure4.figureS_ibd_associations.status_loadings",
        {
            (status_loadings, "loading_points"): lambda: z(
                "S3B-C", "cpd_loadings_by_status"
            ),
            (F4, "association_table"): lambda _: z("S3B-C", "cpd_status_anova"),
            (status_loadings, "_source_data"): lambda *_: None,
        },
        {"S3B": 0, "S3C": 1},
    )


def figure_s4(z: Zip) -> dict:
    """The fits' chosen ranks (`fit_rank`) aren't source data: not marked."""

    def panel(name: str, folder: str, csv: str) -> dict:
        return module(
            f"figure5.figureS_comparator_rank_selection.{name}",
            {
                (comparator_ranks, "both_cohorts"): lambda table: z(folder, csv),
                (comparator_ranks, "mark_chosen"): lambda *_: None,
                (comparator_ranks, "fit_rank"): lambda *_: None,
            },
            {folder: None},
        )

    return panel("pb_fit_fine", "S4F", "pb_fit_r2x_by_rank_and_start") | panel(
        "stac_shares_fine", "S4J", "staccato_eigenvalue_shares"
    )


def missed_columns(z: Zip, folder: str, dataset: str) -> pd.DataFrame:
    """`_missed.columns` from the exported shares, each column's number read
    off its name (e.g. 'P3 C')."""
    t = z(folder, f"comparator_column_share_{dataset}")
    return t.rename(columns={"CCC-RISE component": "Of"}).assign(
        Column=t["Name"].map(lambda n: int(re.findall(r"\d+", n)[0]))
    )


def figure_s5(z: Zip) -> dict:
    """The contents panels' selection (associated columns carrying a
    signalling pair CCC-RISE lacks) is drawn from inputs that aren't their
    source data: their rows are read back from the CSVs instead."""
    columns = {d: missed_columns(z, f, d) for d, f in (("alad", "S5A"), ("ibd", "S5B"))}
    contents = z("S5D", "left_over_column_contents_ibd")
    value = contents.columns[1]
    rows, pairs = [], list(dict.fromkeys(contents["LR pair"]))
    for name, g in contents.groupby("Column", sort=False):
        name = str(name)
        if name != "cccRISE":
            method = "PB" if name.startswith("P") else "STAC"
            rows.append((name, method, g.set_index("LR pair")[value]))
    closest = contents[contents["Column"] == "cccRISE"].set_index("LR pair")[value]
    shares = {(_missed, "columns"): lambda dataset: columns[dataset]}
    return (
        module("figure5.figureS_comparator_missed.alad_share", shares, {"S5A": None})
        | module("figure5.figureS_comparator_missed.ibd_share", shares, {"S5B": None})
        | module(
            "figure5.figureS_comparator_missed.ibd_contents",
            {
                (_missed, "boxed_rows"): lambda dataset: (rows, pairs, closest),
                (_missed, "_contents_source"): lambda *_: None,
            },
            {"S5D": None},
        )
    )


def figure_s6(z: Zip) -> dict:
    weights = z("S6A", "cell_weights_groups_alad")
    seeds_table = z("S6E", "seed_to_seed_lr_set_jaccard").rename(
        columns={"Jaccard": "Value", "Seed A": "SeedA", "Seed B": "SeedB"}
    )

    def seed_agreement(filename, *_, **__):
        cohort = "IBD" if "ibd" in str(filename) else "ALAD"
        return seeds_table[seeds_table["Cohort"] == cohort].drop(columns="Cohort")

    return module(
        "figure5.figureS_comparator_single_cell.alad_weights",
        {(_niches, "load"): lambda name, dataset: weights},
        {"S6A": None},
    ) | module(
        "figure5.figureS_comparator_single_cell.scriabin_seeds",
        {(common, "load_runner_output"): seed_agreement},
        {"S6E": None},
    )


PANELS: dict[str, Callable[[Zip], dict]] = {
    "Figure 2": figure2,
    "Figure 3": figure3,
    "Figure 4": figure4,
    "Figure 5": figure5,
    "Figure EV1": figure_ev1,
    "Figure EV2": figure_ev2,
    "Figure EV3": figure_ev3,
    "Figure EV4": figure_ev4,
    "Figure EV5": figure_ev5,
    "Appendix Figure S1": figure_s1,
    "Appendix Figure S2": figure_s2,
    "Appendix Figure S3": figure_s3,
    "Appendix Figure S4": figure_s4,
    "Appendix Figure S5": figure_s5,
    "Appendix Figure S6": figure_s6,
}


def main() -> None:
    style.apply_style()
    chosen = sys.argv[1:] or list(PANELS)
    failed = []
    for label in chosen:
        if not (ZIP_DIR / f"{label}.zip").exists():
            sys.exit(
                f"{label}.zip not found in {ZIP_DIR}: run `uv run cccrise-sourcedata`"
            )
        out = OUT_DIR / label
        out.mkdir(parents=True, exist_ok=True)
        try:
            panels = PANELS[label](Zip(label))
        except Exception as e:  # report every figure's failures, not only the first
            failed.append(f"{label}: {type(e).__name__}: {e}")
            continue
        for panel, (fig, index) in panels.items():
            save(out / f"{panel}.png", fig, index)
            print(out / f"{panel}.png")
        plt.close("all")
    if failed:
        sys.exit("Not replayed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
