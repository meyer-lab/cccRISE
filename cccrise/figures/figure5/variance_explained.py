"""Figure 5B: per CCC-RISE component and side, the share of its per-cell
weight's variance that (sample x cell type) means hold (the pseudobulk
ceiling, the larger over coarse and fine cell types) against what the
closest PB or STAC LR column holds of it, painted on cells: each cell gets
the column's loading for its sample times its loading for its cell type,
and the squared correlation with the component's weight is taken
(`run_comparator_scores`' ``label_ceiling``). A painted value is constant
within each sample x cell type, so it holds at most the ceiling: every point
sits on or below the diagonal."""

import pandas as pd
from matplotlib.lines import Line2D

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, load
from ._data import short
from ._recovery import LABEL_SIZE, panel_title

AXES = ("fig5_variance",)
TITLE = "Component Variance Explained ($R^2$)"
#: Each side's marker and area.
MARKERS = {"sender": ("o", 14), "receiver": ("s", 12)}
#: Labelled components, by (cohort, component): the most a column holds, the
#: cohort in the label's color; where the one label sits (data), with a
#: leader to its sender and its receiver point.
LABELLED = {("alad", 4): (0.9, 0.36), ("ibd", 4): (0.52, 0.3)}


def best_rows() -> pd.DataFrame:
    """Per cohort, component and side: the larger ceiling over the labellings,
    and the closest column of either labelling (greatest LR cos^2; the column
    Figure 5A scores) with what it holds when painted (``Painted``)."""
    frames = []
    for dataset in DATASET_LABELS:
        t = load("label_ceiling", dataset=dataset).assign(
            Cos2=lambda d: d["Cosine"] ** 2
        )
        for side, g in t.groupby("Side"):
            closest = g.loc[g.groupby("Component")["Cos2"].idxmax()].set_index(
                "Component"
            )
            frames.append(
                closest.assign(
                    Ceiling=g.groupby("Component")["Ceiling"].max(),
                    Dataset=dataset,
                    Side=side,
                ).reset_index()
            )
    return pd.concat(frames, ignore_index=True)


def _labels(ax, t: pd.DataFrame) -> None:
    """Each of `LABELLED` named once, leaders to its sender and receiver."""
    for (dataset, component), at in LABELLED.items():
        rows = t[(t["Dataset"] == dataset) & (t["Component"] == component)]
        for i, (_, r) in enumerate(rows.iterrows()):
            ax.annotate(
                f"C{component}" if i == 0 else "",
                (r["Ceiling"], r["Painted"]),
                xytext=at,
                ha="center",
                va="center",
                fontsize=LABEL_SIZE,
                color=DATASET_COLORS[dataset],
                zorder=4,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.4,
                    "color": "0.4",
                    "shrinkA": 3,
                    "shrinkB": 2,
                },
            )


def _source(t: pd.DataFrame) -> None:
    named = t.assign(
        Cohort=t["Dataset"].map(DATASET_LABELS),
        Closest=[
            f"{short(m, c)} {lab[0].upper()}"
            for m, c, lab in zip(t["Method"], t["Column"], t["Labelling"], strict=True)
        ],
    )
    columns = {
        "Cohort": "Dataset",
        "Component": "Component",
        "Side": "Side",
        "Ceiling": "Pseudobulk ceiling R2",
        "Closest": "Closest comparator LR column",
        "Cos2": "LR cos2",
        "Painted": "Closest comparator LR column R2",
    }
    common.source_data(
        "component_variance_explained",
        named[list(columns)].rename(columns=columns),
        "Per component and side: the share of its per-cell weight's variance that "
        "(sample x cell type) means hold, the larger over coarse and fine cell "
        "types, and the squared correlation of that weight with the closest PB or "
        "STAC LR column (greatest LR cos2, either labelling) painted on cells as "
        "the cell's sample loading x its cell type's loading.",
    )


def plot(axs):
    (ax,) = axs
    t = best_rows()
    _source(t)
    ax.plot(
        [0, 1],
        [0, 1],
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
        zorder=1,
    )
    for _, r in t.iterrows():
        marker, size = MARKERS[r["Side"]]
        ax.scatter(
            r["Ceiling"],
            r["Painted"],
            s=size,
            marker=marker,
            color=DATASET_COLORS[r["Dataset"]],
            ec="white",
            lw=0.3,
            zorder=3,
        )
    _labels(ax, t)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        xlabel="Pseudobulk ceiling",
        ylabel="Closest comparator column",
    )
    panel_title(ax, TITLE)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker=MARKERS[s][0], ls="", ms=3, color="0.4", label=s.title()
            )
            for s in MARKERS
        ],
        loc="upper left",
        fontsize=LABEL_SIZE,
        handletextpad=0.2,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
