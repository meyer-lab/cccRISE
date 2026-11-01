"""Figure 4's supplementary drawing functions, each for one labelling (and,
where a supplement widens it, a set of methods), bound into panels with the
panel modules: reach (cccRISE's programs in each comparator, LR or sample
mode), each comparator LR column in cccRISE by audit class, containment
between every pair of methods, representability, and what Tensor-cell2cell's
published construction drops."""

import itertools

import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from ...comparators.scores import CONTAINED
from .. import common, style
from ._common import COMPARATORS, METHOD_COLORS, METHOD_LABELS, load

MODE_LABELS = {"lr": "LR", "sample": "sample"}
#: Audit classes, bottom of the stack first, and their colors.
CLASSES = {
    "contained in cccRISE": style.COLORS["emphasis"],
    "one sample": "0.55",
    "marker or homophilic": "0.75",
    "abundance readout": style.CATEGORICAL[8],
    "diffuse": style.CATEGORICAL[5],
    "candidate biology": style.COLORS["highlight"],
}


def containment(labelling: str, of: str, mode: str) -> pd.DataFrame:
    """`of`'s columns' containment on `mode` in every other method."""
    table = load("containment", labelling)
    return table[(table["Of"] == of) & (table["Mode"] == mode)]


def reach(axs, labelling: str, methods=COMPARATORS, mode: str = "lr") -> None:
    """Each cccRISE program's containment in each comparator on `mode`,
    contained cells framed; on the LR mode, a dot where the sample mode is
    contained too."""
    ax = axs[0]
    table = containment(labelling, "cccRISE", mode)
    table = table[table["In"].isin(methods)]
    common.source_data(
        f"cccrise_{mode}_containment",
        table.drop(columns=["Of", "Mode"]).rename(columns={"In": "Method"}),
        f"Each cccRISE component's {MODE_LABELS[mode]} column: its share of "
        "squared norm in the comparator's column span, the 95th percentile of "
        "that under 1,000 row permutations, and whether it is contained "
        f"(>= {CONTAINED} and above the null).",
    )
    values = table.pivot(index="Component", columns="In", values="Containment")
    values = values[list(methods)]
    contained = table.pivot(index="Component", columns="In", values="Contained")
    image = ax.imshow(
        values.to_numpy(), cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto"
    )
    for (i, j), value in np.ndenumerate(values.to_numpy()):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color=_ink(value),
        )
        if contained.loc[values.index[i], values.columns[j]]:
            ax.add_patch(_frame(j, i))
    if mode == "lr":
        _mark_sample_contained(ax, labelling, values)
    ax.set_yticks(range(len(values)), [f"C{c}" for c in values.index])
    ax.set_xticks(
        range(len(methods)),
        [METHOD_LABELS[m] for m in methods],
        rotation=30,
        ha="right",
    )
    ax.tick_params(length=0)
    common.colorbar(ax, image, f"{MODE_LABELS[mode]} containment")
    ax.set_title(f"{labelling.capitalize()} labels", loc="left")


#: Least vertical distance between two `label_points` labels (points).
LABEL_GAP = 6.5


def label_points(ax, xs, ys, labels, leaders: bool = False, **text) -> None:
    """Label each point (`xs`, `ys`) with its entry of `labels` to its right, labels
    nudged up just enough to sit `LABEL_GAP` apart, with a leader line to
    any label moved off its point (to every label with `leaders`); `text`
    goes to `annotate`. Call once the axes' limits are final."""
    ax.autoscale_view()
    to_points = 72 / ax.figure.dpi
    xy = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    heights = ax.transData.transform(xy)[:, 1] * to_points
    placed = heights.copy()
    order = np.argsort(heights)
    for below, above in itertools.pairwise(order):
        placed[above] = max(placed[above], placed[below] + LABEL_GAP)
    gap = 10 if leaders else 4
    for (x, y), label, shift in zip(xy, labels, placed - heights, strict=True):
        ax.annotate(
            label,
            (x, y),
            xytext=(gap, shift),
            textcoords="offset points",
            fontsize=style.FONT_SIZE_DENSE,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": style.COLORS["reference"],
                "shrinkA": 0,
                "shrinkB": 2,
            }
            if leaders or shift > 1
            else None,
            **text,
        )


def _ink(value) -> str:
    """Text color readable on `style.CMAPS["weight"]` at `value`."""
    return "white" if float(value) > 0.7 else style.COLORS["ink"]


def _frame(x: float, y: float) -> Rectangle:
    return Rectangle(
        (x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="black", lw=style.LINE_WIDTH
    )


def _mark_sample_contained(ax, labelling: str, values: pd.DataFrame) -> None:
    sample = containment(labelling, "cccRISE", "sample")
    sample = sample.pivot(index="Component", columns="In", values="Contained")
    for (i, j), _ in np.ndenumerate(values.to_numpy()):
        if sample.loc[values.index[i], values.columns[j]]:
            ax.plot(j + 0.32, i - 0.3, "o", color="white", markersize=2.5, mec="black")


def outside(axs, labelling: str, methods=COMPARATORS) -> None:
    """Each comparator LR column's containment in cccRISE's LR span, colored
    by its audit class."""
    ax = axs[0]
    table = load("containment", labelling)
    table = table[
        (table["In"] == "cccRISE") & (table["Mode"] == "lr") & table["Of"].isin(methods)
    ]
    classes = load("audit", labelling).set_index(["Method", "Component"])["Class"]
    keys = pd.MultiIndex.from_arrays([table["Of"], table["Component"]])
    table = table.assign(
        Class=classes.reindex(keys).fillna("contained in cccRISE").to_numpy()
    )
    common.source_data(
        "comparator_lr_containment_in_cccrise",
        table.drop(columns=["In", "Mode"]).rename(columns={"Of": "Method"}),
        "Each comparator LR column's share of squared norm in cccRISE's LR "
        "span, its permutation-null 95th percentile, and its audit class.",
    )
    rng = np.random.default_rng(0)
    for x, method in enumerate(methods):
        group = table[table["Of"] == method]
        ax.scatter(
            x + rng.uniform(-0.2, 0.2, len(group)),
            group["Containment"],
            c=[CLASSES[c] for c in group["Class"]],
            s=style.POINT_SIZE,
            edgecolor="white",
            lw=0.3,
        )
    common.reference_line(ax, CONTAINED)
    ax.set_xticks(
        range(len(methods)),
        [METHOD_LABELS[m] for m in methods],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("LR containment in cccRISE")
    handles = [Patch(color=color, label=name) for name, color in CLASSES.items()]
    common.legend(ax, "below", handles=handles, ncol=2, fontsize=style.FONT_SIZE_DENSE)


def pairwise(axs, labelling: str, methods=("cccRISE", *COMPARATORS)) -> None:
    """Mean LR containment of each method's columns (rows) in each other
    method's span (columns), with the share contained."""
    ax = axs[0]
    table = load("containment", labelling)
    table = table[
        (table["Mode"] == "lr") & table["Of"].isin(methods) & table["In"].isin(methods)
    ]
    summary = table.groupby(["Of", "In"]).agg(
        Mean=("Containment", "mean"),
        Contained=("Contained", "sum"),
        Columns=("Contained", "size"),
    )
    common.source_data(
        "pairwise_lr_containment",
        summary,
        "Mean LR containment of one method's columns (Of) in another's span "
        "(In), and how many of its columns are contained.",
    )
    mean = summary["Mean"].unstack().reindex(index=methods, columns=methods)
    image = ax.imshow(
        mean.to_numpy(), cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto"
    )
    for (i, j), value in np.ndenumerate(mean.to_numpy()):
        if np.isnan(value):
            continue
        row = summary.loc[(methods[i], methods[j])]
        ax.text(
            j,
            i,
            f"{value:.2f}\n{int(row['Contained'])}/{int(row['Columns'])}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color=_ink(value),
        )
    names = [METHOD_LABELS[m] for m in methods]
    ax.set_xticks(range(len(methods)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), names)
    ax.set_xlabel("In the span of")
    ax.set_ylabel("Columns of")
    ax.tick_params(length=0)
    common.colorbar(ax, image, "Mean LR containment")


def representability(axs, labelling: str) -> None:
    """Each cccRISE program's per-cell sender and receiver weight explained
    by its (sample, cell type) means (bars), against random splits of the
    same sizes within each sample (95th percentile, black tick) and by sample
    means alone (white dot)."""
    ax = axs[0]
    table = load("representability", labelling)
    common.source_data(
        "representability_by_sample_and_type",
        table,
        "Per cccRISE component and side: the share of per-cell weight variance "
        "explained by (sample, cell type) means (R2), by sample means alone "
        "(SampleR2), and by (sample, random group) means with each sample's "
        "cell types shuffled among its cells (mean and 95th percentile of 200).",
    )
    width = 0.38
    for offset, side in ((-0.5, "sender"), (0.5, "receiver")):
        rows = table[table["Side"] == side]
        x = rows["Component"].to_numpy() + offset * width
        ax.bar(x, rows["R2"], width, color=style.COLORS[side], label=side.capitalize())
        ax.hlines(rows["NullQ95"], x - width / 2, x + width / 2, color="black", lw=1)
        ax.scatter(
            x,
            rows["SampleR2"],
            s=style.POINT_SIZE / 2,
            color="white",
            ec="black",
            lw=0.4,
        )
    components = sorted(table["Component"].unique())
    ax.set_xticks(components, [f"C{c}" for c in components])
    ax.set_ylim(0, 1)
    ax.set_ylabel("$R^2$, (sample, type) means")
    ax.set_title(f"{labelling.capitalize()} labels", loc="left")
    ax.legend(loc="upper left", fontsize=style.FONT_SIZE_DENSE)


INNER_MEASURES = ("cell types", "cells", "observed entries", "signal")


def inner_loss(axs) -> None:
    """What the published Tensor-cell2cell construction (only the cell types
    present in every sample) drops from the masked tensor, per labelling."""
    ax = axs[0]
    table = pd.concat(
        [load("inner_loss", lab).assign(Labelling=lab) for lab in ("coarse", "fine")]
    )
    common.source_data(
        "published_tc2c_losses",
        table,
        "What keeping only the cell types present in every sample drops from "
        "Tensor-cell2cell's masked tensor: cell types, cells, observed entries "
        "and signal (sum of squares over observed entries).",
    )
    width = 0.38
    x = np.arange(len(INNER_MEASURES))
    for offset, (labelling, color) in zip(
        (-0.5, 0.5),
        (("coarse", METHOD_COLORS["TC2C inner"]), ("fine", style.COLORS["muted"])),
        strict=True,
    ):
        rows = table[table["Labelling"] == labelling].set_index("Measure")
        rows = rows.loc[list(INNER_MEASURES)]
        bars = ax.bar(x + offset * width, rows["Share"], width, color=color)
        ax.bar_label(
            bars,
            [
                f"{d:g}/{t:g}" if m == "cell types" else ""
                for m, d, t in zip(
                    rows.index, rows["Dropped"], rows["Total"], strict=True
                )
            ],
            fontsize=style.FONT_SIZE_DENSE - 1,
            padding=1,
        )
        bars.set_label(f"{labelling.capitalize()} labels")
    ax.set_xticks(x, list(INNER_MEASURES), rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share dropped")
    ax.legend(loc="upper right", fontsize=style.FONT_SIZE_DENSE)


def inner_program_loss(axs, labelling: str) -> None:
    """The share of each cccRISE program's per-cell sender and receiver
    weight carried by the cell types the published construction drops."""
    ax = axs[0]
    table = load("inner_program_loss", labelling)
    common.source_data(
        "cccrise_weight_in_dropped_types",
        table,
        "Share of each cccRISE component's total |per-cell sender or receiver "
        "weight| carried by cells of the types published Tensor-cell2cell "
        "drops (those absent from some sample).",
    )
    width = 0.38
    for offset, side in ((-0.5, "sender"), (0.5, "receiver")):
        rows = table[table["Side"] == side]
        ax.bar(
            rows["Component"] + offset * width,
            rows["Share"],
            width,
            color=style.COLORS[side],
            label=side.capitalize(),
        )
    components = sorted(table["Component"].unique())
    ax.set_xticks(components, [f"C{c}" for c in components])
    ax.set_ylim(0, 1)
    ax.set_ylabel("cccRISE weight in\ndropped cell types")
    ax.set_title(f"{labelling.capitalize()} labels", loc="left")
    ax.legend(loc="upper left", fontsize=style.FONT_SIZE_DENSE)
