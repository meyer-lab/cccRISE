"""Figure 4's supplementary drawing functions, each for one labelling, bound
into panels with the panel modules: reach (cccRISE's programs in each
comparator) and what Tensor-cell2cell's published construction drops."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...comparators.scores import CONTAINED
from .. import common, style
from ._common import COMPARATORS, METHOD_COLORS, METHOD_LABELS, load

MODE_LABELS = {"lr": "LR", "sample": "sample"}


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
