"""Eigenstate composition heatmaps (pre-registration §4A)."""

import numpy as np
import pandas as pd

from .. import style
from . import _common as C


def composition(view: str, level: str) -> pd.DataFrame:
    t = C.table("composition")
    t = t[(t.view == view) & (t.level == level)]
    frame = t.pivot(index="label", columns="column", values="value")
    order = C.lineage_order() if level == "lineage" else C.fine_order()
    cols = (
        sorted(frame.columns, key=lambda c: int(c[1:]))
        if str(frame.columns[0]).startswith("E")
        else list(frame.columns)
    )
    return frame.reindex(index=order, columns=cols)


def eigenstate_order() -> list[str]:
    """Eigenstates clustered on the fine mixture heatmap."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    fine = composition("A1_mixture", "fine")
    return list(
        fine.columns[
            leaves_list(linkage(fine.T.to_numpy(), "average", metric="correlation"))
        ]
    )


def program_strip(order: list[str]) -> pd.DataFrame:
    """|B| and |C| of each program over the eigenstates, relative to max."""
    t = C.table("composition")
    rows = {}
    for side, key in (("sender", "B"), ("receiver", "C")):
        f = t[t.view == f"factor_{side}"].pivot(
            index="label", columns="column", values="value"
        )
        for k in C.programs():
            col = f[f"C{k}"].abs()
            rows[f"C{k} {key}"] = col / col.max()
    return pd.DataFrame(rows).T[order]


def strip_heatmap(ax, order: list[str]) -> None:
    """The programs' |B| and |C| over the eigenstates, above a heatmap."""
    C.heatmap(
        ax,
        program_strip(order),
        style.CMAPS["weight"],
        0,
        1,
        "|factor| / max",
        xticks=False,
    )


def mixture_figure(
    level: str,
    view: str = "A1_mixture",
    label: str = "Mean share of a cell's\nsquared projection",
):
    order = eigenstate_order()
    frame = composition(view, level)[order]
    norm = composition("A1_norm", level)["captured_norm"]
    n_rows = len(frame)
    fig = C.figure(180, 50 + 3.2 * n_rows)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.6 * 12, 2.6 * n_rows], width_ratios=[30, 1.2]
    )
    ax_strip, ax, ax_norm = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    )
    strip_heatmap(ax_strip, order)
    C.heatmap(
        ax,
        frame,
        style.CMAPS["weight"],
        0,
        np.nanpercentile(frame.to_numpy(), 99.5),
        label,
    )
    if level == "fine":
        C.lineage_separators(ax, list(frame.index))
    ax_norm.barh(range(n_rows), norm.to_numpy(), color=style.COLORS["ink"])
    ax_norm.set_ylim(n_rows - 0.5, -0.5)
    ax_norm.set_yticks([])
    ax_norm.tick_params(labelsize=style.FONT_SIZE_DENSE)
    ax_norm.set_xlabel("Captured\nnorm", fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel(
        "RISE eigenstate (clustered on the fine heatmap)",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def share_figure(level: str):
    order = eigenstate_order()
    frame = composition("A2_share_sample", level)[order]
    fig = C.figure(180, 50 + 3.2 * len(frame))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6 * 12, 2.6 * len(frame)])
    ax_strip, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    strip_heatmap(ax_strip, order)
    C.heatmap(
        ax,
        frame,
        style.CMAPS["weight"],
        0,
        np.nanpercentile(frame.to_numpy(), 99.5),
        "Share of the eigenstate's\nenergy (sample mean)",
    )
    if level == "fine":
        C.lineage_separators(ax, list(frame.index))
    ax.set_xlabel("RISE eigenstate (columns sum to 1)", fontsize=style.FONT_SIZE_DENSE)
    return fig


def signed_figure():
    order = eigenstate_order()
    frame = composition("A4_signed", "fine")[order]
    fig = C.figure(180, 40 + 2.6 * len(frame))
    ax = fig.add_subplot()
    lim = np.nanpercentile(np.abs(frame.to_numpy()), 99)
    C.heatmap(
        ax,
        frame,
        style.CMAPS["diverging"],
        -lim,
        lim,
        "Mean signed projection\n(sample-scaled)",
    )
    C.lineage_separators(ax, list(frame.index))
    return fig


def lineage_to_fine_figure(view_suffix: str = ""):
    energy = composition(f"A3_energy_within_lineage{view_suffix}", "fine")
    energy = energy[[f"C{k} {s}" for k in C.programs() for s in ("sender", "receiver")]]
    cells = composition(f"A3_cells_within_lineage{view_suffix}", "fine")["cells"]
    ratio = np.log2(energy.div(cells, axis=0).replace(0, np.nan))
    fig = C.figure(180, 40 + 2.6 * len(energy))
    a0, a1, a2 = fig.subplots(1, 3, width_ratios=[1.2, 12, 12])
    C.heatmap(
        a0,
        cells.to_frame("cells"),
        style.CMAPS["weight"],
        0,
        1,
        "",
        yfont=style.FONT_SIZE_DENSE,
    )
    a0.set_title("Share of\nlineage cells", fontsize=style.FONT_SIZE_DENSE)
    C.heatmap(
        a1, energy, style.CMAPS["weight"], 0, 1, "Share of lineage's\nprogram energy"
    )
    a1.set_yticks([])
    a1.set_title("Program-side energy within lineage", fontsize=style.FONT_SIZE_DENSE)
    lim = np.nanpercentile(np.abs(ratio.to_numpy()), 98)
    C.heatmap(
        a2,
        ratio,
        style.CMAPS["diverging"],
        -lim,
        lim,
        "log2 energy share /\ncell share",
    )
    a2.set_yticks([])
    a2.set_title("Enrichment over cell share", fontsize=style.FONT_SIZE_DENSE)
    for ax in (a0, a1, a2):
        C.lineage_separators(ax, list(energy.index))
    return fig
