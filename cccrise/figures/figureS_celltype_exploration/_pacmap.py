"""PaCMAP panels (pre-registration §4B)."""

import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ...validation import hormone_cells as H
from .. import style
from . import _common as C

PROGRAM_COLUMNS = [
    ("loading", "cccRISE\nloading"),
    ("contribution", "cccRISE\ncontribution"),
    ("expressed", "cccRISE\nexpressed part"),
    ("paint:Pseudobulk CPD", "PB fit paint"),
    ("paint:Tensor-cell2cell", "TC2C fit paint"),
    ("oracle:loading", "Label oracle\n(loading)"),
    ("oracle:contribution", "Label oracle\n(contribution)"),
    ("residual:loading", "Residual\n(loading)"),
    ("residual:contribution", "Residual\n(contribution)"),
]


def legend_below(fig, ax, ncols=2):
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=ncols,
        fontsize=style.FONT_SIZE_DENSE,
        markerscale=4,
        frameon=False,
    )
    return leg


def labels_figure():
    c = C.cells()
    fig = C.figure(180, 120)
    axs = fig.subplots(1, 4)
    shuffled_labels(
        axs[0],
        c["lineage"],
        {lin: C.lineage_colors()[lin] for lin in C.lineage_order()},
        "Lineage",
    )
    legend_below(fig, axs[0], 1)
    carrier_colors = carrier_palette()
    C.scatter_labels(
        axs[1], c["celltype"], carrier_colors, "Program carriers (fine labels)"
    )
    legend_below(fig, axs[1], 1)
    groups = group_palette(c["group"])
    shuffled_labels(axs[2], c["group"], groups, f"Group ({C.CFG.group_column})")
    legend_below(fig, axs[2], 1)
    samples = np.unique(c["sample"])
    rng = np.random.default_rng(0)
    palette = sns.color_palette("husl", len(samples))
    colors = dict(zip(samples, [palette[i] for i in rng.permutation(len(samples))]))
    xy = C.pacmap()
    axs[3].scatter(
        *xy.T,
        c=[colors[s] for s in c["sample"]],
        s=C.POINT,
        linewidths=0,
        rasterized=True,
    )
    C.bare(axs[3], f"Sample ({len(samples)})")
    return fig


IBD_CARRIER_COLORS = {
    "Enteroendocrine cells": "#E69F00",
    "Enterochromaffin cells": "#0072B2",
    "Tuft cells": "#8C564B",
    "Monocytes SOD2+": "#CC79A7",
    "ILCs": "#009E73",
    "Mast cells": "#222222",
}
IBD_GROUP_COLORS = {
    "Healthy": "0.7",
    "CD_uninflamed": style.CATEGORICAL[0],
    "CD_inflamed": style.CATEGORICAL[3],
}


def carrier_palette() -> dict[str, str]:
    """A fixed colour per carrier label (IBD's as in the first report)."""
    if C.DATASET == "ibd":
        return IBD_CARRIER_COLORS
    palette = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#8C564B", "#222222"]
    return dict(zip(C.CFG.all_carriers(), palette, strict=False))


def group_palette(groups: np.ndarray) -> dict[str, str]:
    if C.DATASET == "ibd":
        return IBD_GROUP_COLORS
    levels = sorted(set(groups))
    colors = ["0.7", style.CATEGORICAL[3], style.CATEGORICAL[0], style.CATEGORICAL[2]]
    return dict(zip(levels, colors, strict=False))


def shuffled_labels(ax, labels, colors: dict, title: str):
    """All cells drawn at once in random order, so no label hides another."""
    xy = C.pacmap()
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(
        *xy[order].T,
        c=[colors.get(lab, "0.88") for lab in labels[order]],
        s=C.POINT,
        linewidths=0,
        rasterized=True,
    )
    for lab, color in colors.items():
        ax.scatter([], [], c=color, s=1, label=f"{lab} ({(labels == lab).sum():,})")
    C.bare(ax, title)


def selected_eigenstates(cut: float = 0.4) -> list[int]:
    t = C.table("composition")
    sel = set()
    for side in H.SIDES:
        f = t[t.view == f"factor_{side}"].pivot(
            index="label", columns="column", values="value"
        )
        for col in f:
            a = f[col].abs()
            sel |= {int(e[1:]) for e in a.index[a >= cut * a.max()]}
    return sorted(sel)


def eigenstates_figure():
    states = selected_eigenstates()
    P = C.cells()["projections"]
    n = len(states)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig = C.figure(180, 32 * rows)
    axs = np.atleast_2d(fig.subplots(rows, cols))
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    for ax, e in zip(axs.ravel(), states, strict=False):
        C.scatter_values(ax, P[:, e - 1].astype(float), title=f"E{e}")
    return fig


def column_values(k: int, side: str, key: str):
    v = C.values()
    if key.startswith("paint:"):
        method = key.split(":")[1]
        name = f"paint_C{k}_{method.replace(' ', '_')}_{side}"
        return v.get(name)
    if key.startswith("oracle:"):
        return C.oracle(v[f"C{k}_{side}_{key.split(':')[1]}"].astype(float))
    if key.startswith("residual:"):
        x = v[f"C{k}_{side}_{key.split(':')[1]}"].astype(float)
        return x - C.oracle(x)
    return v[f"C{k}_{side}_{key}"].astype(float)


def program_figure(k: int, span_mode: str = "q99.9"):
    paint = C.table("paint")
    paint = paint[(paint.cut == 0.85) & (paint.component == k)].set_index("method")
    fig = C.figure(180, 62)
    axs = fig.subplots(2, len(PROGRAM_COLUMNS))
    for r, side in enumerate(H.SIDES):
        base = {}
        for j, (key, title) in enumerate(PROGRAM_COLUMNS):
            ax = axs[r, j]
            vals = column_values(k, side, key)
            if vals is None:
                C.bare(ax, title if r == 0 else None)
                ax.text(
                    0.5,
                    0.5,
                    "no fit contains\nthis program",
                    ha="center",
                    va="center",
                    fontsize=style.FONT_SIZE_DENSE,
                    transform=ax.transAxes,
                )
                continue
            quantity = key.split(":")[-1]
            if key.startswith("paint:"):
                lim = 1.0
            else:
                base.setdefault(
                    quantity, C.span(column_values(k, side, quantity), span_mode)
                )
                lim = base[quantity]
            label = title
            if key.startswith("paint:") and r == 0:
                row = paint.loc[key.split(":")[1]]
                label = f"{title}\n{row['labelling'].lstrip('_')}\ncol {int(row['column'])}, |cos| {row['lr_cos']:.2f}"
            C.scatter_values(ax, vals, lim, label if r == 0 else None)
        axs[r, 0].set_ylabel(side.capitalize(), fontsize=style.FONT_SIZE_LABEL)
        axs[r, 0].yaxis.label.set_visible(True)
    fig.suptitle(
        f"{C.program_label(k)}: carriers {', '.join(C.carriers(k, 'sender'))} → {', '.join(C.carriers(k, 'receiver'))}; colour span ±{'99.9th pct |v|' if span_mode == 'q99.9' else '0.3 × max'} per quantity",
        fontsize=style.FONT_SIZE_TITLE,
    )
    sm = ScalarMappable(norm=Normalize(-1, 1), cmap=style.CMAPS["diverging"])
    cb = fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.005)
    cb.set_label("value / span", fontsize=style.FONT_SIZE_DENSE)
    cb.ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    return fig


def legacy_mean_figure(k: int = 3, side: str = "sender"):
    """Per-pixel mean shading (as legacy `plot_wc_pacmap`) next to the sorted
    scatter, for one panel: how averaging hides rare cells."""
    v = C.values()[f"C{k}_{side}_loading"].astype(float)
    xy = C.pacmap()
    fig = C.figure(120, 60)
    a0, a1 = fig.subplots(1, 2)
    bins = 400
    total, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, weights=v / v.max())
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    a0.imshow(
        mean.T,
        origin="lower",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        cmap=style.CMAPS["diverging"],
        vmin=-0.3,
        vmax=0.3,
        aspect="auto",
        interpolation="nearest",
    )
    C.bare(a0, "Legacy: per-pixel mean of value/max, ±0.3")
    C.scatter_values(a1, v / v.max(), 0.3, "Sorted scatter of value/max, ±0.3")
    fig.suptitle(f"{C.program_label(k)} {side} loading", fontsize=style.FONT_SIZE_TITLE)
    return fig


def fig2l_figure():
    """Figure 2L-style sender vs receiver weight per program, coloured by
    the carrier labels, at GROUP_TAU = 0.3 of each weight's max."""
    v = C.values()
    c = C.cells()
    fig = C.figure(180, 120)
    axs = fig.subplots(3, len(C.programs()))
    for j, k in enumerate(C.programs()):
        for i, q in enumerate(("loading", "contribution", "expressed")):
            ax = axs[i, j]
            s, r = (
                v[f"C{k}_sender_{q}"].astype(float),
                v[f"C{k}_receiver_{q}"].astype(float),
            )
            send = np.isin(c["celltype"], C.carriers(k, "sender"))
            recv = np.isin(c["celltype"], C.carriers(k, "receiver"))
            other = ~send & ~recv
            ax.scatter(
                s[other],
                r[other],
                s=0.3,
                c=style.COLORS["muted"],
                linewidths=0,
                rasterized=True,
                label="Other labels",
            )
            if (send & recv).any():
                ax.scatter(
                    s[send & recv],
                    r[send & recv],
                    s=1.5,
                    c=style.COLORS["highlight"],
                    linewidths=0,
                    rasterized=True,
                    label="Carrier (both)",
                )
            ax.scatter(
                s[send & ~recv],
                r[send & ~recv],
                s=1.5,
                c=style.COLORS["sender"],
                linewidths=0,
                rasterized=True,
                label="Sender carrier",
            )
            ax.scatter(
                s[recv & ~send],
                r[recv & ~send],
                s=1.5,
                c=style.COLORS["receiver"],
                linewidths=0,
                rasterized=True,
                label="Receiver carrier",
            )
            cs, cr = 0.3 * s.max(), 0.3 * r.max()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.plot(
                [x0, cs],
                [cr, cr],
                [cs, cs],
                [y0, cr],
                color=style.COLORS["reference"],
                ls="--",
                lw=0.5,
            )
            ax.plot(
                [cs, s.max()],
                [cr, r.max()],
                color=style.COLORS["reference"],
                ls="--",
                lw=0.5,
            )
            ax.set(xlim=(x0, x1), ylim=(y0, y1))
            ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
            if i == 0:
                ax.set_title(C.program_label(k), fontsize=style.FONT_SIZE_DENSE)
            if j == 0:
                ax.set_ylabel(
                    f"{C.QUANTITY_LABELS[q]}\nreceiver", fontsize=style.FONT_SIZE_DENSE
                )
            if i == 2:
                ax.set_xlabel("sender", fontsize=style.FONT_SIZE_DENSE)
    handles, labels = axs[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=4,
        fontsize=style.FONT_SIZE_DENSE,
        markerscale=5,
        frameon=False,
    )
    return fig


def spans_figure(k: int = 3):
    """T8: the same panels at the legacy (±0.3 max) and 99.9th-percentile spans."""
    fig = C.figure(180, 95)
    axs = fig.subplots(2, 4)
    for j, (side, q) in enumerate(
        [(s, q) for s in H.SIDES for q in ("loading", "contribution")]
    ):
        v = C.values()[f"C{k}_{side}_{q}"].astype(float)
        for i, mode in enumerate(("legacy", "q99.9")):
            C.scatter_values(
                axs[i, j],
                v,
                C.span(v, mode),
                f"{side} {q}\n{'±0.3 × max' if mode == 'legacy' else '±99.9th pct'}",
            )
    return fig


def seed_stability_figure():
    t = C.table("pacmap_stability")
    fig = C.figure(120, 60)
    ax = fig.add_subplot()
    t = t.sort_values("kept")
    ax.barh(
        t["celltype"] + " (" + t["seed"].astype(str) + ")",
        t["kept"],
        color=style.COLORS["emphasis"],
    )
    ax.axvline(
        t["random_expected"].iloc[0], color=style.COLORS["reference"], ls="--", lw=0.5
    )
    ax.set_xlabel(
        "Share of 30 nearest neighbours (seed 42) kept in the other seed",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    return fig
