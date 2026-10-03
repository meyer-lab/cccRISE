"""Shared components, in both directions: each component's containment on
the condition mode against the LR mode. Left: cccRISE components in each
comparator fit's span; right: each comparator fit's components in cccRISE's.
The shaded corner (>= 0.85 on both) is "shared"; hollow markers fall short of
the row-permutation null on at least one mode."""

from matplotlib.patches import Rectangle

from .. import common, style
from ._common import fit_label, fit_order, load

THRESHOLD = 0.85


def plot(fig):
    table = load("containment")
    fits = fit_order(table)
    colours = dict(zip(fits, style.CATEGORICAL, strict=False))
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, direction, title in zip(
        axes,
        ("cccRISE in comparator", "comparator in cccRISE"),
        ("cccRISE components in each comparator", "Comparator components in cccRISE"),
        strict=True,
    ):
        sub = table[table["direction"] == direction]
        wide = sub.pivot_table(
            index=["method", "rank_mode", "rank", "component"],
            columns="mode",
            values=["containment", "contained"],
            aggfunc="first",
        )
        ax.add_patch(
            Rectangle(
                (THRESHOLD, THRESHOLD),
                1 - THRESHOLD,
                1 - THRESHOLD,
                color=style.COLORS["muted"],
                alpha=0.35,
                lw=0,
            )
        )
        labelled = set()
        for fit in fits:
            rows = wide.loc[fit]
            _scatter_fit(
                ax,
                rows,
                colours[fit],
                fit_label(fit),
                labelled,
                numbered=direction == "cccRISE in comparator",
            )
        if direction != "cccRISE in comparator" and labelled:
            ax.annotate(
                f"{len(labelled)} fits' components",
                (THRESHOLD, THRESHOLD),
                xytext=(2, -10),
                textcoords="offset points",
                fontsize=style.FONT_SIZE_DENSE,
                color=style.COLORS["ink"],
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Containment, condition mode")
        ax.set_title(title)
    axes[0].set_ylabel("Containment, LR mode")
    axes[1].legend(loc="lower right", fontsize=style.FONT_SIZE_DENSE)


def _scatter_fit(ax, rows, colour, label, labelled, numbered):
    """One fit's components. Shared cccRISE components (`numbered`) are
    labelled once per panel; otherwise shared fits are only counted."""
    x, y = rows[("containment", "condition")], rows[("containment", "lr")]
    both = rows[("contained", "condition")].astype(bool) & rows[
        ("contained", "lr")
    ].astype(bool)
    ax.scatter(
        x,
        y,
        s=style.POINT_SIZE_FOREGROUND,
        label=label,
        edgecolor=colour,
        facecolor=[colour if b else "white" for b in both],
    )
    for k, xi, yi in zip(rows.index, x, y, strict=True):
        if xi < THRESHOLD or yi < THRESHOLD:
            continue
        if not numbered:
            labelled.add(label)
        elif k not in labelled:
            labelled.add(k)
            ax.annotate(
                f"C{k}",
                (xi, yi),
                xytext=(-12, -8),
                textcoords="offset points",
                fontsize=style.FONT_SIZE_DENSE,
                color=style.COLORS["ink"],
            )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
