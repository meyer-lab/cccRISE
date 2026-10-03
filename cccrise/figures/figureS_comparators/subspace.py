"""Linear CKA between cccRISE's and each comparator fit's column spans, on the
condition and LR modes. Filled: above both nulls' 95th percentile (tick: the
larger one); the row-permutation null keeps each fit's column geometry, the
Gaussian null does not. LR-mode agreement is an upper bound on shared
biology: LR-pair abundance alone produces some of it."""

from .. import common, style
from ._common import MODE_LABELS, fit_label, fit_order, load


def plot(fig):
    table = load("subspace")
    cka = table[table["statistic"] == "linear_cka"]
    fits = fit_order(cka)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, mode in zip(axes, ("condition", "lr"), strict=True):
        sub = cka[cka["mode"] == mode].set_index(["method", "rank_mode"])
        for x, fit in enumerate(fits):
            row = sub.loc[(fit[0], fit[1])]
            null = max(row["perm_q95"], row["gauss_q95"])
            ax.plot(
                [x - 0.25, x + 0.25],
                [null] * 2,
                color=style.COLORS["reference"],
                lw=style.LINE_WIDTH,
            )
            ax.scatter(
                x,
                row["observed"],
                s=style.POINT_SIZE_FOREGROUND,
                zorder=3,
                color=style.COLORS["emphasis"] if row["above_null"] else "white",
                edgecolor=style.COLORS["emphasis"],
            )
        ax.set_xticks(
            range(len(fits)), [fit_label(f) for f in fits], rotation=30, ha="right"
        )
        ax.set_ylim(0, 1)
        ax.set_title(MODE_LABELS[mode])
    axes[0].set_ylabel("Linear CKA with cccRISE")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
