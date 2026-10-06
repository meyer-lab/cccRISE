"""Per component of every fit: the |loading|^2-weighted prevalence of its LR
pairs across (sample, sender, receiver) blocks, the effective number of
samples carrying it, and its strongest clinical association. Comparator
components are open where cccRISE does not recover them (not contained in
cccRISE's span); cccRISE's C4, the clinically associated component, is
labelled."""

import numpy as np

from .. import common, style
from ._common import fit_label, fits, load

STATISTICS = {
    "lr_prevalence": "LR prevalence",
    "sample_breadth": "effective no. of samples",
    "max_eta_squared": r"max $\eta^2$ (13 associations)",
}


def plot(fig):
    table = load()
    order = fits(table)
    axes = fig.subplots(1, len(STATISTICS))
    rng = np.random.default_rng(0)
    for ax, (statistic, label) in zip(axes, STATISTICS.items(), strict=True):
        sub = table[table["statistic"] == statistic]
        for x, (method, granularity) in enumerate(order):
            rows = sub[(sub["method"] == method) & (sub["granularity"] == granularity)]
            ours = method == "cccRISE"
            recovered = rows["recovered"].eq(True).to_numpy()
            jitter = rng.uniform(-0.15, 0.15, len(rows))
            color = style.COLORS["emphasis"] if ours else style.COLORS["ink"]
            ax.scatter(
                x + jitter,
                rows["value"],
                s=12,
                edgecolor=color,
                facecolor=np.where(ours | recovered, color, "white"),
                linewidth=0.8,
                zorder=3,
            )
            if ours:
                c4 = rows[rows["component"] == 4]
                ax.annotate(
                    "C4",
                    (x, float(c4["value"].iloc[0])),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=style.FONT_SIZE_DENSE,
                    color=style.COLORS["emphasis"],
                    va="center",
                )
        ax.set_xticks(
            range(len(order)),
            [fit_label(m, g, table) for m, g in order],
            rotation=90,
            fontsize=style.FONT_SIZE_DENSE - 1,
        )
        ax.set_title(label)
        ax.grid(False)
        if statistic == "sample_breadth":
            ax.set_yscale("log")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
