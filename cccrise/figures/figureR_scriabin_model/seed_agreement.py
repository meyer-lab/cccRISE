"""Seed-to-seed agreement of Scriabin's merged interaction programs, every
dataset under each preprocessing, against cccRISE's components' LR sets:
each point a pair of seeds, each bar their median."""

import numpy as np
import pandas as pd

from .. import common, style
from . import _common as C

AXES = ("tall_rect",)
#: Row of each Scriabin run (top axes) and its offset within the row
SCRIABIN_ROWS = {
    ("alad", "sample", "cccrise"): (3.0, 0.25),
    ("alad", "sample", "sct_alra"): (3.0, 0.0),
    ("alad", "sample", "sct"): (3.0, -0.25),
    ("ibd", "sample", "cccrise"): (2.0, 0.0),
    ("panc8", "sample", "cccrise"): (1.0, 0.15),
    ("panc8", "sample", "sct_alra"): (1.0, -0.15),
    ("panc8", "celltype", "cccrise"): (0.0, 0.15),
    ("panc8", "celltype", "sct_alra"): (0.0, -0.15),
}
CCCRISE_ROWS = {"alad": 1.0, "ibd": 0.0}
#: Each axes' heading, its y (data units) and y limits; their heights in
#: the panel follow the limits, so rows are evenly spaced in both
AXES_ROWS = {"Scriabin": (3.75, (-0.45, 4.05)), "cccRISE": (1.65, (-0.45, 1.95))}
GAP = 0.04  # between the two axes, a fraction of the panel's height
LEGEND_X = -0.3  # the legend's left edge: over the row labels, clear of the letter


def _strip(ax, y: float, values: pd.Series, color: str, rng) -> None:
    v = values.to_numpy(dtype=float)
    ax.scatter(
        v,
        y + rng.uniform(-0.08, 0.08, len(v)),
        s=style.POINT_SIZE_BACKGROUND,
        color=color,
        alpha=0.6,
        linewidths=0,
        zorder=2,
    )
    ax.plot(
        [np.median(v)] * 2,
        [y - 0.17, y + 0.17],
        color=color,
        lw=1.6,
        solid_capstyle="butt",
        zorder=3,
    )


def _axes(host):
    spans = {k: lim[1] - lim[0] for k, (_, lim) in AXES_ROWS.items()}
    unit = (1 - GAP) / sum(spans.values())
    low = host.inset_axes((0, 0, 1, spans["cccRISE"] * unit))
    top = host.inset_axes(
        (0, 1 - spans["Scriabin"] * unit, 1, spans["Scriabin"] * unit), sharex=low
    )
    return top, low


def _rows(table: pd.DataFrame) -> pd.DataFrame:
    """The plotted rows: Scriabin's programs and cccRISE's LR sets, with
    their axes, y and preprocessing."""
    parts = []
    for run, (y, dy) in SCRIABIN_ROWS.items():
        rows = C.run_rows(table, *run, method="Scriabin")
        parts.append(
            rows[rows.Stage == "programs"].assign(Axes="Scriabin", Y=y + dy, Row=y)
        )
    for dataset, y in CCCRISE_ROWS.items():
        rows = C.run_rows(table, dataset, "sample", "cccrise", method="cccRISE")
        parts.append(rows.assign(Axes="cccRISE", Y=y, Row=y))
    return pd.concat(parts, ignore_index=True)


def _decorate(axes: dict, rows: pd.DataFrame) -> None:
    for name, ax in axes.items():
        labels = rows[rows.Axes == name].drop_duplicates("Row")
        ax.set_yticks(
            labels.Row,
            [
                C.DATASET_LABELS[(d, g)]
                for d, g in zip(labels.Dataset, labels.Grouping, strict=True)
            ],
        )
        y, lim = AXES_ROWS[name]
        ax.set_ylim(*lim)
        ax.text(
            -0.03,
            y,
            name,
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=style.FONT_SIZE_TICK,
            transform=ax.get_yaxis_transform(),
        )
    low = axes["cccRISE"]
    low.set_xlim(0, 1.02)
    low.set_xticks([0, 0.25, 0.5, 0.75, 1])
    low.set_xlabel("Seed-to-seed agreement (LR-set Jaccard)")
    axes["Scriabin"].tick_params(labelbottom=False)


def _legend(host) -> None:
    """Preprocessing colors, each pointing to its LR resource; the two
    SCTransform pipelines share OmniPath through a brace."""
    lax = host.inset_axes((0, 1.04, 1, 0.19))
    lax.set_axis_off()
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    ys = {"cccrise": 0.84, "sct_alra": 0.5, "sct": 0.16}
    texts = {}
    for key, y in ys.items():
        lax.plot(
            [LEGEND_X + 0.012],
            [y],
            ls="",
            marker="o",
            ms=style.MARKER_SIZE,
            color=C.PREPROCESSING_COLORS[key],
            clip_on=False,
        )
        texts[key] = lax.text(
            LEGEND_X + 0.04,
            y,
            C.PREPROCESSING_LABELS[key],
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
        )
    lax.figure.canvas.draw()
    ends = {k: C.extent(t, lax)[1, 0] for k, t in texts.items()}
    arrow = "→  "
    lax.text(
        ends["cccrise"] + 0.025,
        ys["cccrise"],
        arrow + C.LR_RESOURCES["cccrise"],
        va="center",
        fontsize=style.FONT_SIZE_DENSE,
    )
    col = max(ends["sct_alra"], ends["sct"]) + 0.05
    C.brace(
        lax,
        ys["sct_alra"] + 0.13,
        ys["sct"] - 0.13,
        col,
        0.03,
        vertical=True,
        transform=lax.transAxes,
    )
    lax.text(
        col + 0.05,
        (ys["sct_alra"] + ys["sct"]) / 2,
        arrow + C.LR_RESOURCES["sct"],
        va="center",
        fontsize=style.FONT_SIZE_DENSE,
    )


def _source_data(rows: pd.DataFrame) -> None:
    frame = pd.DataFrame(
        {
            "Method": rows.Method,
            "Dataset": [
                C.DATASET_LABELS[(d, g)]
                for d, g in zip(rows.Dataset, rows.Grouping, strict=True)
            ],
            "Preprocessing": rows.Preprocessing.map(
                {
                    "cccrise": "cccRISE's",
                    "sct_alra": "SCTransform + ALRA",
                    "sct": "SCTransform",
                }
            ),
            "LR resource": rows.Preprocessing.map(C.LR_RESOURCES),
            "Seed A": rows.SeedA,
            "Seed B": rows.SeedB,
            "Agreement": rows.Value,
        }
    )
    common.source_data(
        "lr_set_agreement_by_seed_pair",
        frame,
        "Agreement between two runs with different random seeds, one row per seed pair. Scriabin: "
        "best-match Jaccard (mean over both directions) of the interaction programs merged across the "
        "subsampled samples or cell types (more than 300 cells; smaller ones are identical across seeds), "
        "upstream R at its published defaults. cccRISE: best-match Jaccard of each component's carrying "
        "LR pairs (90% of its loading) between seed refits. Bars: medians over seed pairs.",
    )


def plot(axs):
    host = axs[0]
    host.set_axis_off()
    top, low = _axes(host)
    axes = {"Scriabin": top, "cccRISE": low}
    rows = _rows(C.seed_agreement())
    rng = np.random.default_rng(common.JITTER_SEED)
    for _, g in rows.groupby(["Axes", "Y", "Preprocessing"], sort=False):
        first = g.iloc[0]
        _strip(
            axes[first.Axes],
            first.Y,
            g.Value,
            C.PREPROCESSING_COLORS[first.Preprocessing],
            rng,
        )
    _decorate(axes, rows)
    _legend(host)
    _source_data(rows)


def makeFigure():
    return common.standalone_fixed(__name__)
