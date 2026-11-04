"""Merged-program agreement under each subsampling (the published number of
subsamples, 512, or the full cell-cell interaction matrix) and settings
(the published defaults, or tuned): between seeds (top) and with one sample
or cell type left out (bottom)."""

import itertools

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .. import common, style
from . import _common as C

AXES = ("tall_mid",)
SUBSAMPLING = {
    "published": "Published\nsubsampling",
    "512": "512\nsubsamples",
    "full": "Full\nCCIM",
}
SETTINGS = {"default": "Default", "tuned": "Tuned"}
#: Bar positions, one per (subsampling, settings), and bar width
X = np.array([0, 1, 2.6, 3.6, 5.2, 6.2])
WIDTH = 0.38
ROWS = {"seed": "Seed-to-seed\nagreement", "leave_one_out": "Leave-one-out\nagreement"}
HSPACE = 0.08  # between the two axes, a fraction of the panel's height
HATCH = "////"


def summaries(table: pd.DataFrame) -> dict:
    """Each (subsampling, settings, measure)'s bar: the mean over seed
    pairs, or over seeds of the median over left-out groups; 1 for the full
    CCIM's seed agreement, which involves no randomness."""
    out = {}
    for sub, st in itertools.product(SUBSAMPLING, SETTINGS):
        rows = table[(table.Subsampling == sub) & (table.Settings == st)]
        seed = rows[rows.Measure == "seed"].Value
        out[(sub, st, "seed")] = 1.0 if sub == "full" else float(seed.mean())
        loo = rows[rows.Measure == "leave_one_out"]
        out[(sub, st, "leave_one_out")] = float(
            loo.groupby("SeedA", dropna=False).Value.median().mean()
        )
    return out


def _bars(axes: dict, bars: dict) -> None:
    for k, dataset in enumerate(C.REMEDY_DATASETS):
        for measure, ax in axes.items():
            values = [
                bars[dataset][(sub, st, measure)]
                for sub, st in itertools.product(SUBSAMPLING, SETTINGS)
            ]
            drawn = ax.bar(
                X + (k - 0.5) * (WIDTH + 0.02),
                values,
                WIDTH,
                color=C.REMEDY_COLORS[dataset],
            )
            if measure == "seed":
                for bar, sub in zip(
                    drawn, np.repeat(list(SUBSAMPLING), len(SETTINGS)), strict=True
                ):
                    if sub == "full":
                        bar.set_hatch(HATCH)
                        bar.set_edgecolor("white")
                        bar.set_linewidth(0)


def _decorate(axes: dict) -> None:
    for measure, ax in axes.items():
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(ROWS[measure])
        ax.set_xlim(-0.6, 6.8)
        ax.set_xticks(X)
    axes["seed"].set_xticklabels([])
    low = axes["leave_one_out"]
    low.set_xticklabels(
        list(SETTINGS.values()) * len(SUBSAMPLING),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    axes["seed"].legend(
        handles=[
            Patch(color=C.REMEDY_COLORS[d], label=label)
            for d, label in C.REMEDY_DATASETS.items()
        ]
        + [
            Patch(
                facecolor=style.CATEGORICAL[7],
                hatch=HATCH,
                edgecolor="white",
                label="No randomness",
            )
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncols=3,
        fontsize=style.FONT_SIZE_DENSE,
        handlelength=0.9,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0,
    )


def _group_braces(ax) -> None:
    """A brace under each subsampling's two settings, pointing to its name."""
    ax.figure.canvas.draw()
    labels = ax.get_xticklabels()
    to_data = ax.transData.inverted()
    renderer = ax.figure.canvas.get_renderer()
    for i, name in enumerate(SUBSAMPLING.values()):
        ext = [labels[k].get_window_extent(renderer) for k in (2 * i, 2 * i + 1)]
        (x0, _), (x1, _) = to_data.transform(
            [[min(e.x0 for e in ext), 0], [max(e.x1 for e in ext), 0]]
        )
        y = (
            ax.transAxes.inverted().transform([[0, min(e.y0 for e in ext)]])[0, 1]
            - 0.03
        )
        mid = (x0 + x1) / 2
        tip = C.brace(
            ax, mid - 0.75, mid + 0.75, y, -0.07, transform=ax.get_xaxis_transform()
        )
        ax.text(
            mid,
            tip - 0.015,
            name,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=style.FONT_SIZE_DENSE,
            linespacing=1.1,
        )


def _source_data(tables: dict) -> None:
    frame = pd.concat(
        [t.assign(Dataset=C.REMEDY_DATASETS[d]) for d, t in tables.items()]
        + [
            pd.DataFrame(
                {
                    "Dataset": label,
                    "Subsampling": "full",
                    "Settings": list(SETTINGS),
                    "Measure": "seed",
                    "Value": 1.0,
                }
            )
            for label in C.REMEDY_DATASETS.values()
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "Dataset": frame.Dataset,
            "Subsampling": frame.Subsampling.map(
                lambda s: SUBSAMPLING[s].replace("\n", " ")
            ),
            "Settings": frame.Settings.map(SETTINGS),
            "Measure": frame.Measure.map(
                {"seed": "Seed to seed", "leave_one_out": "Leave one out"}
            ),
            "Seed A": frame.SeedA.astype("Int64"),
            "Seed B": frame.SeedB.astype("Int64"),
            "Left-out group": frame.Group,
            "Agreement": frame.Value,
        }
    )
    common.source_data(
        "program_agreement_by_condition",
        frame,
        "Best-match Jaccard of Scriabin's merged programs, computed with our implementation (equal to "
        "upstream R wherever no randomness is involved). ALAD: 47 samples, cccRISE's inputs; panc8: the "
        "5 cell types of more than 300 cells, the authors' preprocessing. Subsampling: the published "
        "number of 300-cell subsamples, 512, or the full cell-cell interaction matrix (CCIM). Settings: "
        "Default (published: scale-free R^2 0.6, soft-power cap 3, cut-height quantile 0.5, merge Jaccard "
        "0.15) or Tuned (R^2 0.5, no cap, quantile 0.25, merge 0.4). Seed to seed: between the runs of "
        "Seed A and Seed B; bar = mean over pairs. With the full CCIM there is no randomness, so it is 1 "
        "by construction (rows without seeds). Leave one out: the merge without the left-out group "
        "against the merge of all groups, at Seed A (none for the full CCIM); bar = mean over seeds of "
        "the median over groups.",
    )


def plot(axs):
    host = axs[0]
    host.set_axis_off()
    h = (1 - HSPACE) / 2
    axes = {
        "seed": host.inset_axes((0, 1 - h, 1, h)),
        "leave_one_out": host.inset_axes((0, 0, 1, h)),
    }
    tables = {d: C.load("remedy", d) for d in C.REMEDY_DATASETS}
    _bars(axes, {d: summaries(t) for d, t in tables.items()})
    _decorate(axes)
    _group_braces(axes["leave_one_out"])
    _source_data(tables)


def makeFigure():
    return common.standalone_fixed(__name__)
