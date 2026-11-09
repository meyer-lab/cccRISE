"""NicheNet on cccRISE and every comparator column of ALAD
(``uv run cccrise-comparator-nichenet``)."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import COMPARATORS, DATASET, METHOD_COLORS, load
from ._data import LABELLINGS

SLOTS = [("cccRISE", "coarse", "CCC-RISE")] + [
    (m, lab, f"{m} {lab[0].upper()}") for m in COMPARATORS for lab in LABELLINGS
]
#: Each program, then its closest columns (method, labelling, column).
CLOSEST = [
    (4, [("STAC", "fine", 14), ("STAC", "coarse", 12), ("STAC", "coarse", 15)]),
    (5, [("STAC", "fine", 10), ("STAC", "coarse", 4)]),
    (3, [("STAC", "coarse", 9), ("STAC", "coarse", 16), ("STAC", "fine", 5)]),
    (
        2,
        [
            ("PB", "coarse", 4),
            ("PB", "fine", 4),
            ("STAC", "coarse", 6),
            ("STAC", "fine", 1),
        ],
    ),
]
SCORE_LABEL = "Weighted target score\nabove shuffled cells"


def table() -> pd.DataFrame:
    t = load("nichenet", dataset=DATASET)
    return t.assign(Effect=t["weighted_auroc"] - t["null_mean_cell"])


def _strip(ax, t: pd.DataFrame, value: str, q: str, reference: float) -> None:
    rng = np.random.default_rng(2)
    for i, (method, lab, _) in enumerate(SLOTS):
        g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
        xs = i + rng.uniform(-0.2, 0.2, len(g))
        hit = (g[q] < common.SIG_ALPHA).to_numpy()
        color = METHOD_COLORS[method]
        ax.scatter(
            xs[~hit],
            g[value][~hit],
            s=9,
            facecolor="white",
            edgecolor=color,
            lw=0.6,
            zorder=3,
        )
        ax.scatter(xs[hit], g[value][hit], s=13, color=color, lw=0, zorder=4)
        if method == "cccRISE":
            for x, y, c in zip(xs, g[value], g["Component"], strict=True):
                ax.text(
                    x + 0.09,
                    y,
                    f"C{c}",
                    va="center",
                    fontsize=style.FONT_SIZE_DENSE - 0.7,
                )
    ax.axhline(reference, color=style.COLORS["reference"], ls="--", lw=0.6, zorder=1)
    ax.set_xticks(
        range(len(SLOTS)),
        [s[2] for s in SLOTS],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE,
    )


def _source(name: str, t: pd.DataFrame, columns: list, description: str) -> None:
    keys = ["Method", "Labelling", "Component", "Rule", "High cells"]
    common.source_data(name, t[keys + columns], description)


def cell_shuffle_strip(ax) -> None:
    t = table()
    _source(
        "nichenet_cell_shuffle_by_column",
        t,
        ["weighted_auroc", "null_mean_cell", "Effect", "p_cell", "q_cell"],
        "Each column's ligand-weighted mean target score, its mean over rankings with "
        "the high and low labels shuffled, their difference, and the one-sided "
        "permutation p, Benjamini-Hochberg adjusted within the fit (q).",
    )
    _strip(ax, t, "Effect", "q_cell", 0)
    ax.set_ylabel(SCORE_LABEL)


def prior_shuffle_strip(ax) -> None:
    t = table()
    _source(
        "nichenet_prior_shuffle_by_column",
        t,
        ["weighted_pct", "null_mean_prior", "p_prior", "q_prior"],
        "Each column's ligand-weighted target-score percentile, its mean over copies "
        "of the prior with links reassigned, and the one-sided permutation p, "
        "Benjamini-Hochberg adjusted within the fit (q).",
    )
    _strip(ax, t, "weighted_pct", "q_prior", 0.5)
    ax.set_ylabel("Weighted score percentile\nof the fit's ligands")


def closest_strip(ax) -> None:
    t = table().set_index(["Method", "Labelling", "Component"])
    common.source_data(
        "nichenet_program_and_carrying_columns",
        pd.DataFrame(
            [
                {
                    "Program": f"C{program}",
                    "Method": method,
                    "Labelling": lab,
                    "Column": column,
                    "Effect": t.loc[(method, lab, column), "Effect"],
                }
                for program, columns in CLOSEST
                for method, lab, column in [("cccRISE", "coarse", program), *columns]
            ]
        ),
        "Each CCC-RISE program's ligand-weighted NicheNet target score above its "
        "mean over shuffled cells (Effect), beside the comparator LR columns "
        "carrying its LR pairs (`CLOSEST`); cccRISE rows are the program itself.",
    )
    x, ticks, names = 0.0, [], []
    for program, columns in CLOSEST:
        effect = t.loc[("cccRISE", "coarse", program), "Effect"]
        ax.scatter(
            x, effect, s=20, marker="D", color=METHOD_COLORS["cccRISE"], lw=0, zorder=4
        )
        ticks.append(x)
        names.append(f"C{program}")
        x += 1
        for method, lab, column in columns:
            effect = t.loc[(method, lab, column), "Effect"]
            ax.scatter(x, effect, s=11, color=METHOD_COLORS[method], lw=0, zorder=3)
            ticks.append(x)
            names.append(f"{method[0]}{column} {lab[0].upper()}")
            x += 1
        x += 0.7
    ax.axhline(0, color=style.COLORS["reference"], ls="--", lw=0.6)
    ax.set_xticks(
        ticks,
        names,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE - 0.5,
    )
    ax.set_ylabel(SCORE_LABEL)
