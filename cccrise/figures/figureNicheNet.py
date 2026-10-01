"""
Figure: each CPD component's top ligands against every other ligand the model
sees, plotted from ``nichenet_scores_<dataset>.parquet`` written by
``uv run cccrise-nichenet --dataset <dataset>``.

One box per component: the NicheNet-style AUROC
(``nichenet.component_ligand_scores``) of every scored ligand on the component's
receiver ranking, restricted to genes expressed in the receiver groups. Points
are colored and sized by the ligand's weight in the component; the top
ligands (``nichenet.top_pair_ligands``) are outlined and labeled. Above each
box: a two-sided Mann-Whitney test of the top ligands against all the others,
and its Benjamini-Hochberg q across components (``nichenet.top_ligand_tests``).
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from ..runners.datasets import DATASETS, VALIDATION_DIR
from ..validation.nichenet import top_ligand_tests
from .common import getSetup, load_runner_output

SELECTED_DATASET = "alad"

ALPHA = 0.05
WEIGHT_FLOOR = 1e-3
LABEL_GAP = 0.035  # fraction of the y-range between stacked labels
INK, MUTED = "#2b2b29", "#6f6e69"
# light until weights are substantial, so only heavy ligands read as dark
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weight",
    [
        (0, "#eef2f7"),
        (0.45, "#cde2fb"),
        (0.7, "#86b6ef"),
        (0.85, "#2a78d6"),
        (1, "#0d366b"),
    ],
)


def _spread(values: np.ndarray, gap: float) -> np.ndarray:
    """Label heights at least `gap` apart, in the input order."""
    order = np.argsort(values)
    placed = values[order].astype(float).copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out


def plot_component(ax, c: pd.DataFrame, x: int, gap: float, rng) -> PathCollection:
    """Weight-colored points for one component's ligands, top ligands labeled."""
    c = c.assign(w=np.clip(c["rel_weight"], WEIGHT_FLOOR, 1)).sort_values("w")
    jitter = rng.uniform(-0.22, 0.22, len(c))
    top = c["is_top"].to_numpy()
    points = ax.scatter(
        x + jitter,
        c["auroc"],
        c=c["w"],
        cmap=WEIGHT_CMAP,
        norm=LogNorm(vmin=WEIGHT_FLOOR, vmax=1),
        s=5 + 60 * c["w"],
        lw=np.where(top, 1.0, 0.2),
        edgecolor=np.where(top, INK, "white"),
        zorder=3,
    )
    labeled = c[top]
    heights = _spread(labeled["auroc"].to_numpy(), gap)
    for lig, a, xi, y in zip(
        labeled["ligand"], labeled["auroc"], x + jitter[top], heights
    ):
        ax.plot([xi, x + 0.3], [a, y], color=MUTED, lw=0.4, zorder=2)
        ax.text(x + 0.31, y, lig, fontsize=5.5, va="center", color=INK, zorder=4)
    return points


def makeFigure():
    spec = DATASETS[SELECTED_DATASET]
    scores = load_runner_output(
        spec.nichenet_scores_filename(),
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {SELECTED_DATASET}",
    ).dropna(subset=["auroc"])
    tests = top_ligand_tests(scores).set_index("component")
    components = sorted(scores["component"].unique())

    ax, f = getSetup((8.5, 5.2), (1, 1))
    ax = ax[0]
    rng = np.random.default_rng(0)
    top_y, low_y = scores["auroc"].max(), scores["auroc"].min()
    span = top_y - low_y
    for x, cmp in enumerate(components):
        points = plot_component(
            ax, scores[scores["component"] == cmp], x, LABEL_GAP * span, rng
        )
        t = tests.loc[cmp]
        sig = t["q"] < ALPHA
        ax.text(
            x,
            top_y + 0.09 * span,
            f"{'↑' if t['direction'] == 'up' else '↓'} p={t['p']:.2g}\nq={t['q']:.2g}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=INK if sig else MUTED,
            fontweight="bold" if sig else "normal",
        )
    ax.boxplot(
        [scores.loc[scores["component"] == cmp, "auroc"] for cmp in components],
        positions=range(len(components)),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        zorder=1,
        boxprops={"facecolor": "#f2f1ee", "edgecolor": MUTED, "lw": 0.8},
        medianprops={"color": INK, "lw": 1.2},
        whiskerprops={"color": MUTED, "lw": 0.8},
        capprops={"color": MUTED, "lw": 0.8},
    )
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.7)
    ax.set_xticks(
        range(len(components)),
        [
            f"Cmp{c}\n({int(tests.loc[c, 'n_top'])} top / {int(tests.loc[c, 'n_comparison'])})"
            for c in components
        ],
    )
    ax.set_xlim(-0.6, len(components) - 0.1)
    ax.set_ylim(top=top_y + 0.16 * span)
    ax.set_ylabel("Ligand AUROC (receiver-expressed genes)")
    ax.set_title(
        "Top ligands (outlined) vs all other ligands; two-sided Mann–Whitney, "
        f"q = BH across components (bold q<{ALPHA})",
        fontsize=7,
    )
    cbar = f.colorbar(points, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Ligand weight / component's top scorable ligand", fontsize=7)

    print(f"=== NicheNet top ligands vs all ligands: {SELECTED_DATASET} ===")
    print(tests.round(4).to_string())
    return f
