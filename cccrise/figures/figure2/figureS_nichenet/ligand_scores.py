"""NicheNet target score of every ligand the model sees, per CP component
(``nichenet.component_ligand_scores``): the AUROC of the component's receiver
ranking for the ligand's downstream targets, targets shared by many ligands
down-weighted. Points are colored and sized by the ligand's weight in the
component, opaque for the top ligands (``nichenet.top_pair_ligands``) and
translucent for the rest."""

import numpy as np
from matplotlib.colors import LogNorm

from ... import common, style
from ..nichenet import BOX_STYLE, load_scores

WEIGHT_FLOOR = 1e-3
JITTER = 0.22
POINT_SIZE_MAX = 8.4
BACKGROUND_ALPHA = 0.5  # non-top ligands, so the boxes show through
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    scores = load_scores().dropna(subset=["auroc"])
    components = sorted(scores["component"].unique())
    rng = np.random.default_rng(0)
    for x, component in enumerate(components):
        points = _component_points(scores[scores["component"] == component], x, ax, rng)
    ax.boxplot(
        [scores.loc[scores["component"] == c, "auroc"] for c in components],
        positions=range(len(components)),
        zorder=1,
        **BOX_STYLE,
    )
    common.reference_line(ax, 0.5)
    ax.set_xticks(range(len(components)), components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    ax.set_xlabel("CP component")
    ax.set_ylabel("NicheNet target score")
    common.colorbar(ax, points, "Relative ligand weight")


def _component_points(component_scores, x, ax, rng):
    """Weight-colored points for one component's ligands, the non-top ones
    translucent; returns the (opaque) top ligands' points."""
    c = component_scores.assign(
        w=np.clip(component_scores["rel_weight"], WEIGHT_FLOOR, 1)
    ).sort_values("w")
    c["x"] = x + rng.uniform(-JITTER, JITTER, len(c))
    for top, alpha in ((False, BACKGROUND_ALPHA), (True, 1.0)):
        subset = c[c["is_top"] == top]
        points = ax.scatter(
            subset["x"],
            subset["auroc"],
            c=subset["w"],
            cmap=style.CMAPS["weight"],
            norm=LogNorm(vmin=WEIGHT_FLOOR, vmax=1),
            s=style.POINT_SIZE_BACKGROUND / 2 + POINT_SIZE_MAX * subset["w"],
            linewidths=0,
            alpha=alpha,
            zorder=3 + top,
        )
    return points


def makeFigure():
    return common.standalone_fixed(__name__)
