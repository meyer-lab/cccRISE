"""NicheNet target score of every ligand the model sees, per CP component
(``nichenet.component_ligand_scores``): the AUROC of the component's receiver
ranking for the ligand's downstream targets, targets shared by many ligands
down-weighted. Points are colored and sized by the ligand's weight in the
component, opaque for the top ligands (``nichenet.top_pair_ligands``) and
translucent for the rest."""

import numpy as np

from ... import common
from ..nichenet import BOX_STYLE, ligand_points, load_scores

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    scores = load_scores().dropna(subset=["auroc"])
    common.source_data(
        "nichenet_target_score_per_ligand",
        scores[
            ["component", "ligand", "auroc", "weight", "rel_weight", "is_top"]
        ].rename(
            columns={
                "component": "CP component",
                "ligand": "Ligand",
                "auroc": "NicheNet target score (AUROC)",
                "weight": "Ligand weight",
                "rel_weight": "Relative ligand weight",
                "is_top": "Top ligand",
            }
        ),
        "One row per ligand NicheNet scores and CP component: the AUROC of the "
        "component's receiver ranking for the ligand's downstream targets; the "
        "ligand's weight in the component (and relative to the component's "
        "largest, plotted floored at 1e-3); whether it is one of the "
        "component's top ligands.",
    )
    components = sorted(scores["component"].unique())
    rng = np.random.default_rng(0)
    for x, component in enumerate(components):
        points = ligand_points(ax, scores[scores["component"] == component], x, rng)
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


def makeFigure():
    return common.standalone_fixed(__name__)
