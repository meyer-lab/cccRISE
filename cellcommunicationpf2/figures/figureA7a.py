"""
Figure A7a: Overview of the independent ALAD airway brushing dataset.
68,140 cells from 20 patients across 4 ALAD groups.
"""

import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from .common import getSetup, subplotLabel
from ..import_data import import_alad_brush


def makeFigure():
    ax, f = getSetup((14, 4), (1, 3))
    subplotLabel(ax)

    X = import_alad_brush()

    alad_order = ["Control", "ALAD int", "ALAD recovered", "ALAD declined"]
    palette = dict(zip(alad_order, sns.color_palette("Set2", len(alad_order))))

    # (a) Cell type composition
    ct_counts = X.obs["cell_type_inintegrated"].value_counts().sort_values()
    ct_counts.plot.barh(ax=ax[0], color=sns.color_palette("colorblind", len(ct_counts)))
    ax[0].set_xlabel("Number of Cells")
    ax[0].set_ylabel("")
    ax[0].set_title("Cell Type Composition")

    # (b) Cells per patient colored by ALAD group
    patient_df = (
        X.obs.groupby("subj", observed=True)
        .agg(n_cells=("subj", "size"), ALAD=("ALAD", "first"))
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    ax[1].bar(
        range(len(patient_df)),
        patient_df["n_cells"],
        color=[palette[a] for a in patient_df["ALAD"]],
    )
    ax[1].set_xticks(range(len(patient_df)))
    ax[1].set_xticklabels(patient_df["subj"], rotation=90, fontsize=5)
    ax[1].set_ylabel("Number of Cells")
    ax[1].set_title("Cells per Patient")
    handles = [Patch(facecolor=palette[k], label=k) for k in alad_order]
    ax[1].legend(handles=handles, fontsize=5)

    # (c) Cell type proportions per ALAD group
    prevalence_order = X.obs["cell_type_inintegrated"].value_counts().index.tolist()
    ct = pd.crosstab(
        X.obs["ALAD"], X.obs["cell_type_inintegrated"], normalize="index"
    )
    ct = ct.reindex(alad_order)[prevalence_order]
    ct_colors = sns.color_palette("tab20", ct.shape[1])
    ct.plot.bar(stacked=True, ax=ax[2], color=ct_colors, width=0.7)
    ax[2].set_ylabel("Proportion")
    ax[2].set_xlabel("")
    ax[2].set_title("Cell Type by ALAD Group")
    ax[2].set_xticklabels(
        ax[2].get_xticklabels(), rotation=45, ha="right", fontsize=6
    )
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels, fontsize=4, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)

    return f
