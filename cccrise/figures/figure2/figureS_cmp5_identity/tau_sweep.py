"""Signed Wilcoxon rank-sum q of each Figure 2 epithelial subtype marker
between sender and receiver cells across group cutoffs (tau).

Robustness of Fig. 2M's sender vs. receiver subtype markers to the group
cutoff: each shown marker's Wilcoxon q, signed by the group it is higher in,
against the cutoff (`S.split_cells`'s tau), the markers fixed at those shown
at `S.GROUP_TAU`."""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from ....runners.run_alad_programs import DE_TAUS
from ... import common, style
from .. import _common as S
from ..epithelial_markers import stratifying_markers
from . import _common as V

AXES = ("square",)
TAUS = DE_TAUS  # the cutoffs `cccrise-alad-programs` tests at
SECTION_COLORS = dict(
    zip(
        ("Ciliated", "Secretory", "Basal"),
        (style.CATEGORICAL[2], style.CATEGORICAL[4], style.CATEGORICAL[5]),
        strict=True,
    )
)


def plot(axs):
    (ax,) = axs
    sweep = tau_sweep()
    for _, rows in sweep.groupby(["Section", "Gene"], sort=False):
        section = rows["Section"].iat[0]
        ax.plot(
            rows["tau"],
            rows["Signed -log10 q"],
            color=SECTION_COLORS[section],
            lw=style.LINE_WIDTH_THIN,
            marker="o",
            markersize=style.MARKER_SIZE / 2,
            label=section,
        )
    handles, labels = ax.get_legend_handles_labels()
    by_section = dict(zip(labels, handles, strict=True))
    q_line, _ = (
        common.reference_line(ax, y, label=f"q = {common.SIG_ALPHA}")
        for y in (-np.log10(common.SIG_ALPHA), np.log10(common.SIG_ALPHA))
    )
    common.reference_line(ax, 0, linestyle="-")
    tau_line = common.reference_line(
        ax, S.GROUP_TAU, horizontal=False, linestyle=":", label=f"τ = {S.GROUP_TAU}"
    )
    ax.set(
        xlabel="Group cutoff (τ)",
        ylabel="−log$_{10}$ q, signed\n(+ sender, − receiver)",
    )
    V.hang_above(
        ax,
        handles=list(by_section.values()),
        labels=list(by_section),
        handlelength=1.0,
        columnspacing=0.8,
    )
    ax.add_artist(ax.get_legend())
    common.legend(ax, "right", handles=[tau_line, q_line], handlelength=1.5)
    common.source_data(
        "tau_sweep_sender_vs_receiver",
        sweep,
        "One row per Fig. 2M epithelial subtype marker and group cutoff tau: the "
        "sender and receiver group sizes at tau, the two-sided Wilcoxon rank-sum "
        "test of sender against receiver cells' normalized expression (p; q "
        "Benjamini-Hochberg adjusted over the markers shown in Fig. 2M), and "
        "-log10 q signed positive if higher in the sender group.",
    )


def tau_sweep() -> pd.DataFrame:
    """Sender vs. receiver Wilcoxon p and BH q, over Fig. 2M's subtype
    markers, of each of those markers at each of `TAUS`."""
    sections = stratifying_markers()
    family = [g for genes in sections.values() for g in genes]
    sender, receiver = S.component_weights()
    rows = []
    for tau in TAUS:
        sizes = {g: m.sum() for g, m in S.split_cells(sender, receiver, tau).items()}
        test = S.rank_genes(("sender",), "receiver", tau)["sender"].loc[family]
        q = pd.Series(false_discovery_control(test["p"]), index=family)
        rows += [
            {
                "tau": tau,
                "Section": section,
                "Gene": g,
                "n sender": sizes["sender"],
                "n receiver": sizes["receiver"],
                "Wilcoxon score": test.loc[g, "score"],
                "p": test.loc[g, "p"],
                "q": q[g],
                "Signed -log10 q": np.sign(test.loc[g, "score"]) * -np.log10(q[g]),
            }
            for section, genes in sections.items()
            for g in genes
        ]
    return pd.DataFrame(rows)


def makeFigure():
    return common.standalone_fixed(__name__)
