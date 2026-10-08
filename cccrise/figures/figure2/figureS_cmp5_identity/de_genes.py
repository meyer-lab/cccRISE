"""The genes most enriched in the sender and in the receiver group, each
against the rest of the cells (Wilcoxon, `S.rank_genes`): the `N_TOP` highest
scoring with q < `common.SIG_ALPHA` that the group expresses, split into those
shared by both groups and those particular to one."""

from ... import common
from .. import _common as S

N_TOP = 10
MIN_DETECTION = 0.1  # fraction of a group's cells detecting a gene to call it expressed
AXES = ("strip",)


def plot(axs):
    (ax,) = axs
    S.plot_dotplot(ax, gene_sets(), show_legend=True)


def gene_sets() -> dict[str, list[str]]:
    """{"Shared", "Sender", "Receiver": top genes}."""
    ranked = S.rank_genes(S.DE_GROUPS, "rest")
    top = {g: _top_genes(ranked[g], g) for g in ("sender", "receiver")}
    shared = [g for g in top["sender"] if g in top["receiver"]]
    return {
        "Shared": shared,
        "Sender": [g for g in top["sender"] if g not in shared],
        "Receiver": [g for g in top["receiver"] if g not in shared],
    }


def _top_genes(ranking, group: str) -> list[str]:
    candidates = ranking[ranking["q"] < common.SIG_ALPHA].index[: 10 * N_TOP].tolist()
    detected, _ = S.group_expression(candidates)
    return [g for g in candidates if detected.loc[g, group] >= MIN_DETECTION][:N_TOP]


def makeFigure():
    return common.standalone_fixed(__name__)
