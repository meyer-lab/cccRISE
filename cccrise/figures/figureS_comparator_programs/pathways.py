"""CellChat pathways of each CP fit's components (coarse cell types): the
distinct pathways among its five largest |LR loadings|, with the effective
number of samples carrying it."""

from .. import common, style
from ._common import SHORT, load

FITS = ("cccRISE", "Pseudobulk CPD", "Tensor-cell2cell")


def plot(fig):
    table = load()
    table = table[(table["granularity"] == "coarse") & table["method"].isin(FITS)]
    lr = table[table["mode"] == "lr"].set_index(["method", "component"])["top_pathways"]
    samples = table[table["statistic"] == "sample_breadth"].set_index(
        ["method", "component"]
    )["value"]
    rows = [
        [
            f"{SHORT.get(m, m)} {k}",
            f"{samples.loc[(m, k)]:.1f}",
            lr.loc[(m, k)],
        ]
        for m in FITS
        for k in sorted(lr.loc[m].index)
    ]
    ax = fig.subplots()
    ax.set_axis_off()
    cells = ax.table(
        cellText=rows,
        colLabels=["component", "samples", "top-5 LR pathways"],
        colWidths=[0.2, 0.13, 0.67],
        loc="center",
        cellLoc="left",
    )
    cells.auto_set_font_size(False)
    cells.set_fontsize(style.FONT_SIZE_DENSE - 1)
    for (r, _c), cell in cells.get_celld().items():
        cell.set_linewidth(0)
        if r > 0 and rows[r - 1][0].startswith("cccRISE 4"):
            cell.set_text_props(color=style.COLORS["emphasis"], weight="bold")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
