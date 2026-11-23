"""Figure 5's supplement on what the comparators find that CCC-RISE does not
(`figureS_comparator_missed`): each PB and STAC LR column's largest share of
its squared loading on one CCC-RISE component's carrying pairs (`share`; the
measure of 5C's related columns), both cohorts' bars in shared fit slots
(`slots`); and the carrying pairs of the columns under `MISSED` share that are
associated with an outcome (`significant`) and carry a signalling pair
CCC-RISE lacks (`lacking`), beside each pair's largest loading in any CCC-RISE
component (`boxed_contents`). A carrying pair of a column is one of the fewest
pairs carrying 90% of its squared loading (`covariation.carrying_pairs`), as
for the components. `associated` lists every column's and component's
associations at q < 0.05, which Figure 5E-F name."""

from functools import cache

import numpy as np
import pandas as pd

from ... import covariation
from .. import common, style
from ._common import METHOD_COLORS, cohort_bar, load
from ._data import LABELLINGS, short
from ._programs import lr_matrix
from ._recovery import LOADING_LABEL

#: The associations drawn, by their short names (not IBD's 3-way status).
ASSOCIATION_NAMES = {
    "Control vs. ALAD (All)": "ALAD",
    "ALAD Status (3-Way)": "ALAD status",
    "6-Month Outcome (4-Way)": "6-mo outcome",
    "6-Month CLAD vs. Stable/Recovered": "6-mo CLAD",
    "6-Month Died vs. Alive": "6-mo death",
    "1-Year Outcome (4-Way)": "1-yr outcome",
    "1-Year CLAD vs. No CLAD": "1-yr CLAD",
    "1-Year Died vs. Alive": "1-yr death",
    "Healthy vs. CD": "H vs. CD",
    "CD uninflamed vs. inflamed": "U vs. I",
}


GUIDE = {"color": "0.9", "lw": 0.4, "zorder": 1}


def dot_size(value):
    return 1 + DOT_AREA * value


#: The largest dot's area (pt^2): no overlap at the stacked panels' 5 mm
DOT_AREA = 80
#: Tick names in the bar and contents panels (pt)
NAME_SIZE = style.FONT_SIZE_DENSE


METHODS = ("PB", "STAC")
FITS = [(m, lab) for m in METHODS for lab in LABELLINGS]
#: Columns placing less than this share of their squared loading on one
#: CCC-RISE component's carrying pairs can enter the contents panels.
MISSED = 0.5
#: Gap between fits' bars, in bars.
FIT_GAP = 1.2


def _column_name(method: str, column: int, labelling: str) -> str:
    return f"{short(method, column)} {labelling[0].upper()}"


@cache
def carrying(dataset: str, method: str, labelling: str = "coarse") -> dict:
    lr = lr_matrix(dataset, method, labelling)
    return {
        int(c): lr.index[covariation.carrying_pairs(lr[c].to_numpy())].tolist()
        for c in lr.columns
    }


def cccrise_pairs(dataset: str) -> set:
    return set().union(*map(set, carrying(dataset, "cccRISE").values()))


def _on_components(squared: pd.Series, dataset: str) -> np.ndarray:
    """A column's share of its squared loading on each CCC-RISE component's
    carrying pairs."""
    members = pd.DataFrame(
        {k: squared.index.isin(p) for k, p in carrying(dataset, "cccRISE").items()},
        index=squared.index,
    ).to_numpy(float)
    return squared.to_numpy() / squared.sum() @ members


@cache
def columns(dataset: str) -> pd.DataFrame:
    """Every PB and STAC LR column: the CCC-RISE component on whose carrying
    pairs it places the largest share of its squared loading, and that share
    (`_split.on_component`, as 5C's related columns)."""
    rows = []
    for method, lab in FITS:
        lr = lr_matrix(dataset, method, lab)
        for c in lr.columns:
            shares = _on_components(lr[c] ** 2, dataset)
            rows.append(
                {
                    "Method": method,
                    "Labelling": lab,
                    "Column": int(c),
                    "Name": _column_name(method, int(c), lab),
                    "Share": float(shares.max()),
                    "Of": int(np.argmax(shares) + 1),
                }
            )
    return pd.DataFrame(rows)


@cache
def slots() -> dict:
    """Each fit's first y in both cohorts' bar panels: every fit as many rows as
    its larger cohort's, `FIT_GAP` between fits, so a fit starts at the same
    height in both."""
    starts, y = {}, 0.0
    for method, lab in FITS:
        starts[(method, lab)] = y
        y += max(
            ((t := columns(d))["Method"].eq(method) & t["Labelling"].eq(lab)).sum()
            for d in ("alad", "ibd")
        )
        y += FIT_GAP
    starts["end"] = y - FIT_GAP
    return starts


def _hbars(ax, dataset: str, value: str, xlabel: str) -> None:
    """Each column's `value` as a horizontal bar, each fit (`FITS`) from its
    slot's top (`slots`, shared by the cohorts), largest first, named with its
    labelling (`_column_name`)."""
    t = columns(dataset)
    ticks, names = [], []
    for method, lab in FITS:
        g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
        y = slots()[(method, lab)]
        for r in g.sort_values([value, "Column"], ascending=[False, True]).itertuples():
            ax.barh(y, getattr(r, value), height=0.8, color=METHOD_COLORS[method], lw=0)
            ticks.append(y)
            names.append(r.Name)
            y += 1
    ax.set(xlim=(0, 1.04), xticks=[0, 0.5, 1], ylim=(slots()["end"] - 0.2, -0.8))
    ax.set_yticks(ticks, names, fontsize=NAME_SIZE)
    ax.tick_params(axis="y", pad=1)
    ax.set_xlabel(xlabel)


def share(ax, dataset: str) -> None:
    common.source_data(
        f"comparator_column_share_{dataset}",
        columns(dataset)[["Method", "Labelling", "Name", "Share", "Of"]].rename(
            columns={"Of": "CCC-RISE component"}
        ),
        "Each PB and STAC LR column's largest share of its squared LR pair loading "
        "on one CCC-RISE component's carrying pairs, and that component.",
    )
    _hbars(ax, dataset, "Share", "Squared loading on\ncomponent's top LR pairs")


def missed(dataset: str) -> pd.DataFrame:
    """The PB and STAC columns placing under `MISSED` of their squared loading
    on any one CCC-RISE component's carrying pairs."""
    t = columns(dataset)
    return t[t["Share"] < MISSED].sort_values(
        ["Method", "Labelling", "Share", "Column"]
    )


def significant(dataset: str) -> set[str]:
    """Names of C's columns (those below `MISSED`) whose strongest association
    (of the association panel's) has q < `common.SIG_ALPHA`."""
    t = load("associations", dataset=dataset)
    t = t[t["Association"].isin(ASSOCIATION_NAMES) & (t["PAdj"] < common.SIG_ALPHA)]
    hits = missed(dataset).merge(
        t[["Method", "Labelling", "Component"]].drop_duplicates(),
        left_on=["Method", "Labelling", "Column"],
        right_on=["Method", "Labelling", "Component"],
    )
    return set(hits["Name"])


def associated(dataset: str) -> pd.DataFrame:
    """The associations at q < `common.SIG_ALPHA` of every PB and STAC column
    and of the reported CCC-RISE fit's components (those the association
    panel's box lists)."""
    t = load("associations", dataset=dataset)
    t = t[t["Association"].isin(ASSOCIATION_NAMES) & (t["PAdj"] < common.SIG_ALPHA)]
    reported = (t["Method"] == "cccRISE") & (t["Labelling"] == "coarse")
    return t[t["Method"].isin(METHODS) | reported]


def homophilic(pair: str) -> bool:
    """Whether an LR pair's ligand is its receptor (adhesion, as CNTN2–CNTN2)."""
    ligand, receptor = pair.split("^", 1)
    return ligand == receptor


def lacking(pairs, ours: set) -> bool:
    """Whether `pairs` hold a signalling pair CCC-RISE lacks: one outside its
    carrying pairs (`ours`) that is not homophilic adhesion."""
    return any(p not in ours and not homophilic(p) for p in pairs)


def _contents_rows(dataset: str) -> tuple[list, list, pd.Series]:
    """Each column under `MISSED` share, its name, method and carrying
    pairs at |loading| / max |loading|; the pairs, in order of first
    appearance down the rows (a diagonal); and each pair's largest such value
    in a CCC-RISE component."""
    rows, pairs = [], []
    for r in missed(dataset).to_dict("records"):
        method, lab, col = r["Method"], r["Labelling"], r["Column"]
        column = lr_matrix(dataset, method, lab)[col]
        kept = carrying(dataset, method, lab)[col]
        rows.append((r["Name"], method, (column.abs() / column.abs().max()).loc[kept]))
        pairs += [p for p in kept if p not in pairs]
    cccrise = lr_matrix(dataset, "cccRISE").abs()
    closest = (cccrise / cccrise.max()).max(axis=1).reindex(pairs).fillna(0)
    return rows, pairs, closest


def _contents_source(dataset: str, rows: list, closest: pd.Series) -> None:
    def long(values: pd.Series, name: str) -> pd.DataFrame:
        return (
            values.rename(LOADING_LABEL)
            .rename_axis("LR pair")
            .reset_index()
            .assign(Column=name)
        )

    common.source_data(
        f"left_over_column_contents_{dataset}",
        pd.concat(
            [long(v, n) for n, _, v in rows] + [long(closest, "CCC-RISE")],
            ignore_index=True,
        ),
        f"Each PB or STAC column placing under {MISSED} of its squared loading on "
        "one CCC-RISE component's carrying pairs, "
        "its carrying pairs; the CCC-RISE row gives each pair's largest value in any "
        "component.",
    )


def boxed_rows(dataset: str) -> tuple[list, list, pd.Series]:
    """`_contents_rows` kept to the columns `_mark` boxes (associated, carrying
    a signalling pair CCC-RISE lacks), and to their carrying pairs."""
    rows, _, closest = _contents_rows(dataset)
    hits, ours = significant(dataset), cccrise_pairs(dataset)
    rows = [r for r in rows if r[0] in hits and lacking(r[2].index, ours)]
    pairs = list(dict.fromkeys(p for _, _, v in rows for p in v.index))
    return rows, pairs, closest.reindex(pairs)


def boxed_contents(ax, dataset: str, key: bool) -> None:
    """The boxed columns' carrying pairs (`boxed_rows`) at |loading| / max
    |loading|, pairs down and columns across, then a CCC-RISE column of each
    pair's largest such value in any component, under a bar in the cohort's
    color (`cohort_bar`); with `key`, the dot sizes keyed under the grid."""
    rows, pairs, closest = boxed_rows(dataset)
    _contents_source(dataset, rows, closest)
    ys = {p: i for i, p in enumerate(pairs)}
    for x, (_, method, values) in enumerate(rows):
        ax.scatter(
            [x] * len(values),
            [ys[p] for p in values.index],
            s=dot_size(values.to_numpy()),
            color=METHOD_COLORS[method],
            lw=0,
            zorder=3,
        )
    x = len(rows) + 0.3
    for y in ys.values():
        ax.plot([-0.7, x + 0.7], [y, y], **GUIDE)
    for xx in [*range(len(rows)), x]:
        ax.plot([xx, xx], [-0.7, len(ys) - 0.3], **GUIDE)
    ax.scatter(
        [x] * len(ys),
        list(ys.values()),
        s=dot_size(closest.to_numpy()),
        color=METHOD_COLORS["cccRISE"],
        lw=0,
        zorder=3,
    )
    ax.set(xlim=(-0.7, x + 0.7), ylim=(len(ys) - 0.3, -0.7))
    cohort_bar(ax, dataset, -0.7, x + 0.7)
    ax.set_xticks(
        [*range(len(rows)), x],
        [n for n, _, _ in rows] + ["CCC-\nRISE"],
        fontsize=NAME_SIZE,
    )
    ax.set_yticks(
        list(ys.values()),
        [p.replace("^", "–").replace("&", "+") for p in ys],
        fontsize=NAME_SIZE,
    )
    if key:
        size_legend(ax, "below")


#: The size key's levels, as Figures 2 and 4 key detection, and its text (pt)
SIZE_LEVELS = (0.25, 0.5, 0.75, 1)
KEY_SIZE = style.FONT_SIZE_LABEL  # as the bar panels' axis labels


def size_legend(ax, side: str = "right") -> None:
    """The dot sizes keyed `side` of `ax`, titled, as Figures 2 and 4 key
    detected fraction (`common.legend`)."""
    handles = [
        ax.scatter([], [], s=dot_size(f), color=style.COLORS["ink"], lw=0)
        for f in SIZE_LEVELS
    ]
    common.legend(
        ax,
        side,
        handles=handles,
        labels=[f"{f:g}" for f in SIZE_LEVELS],
        title="|Loading| / max |loading|",
        ncols=len(SIZE_LEVELS),
        fontsize=KEY_SIZE,
        title_fontsize=KEY_SIZE,
        columnspacing=0.8,
        handletextpad=0.1,
    )
