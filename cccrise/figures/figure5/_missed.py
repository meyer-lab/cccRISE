"""Figure 5's supplements on what the comparators find that CCC-RISE does not,
one per cohort (`figureS_comparator_missed_<cohort>`): each PB and STAC LR
column's largest share of its squared loading on one CCC-RISE component's
carrying pairs (`share`; the measure of 5C's related columns), every column's
strongest association with those of CCC-RISE's components the box lists
(`associations`), and the carrying pairs of the columns under `MISSED` share
(`contents`), the associated ones carrying a signalling pair CCC-RISE lacks
(`lacking`) bold and boxed. A carrying pair of a column is one of the fewest
pairs carrying 90% of its squared loading (`covariation.carrying_pairs`), as
for the components."""

from functools import cache

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation

from ... import covariation
from .. import common, style
from ._common import METHOD_COLORS, load
from ._data import LABELLINGS, short
from ._programs import lr_matrix
from ._recovery import LINK, LOADING_LABEL

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
#: A box row splits comparators from cccRISE past this many components.
ROW_COMPONENTS = 4
#: The box's top left, as a share of the axes, and its text.
BOX_AT = (0.06, 0.86)
BOX_SIZE = style.FONT_SIZE_DENSE - 1.2
TEXT_SIZE = style.FONT_SIZE_DENSE - 0.7


GUIDE = {"color": "0.9", "lw": 0.4, "zorder": 1}
#: Association names placed by hand after `_separate_labels`: offset (points)
#: and alignment, by cohort and name.
NUDGES = {
    "ibd": {"C1": ((3, -6), "left"), "S5 F": ((3, -4), "left")},
}


def dot_size(value):
    return 1 + 24 * value


def _name(r) -> str:
    if r.Method == "cccRISE":
        return f"C{r.Component}"
    return f"{short(r.Method, r.Component)} {r.Labelling[0].upper()}"


def _box_rows(t: pd.DataFrame) -> list[tuple[str, str]]:
    """Each association and its significant components, comparators and cccRISE
    on rows of their own past `ROW_COMPONENTS`."""
    rows = []
    hits = t[t["PAdj"] < common.SIG_ALPHA]
    for association, name in ASSOCIATION_NAMES.items():
        g = hits[hits["Association"] == association]
        if g.empty:
            continue
        names = [_name(r) for r in g.sort_values("PAdj").itertuples()]
        if len(names) <= ROW_COMPONENTS:
            rows.append((name, ", ".join(names)))
            continue
        rows.append((name, ", ".join(n for n in names if not n.startswith("C"))))
        rows.append(("", ", ".join(n for n in names if n.startswith("C"))))
    return rows


def _box(ax, rows: list) -> None:
    """`rows` in a box fitted to them, at `BOX_AT` of the axes."""
    x, top = BOX_AT
    renderer = ax.figure.canvas.get_renderer()
    pitch = 2.7 * style.MM / (ax.bbox.height / ax.figure.dpi)
    pad_x = 1.2 * style.MM / (ax.bbox.width / ax.figure.dpi)
    pad_y = 0.75 * pitch

    def write(at: float, y: float, text: str, **kwargs):
        return ax.text(
            at,
            y,
            text,
            transform=ax.transAxes,
            fontsize=BOX_SIZE,
            va="center",
            zorder=7,
            **kwargs,
        )

    def width(artist) -> float:
        return artist.get_window_extent(renderer).width / ax.bbox.width

    ys = [top - pad_y - i * pitch for i in range(len(rows))]
    names = [
        write(x + pad_x, y, name, fontweight="bold")
        for (name, _), y in zip(rows, ys, strict=True)
    ]
    left = x + 2 * pad_x + max(width(n) for n in names)
    values = [write(left, y, value) for (_, value), y in zip(rows, ys, strict=True)]
    right = left + max(width(v) for v in values) + pad_x
    height = 2 * pad_y + (len(rows) - 1) * pitch
    ax.add_patch(
        Rectangle(
            (x, top - height),
            right - x,
            height,
            transform=ax.transAxes,
            fc="white",
            ec="0.3",
            lw=0.6,
            zorder=6,
        )
    )


def _associations(ax, dataset: str, columns: pd.DataFrame) -> None:
    """Each of `columns`' strongest association, and that of each CCC-RISE
    component with one at q < `common.SIG_ALPHA` (blue); the box lists, per
    association, the columns and components significant in it."""
    t = load("associations", dataset=dataset)
    t = t[t["Association"].isin(ASSOCIATION_NAMES)]
    keys = ["Method", "Labelling", "Component"]
    drawn = t.merge(columns[keys], on=keys)
    programs = t[(t["Method"] == "cccRISE") & (t["Labelling"] == "coarse")]
    hits = programs[programs["PAdj"] < common.SIG_ALPHA]
    best = (
        pd.concat([drawn, programs[programs["Component"].isin(hits["Component"])]])
        .sort_values(["PAdj", "EtaSquared"], ascending=[True, False])
        .drop_duplicates(keys)
        .assign(NegLog=lambda d: -np.log10(d["PAdj"]))
    )
    common.source_data(
        f"comparator_column_associations_{dataset}",
        pd.concat([drawn, programs])[
            [*keys, "Association", "EtaSquared", "PAdj"]
        ].rename(columns={"EtaSquared": "Eta squared", "PAdj": "q"}),
        "Each PB and STAC column's and CCC-RISE component's associations; the panel "
        "draws each column's lowest q, and each CCC-RISE component's where it is "
        f"below {common.SIG_ALPHA}; its box lists those below {common.SIG_ALPHA}.",
    )
    for method, g in best.groupby("Method"):
        ax.scatter(
            g["EtaSquared"],
            g["NegLog"],
            s=11,
            color=METHOD_COLORS[method],
            ec="white",
            lw=0.3,
            zorder=4 if method == "cccRISE" else 3,
        )
    ax.axhline(
        -np.log10(common.SIG_ALPHA),
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
    )
    ax.set(
        xlabel=r"$\eta^2$, strongest association",
        ylabel=r"$-\log_{10}\ q$",
        xlim=(0, best["EtaSquared"].max() * 1.25),
        ylim=(0, best["NegLog"].max() * 1.25),
    )
    for r in best[best["PAdj"] < common.SIG_ALPHA].itertuples():
        ax.annotate(
            _name(r),
            (r.EtaSquared, r.NegLog),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color=METHOD_COLORS["cccRISE"] if r.Method == "cccRISE" else "black",
            zorder=5,
        )
    _box(ax, _box_rows(pd.concat([drawn, programs])))


METHODS = ("PB", "STAC")
FITS = [(m, lab) for m in METHODS for lab in LABELLINGS]
#: Columns placing less than this share of their squared loading on one
#: CCC-RISE component's carrying pairs are drawn in C.
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


def _bars(ax, dataset: str, value: str, ylabel: str) -> None:
    """Each column's `value` as a bar, fits side by side."""
    t = columns(dataset)
    x, ticks, names, heads = 0.0, [], [], {}
    for method, lab in FITS:
        g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
        g = g.sort_values([value, "Column"], ascending=[False, True])
        start = x
        for r in g.itertuples():
            ax.bar(x, getattr(r, value), width=0.8, color=METHOD_COLORS[method], lw=0)
            ticks.append(x)
            names.append(short(method, r.Column))
            x += 1
        heads[f"{method} {lab}"] = (start + x - 1) / 2
        x += FIT_GAP
    ax.set(xlim=(-0.8, x - FIT_GAP + 0.2), ylim=(0, 1.04), yticks=[0, 0.5, 1])
    ax.set_xticks(ticks, names, rotation=90, fontsize=TEXT_SIZE - 0.5)
    ax.tick_params(axis="x", pad=1)
    ax.set_ylabel(ylabel)
    for name, at in heads.items():
        ax.text(
            at,
            1.06,
            name,
            ha="center",
            va="bottom",
            fontsize=TEXT_SIZE,
            transform=ax.get_xaxis_transform(),
        )


def share(ax, dataset: str) -> None:
    common.source_data(
        f"comparator_column_share_{dataset}",
        columns(dataset)[["Method", "Labelling", "Name", "Share", "Of"]].rename(
            columns={"Of": "CCC-RISE component"}
        ),
        "Each PB and STAC LR column's largest share of its squared LR pair loading "
        "on one CCC-RISE component's carrying pairs, and that component.",
    )
    _bars(
        ax,
        dataset,
        "Share",
        "Squared loading on\ncomponent's top LR pairs",
    )
    ax.axhline(
        MISSED,
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
        zorder=0,
    )


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


def homophilic(pair: str) -> bool:
    """Whether an LR pair's ligand is its receptor (adhesion, as CNTN2–CNTN2)."""
    ligand, receptor = pair.split("^", 1)
    return ligand == receptor


def lacking(pairs, ours: set) -> bool:
    """Whether `pairs` hold a signalling pair CCC-RISE lacks: one outside its
    carrying pairs (`ours`) that is not homophilic adhesion."""
    return any(p not in ours and not homophilic(p) for p in pairs)


def _mark(ax, rows: list, xs: dict, hits: set[str], ours: set) -> None:
    """Bold each associated column carrying a signalling pair CCC-RISE lacks
    (`lacking`) and box its dots in red."""
    for y, (name, _, values) in enumerate(rows):
        if name not in hits or not lacking(values.index, ours):
            continue
        ax.get_yticklabels()[y].set_fontweight("bold")
        x = [xs[p] for p in values.index]
        ax.add_patch(
            Rectangle(
                (min(x) - 0.45, y - 0.42),
                max(x) - min(x) + 0.9,
                0.84,
                fc="none",
                ec=LINK,
                lw=0.8,
                zorder=4,
            )
        )


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


def _contents_grid(ax, n_rows: int, xs: dict) -> float:
    """Guides; the CCC-RISE row's y."""
    y = n_rows + 0.3
    for x in xs.values():
        ax.plot([x, x], [-0.7, y + 0.7], **GUIDE)
    for yy in [*range(n_rows), y]:
        ax.plot([-0.7, len(xs) - 0.3], [yy, yy], **GUIDE)
    return y


def contents(ax, dataset: str) -> None:
    """Each column under `MISSED` share, its carrying pairs at
    |loading| / max |loading|, in order of first appearance; a
    CCC-RISE row with each pair's largest such value in any component; the
    columns with an association at q < 0.05 that carry a signalling pair
    CCC-RISE lacks bold and boxed (`_mark`)."""
    rows, pairs, closest = _contents_rows(dataset)
    _contents_source(dataset, rows, closest)
    xs = {p: i for i, p in enumerate(pairs)}
    for y, (_, method, values) in enumerate(rows):
        ax.scatter(
            [xs[p] for p in values.index],
            [y] * len(values),
            s=dot_size(values.to_numpy()),
            color=METHOD_COLORS[method],
            lw=0,
            zorder=3,
        )
    y = _contents_grid(ax, len(rows), xs)
    ax.scatter(
        list(xs.values()),
        [y] * len(xs),
        s=dot_size(closest.to_numpy()),
        color=METHOD_COLORS["cccRISE"],
        lw=0,
        zorder=3,
    )
    ax.set(xlim=(-0.7, len(xs) - 0.3), ylim=(y + 0.7, -0.7))
    ax.set_yticks(
        [*range(len(rows)), y],
        [n for n, _, _ in rows] + ["CCC-RISE"],
        fontsize=TEXT_SIZE,
    )
    ax.set_xticks(
        list(xs.values()),
        [p.replace("^", "–").replace("&", "+") for p in xs],
        rotation=90,
        fontsize=style.FONT_SIZE_DENSE - 1,
    )
    _mark(ax, rows, xs, significant(dataset), cccrise_pairs(dataset))
    _size_key(ax)


def _size_key(ax) -> None:
    """The dot sizes in a black-outlined box at the grid's top right."""
    w, h = 26 * style.MM, 9 * style.MM
    width, height = ax.bbox.width / ax.figure.dpi, ax.bbox.height / ax.figure.dpi
    key = ax.inset_axes(
        (1 - w / width - 0.004, 1 - h / height - 0.006, w / width, h / height), zorder=5
    )
    key.set(xlim=(0, 26), ylim=(0, 9), xticks=[], yticks=[])
    key.set_facecolor("white")
    for spine in key.spines.values():
        spine.set(visible=True, color="0.3", linewidth=0.6)
    key.text(13, 6.6, LOADING_LABEL, ha="center", va="center", fontsize=TEXT_SIZE)
    for i, level in enumerate((0.25, 0.5, 1)):
        x = 3 + 8 * i
        key.scatter(x, 2.6, s=dot_size(level), color="0.4", lw=0)
        key.text(x + 1.4, 2.6, f"{level:g}", va="center", fontsize=TEXT_SIZE)


def associations(ax, dataset: str) -> None:
    """Every PB and STAC column's strongest association, with those of the
    CCC-RISE components the box lists; those with q < 0.05 named and listed,
    the names in `NUDGES` placed by hand."""
    _associations(ax, dataset, columns(dataset).rename(columns={"Column": "Component"}))
    _clear_box(ax)
    _separate_labels(ax)
    _nudge(ax, NUDGES.get(dataset, {}))


def _nudge(ax, nudges: dict) -> None:
    """Place each name in `nudges` at its offset and alignment."""
    for label in (t for t in ax.texts if isinstance(t, Annotation)):
        if label.get_text() in nudges:
            label.xyann, ha = nudges[label.get_text()]
            label.set(ha=ha)


def _clear_box(ax) -> None:
    """Headroom so the points sit below the association box at the top."""
    top = max(
        c.get_offsets()[:, 1].max() for c in ax.collections if len(c.get_offsets())
    )
    ax.set_ylim(0, top * 2.5)


def _separate_labels(ax) -> None:
    """Move a point's name left of it, then below it, when it overlaps a name
    already placed."""
    renderer = ax.figure.canvas.get_renderer()
    placed = []
    for label in (t for t in ax.texts if isinstance(t, Annotation)):
        for offset, ha in (((3, 2), "left"), ((-3, 2), "right"), ((3, -7), "left")):
            label.set(ha=ha)
            label.xyann = offset
            box = label.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                break
        placed.append(label.get_window_extent(renderer))
