"""What the comparators find that no cccRISE program carries: the LR contents of
their columns of other pairs, each pair's place in cccRISE, and the associations of
those columns."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, METHOD_COLORS, load
from ._data import short
from ._programs import column_census, lr_matrix
from ._recovery import LOADING_LABEL, panel_title

#: A pair is shown from this share of its column's squared loading.
SHOWN_SHARE = 0.05
#: A cohort's quadrant's alpha, and the key's swatches'.
SHADE, KEY_SHADE = 0.13, 0.26
GUIDE = {"color": "0.9", "lw": 0.4, "zorder": 1}
#: The key, in mm of the panel: its left, bottom, width and height.
KEY_BOX = (1.5, 3, 62, 14)
KEY_LEVELS = (0.25, 0.5, 0.75, 1)
LEFT_OVER = ("Single pair", "Mixed pairs")
#: The associations drawn, by their short names (not IBD's 3-way status).
ASSOCIATION_NAMES = {
    "Control vs. ALAD (All)": "ALAD",
    "ALAD Status (3-Way)": "ALAD status",
    "6-Month Outcome (4-Way)": "6-mo outcome",
    "6-Month CLAD vs. No CLAD": "6-mo CLAD",
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


def dot_size(value):
    return 1 + 24 * value


def left_over(dataset: str) -> pd.DataFrame:
    census = column_census(dataset)
    return census[census["Class"].isin(LEFT_OVER)]


def contents(dataset: str) -> tuple[list, list, pd.Series]:
    """Each left-over column's name, method and its pairs past `SHOWN_SHARE` at
    |loading| / max |loading|; the pairs in order; and each pair's largest such
    value in a cccRISE component."""
    rows, pairs = [], []
    for r in left_over(dataset).itertuples():
        column = lr_matrix(dataset, r.Method, r.Labelling)[r.Component]
        squared = column**2 / (column**2).sum()
        kept = squared[squared >= SHOWN_SHARE].sort_values(ascending=False).index
        name = f"{short(r.Method, r.Component)} {r.Labelling[0].upper()}"
        rows.append((name, r.Method, (column.abs() / column.abs().max()).loc[kept]))
        pairs += [pair for pair in kept if pair not in pairs]
    cccrise = lr_matrix(dataset, "cccRISE").abs()
    closest = (cccrise / cccrise.max()).max(axis=1).reindex(pairs)
    return rows, pairs, closest


def _cohort_rows(ax, dataset: str, x0: float, y0: float) -> tuple[dict, list, list]:
    """One cohort's columns and cccRISE row from (x0, y0): the pairs' x, and the
    rows' y and names."""
    rows, pairs, closest = contents(dataset)
    xs = {pair: x0 + i for i, pair in enumerate(pairs)}
    ticks, names, y = [], [], y0
    for name, method, values in rows:
        ax.scatter(
            [xs[pair] for pair in values.index],
            [y] * len(values),
            s=dot_size(values.to_numpy()),
            color=METHOD_COLORS[method],
            lw=0,
            zorder=3,
        )
        ticks.append(y)
        names.append(name)
        y += 1
    y += 0.3
    ax.scatter(
        list(xs.values()),
        [y] * len(xs),
        s=dot_size(closest.to_numpy()),
        color=METHOD_COLORS["cccRISE"],
        lw=0,
        zorder=3,
    )
    return xs, [*ticks, y], [*names, "cccRISE"]


def _contents_source() -> None:
    frames = []
    for dataset, label in DATASET_LABELS.items():
        rows, _, closest = contents(dataset)
        frames += [
            values.rename(LOADING_LABEL)
            .rename_axis("LR pair")
            .reset_index()
            .assign(Cohort=label, Column=name)
            for name, _, values in rows
        ]
        frames.append(
            closest.rename(LOADING_LABEL)
            .rename_axis("LR pair")
            .reset_index()
            .assign(Cohort=label, Column="cccRISE")
        )
    common.source_data(
        "left_over_column_contents",
        pd.concat(frames, ignore_index=True)[
            ["Cohort", "Column", "LR pair", LOADING_LABEL]
        ],
        "Each column in no cccRISE program's class, and its pairs holding at least "
        f"{SHOWN_SHARE} of its squared loading; the cccRISE rows give each pair's "
        "largest value in any component.",
    )


def _key(ax, width: float, height: float) -> None:
    x, y, w, h = KEY_BOX
    key = ax.inset_axes((x / width, y / height, w / width, h / height), zorder=5)
    key.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    for spine in key.spines.values():
        spine.set(visible=True, color="0.3", linewidth=0.6)
    key.text(21, 11, LOADING_LABEL, ha="center", va="center", fontsize=TEXT_SIZE)
    for i, level in enumerate(KEY_LEVELS):
        x = 4 + 10.5 * i
        key.scatter(x, 5, s=dot_size(level), color="0.4", lw=0)
        key.text(x + 2, 5, f"{level:g}", va="center", fontsize=TEXT_SIZE)
    for i, (dataset, label) in enumerate(DATASET_LABELS.items()):
        y = 8.3 - 5.6 * i
        key.add_patch(
            Rectangle(
                (47, y), 4, 3, fc=DATASET_COLORS[dataset], alpha=KEY_SHADE, ec="none"
            )
        )
        key.text(52.5, y + 1.5, label, va="center", fontsize=TEXT_SIZE)


def contents_matrix(ax) -> None:
    """ALAD's left-over columns against their pairs, then IBD's, each cohort's
    quadrant shaded, one line between the cohorts' rows."""
    _contents_source()
    alad_xs, alad_ticks, alad_names = _cohort_rows(ax, "alad", 0, 0)
    split = alad_ticks[-1] + 0.75
    ibd_xs, ibd_ticks, ibd_names = _cohort_rows(ax, "ibd", len(alad_xs), split + 0.75)
    left, top = -0.7, -0.7
    right, bottom = len(alad_xs) + len(ibd_xs) - 0.3, ibd_ticks[-1] + 0.7
    divide = len(alad_xs) - 0.5
    for x0, x1, y0, y1, dataset in (
        (left, divide, top, split, "alad"),
        (divide, right, split, bottom, "ibd"),
    ):
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fc=DATASET_COLORS[dataset],
                alpha=SHADE,
                ec="none",
                zorder=0,
            )
        )
    xs = {**alad_xs, **ibd_xs}
    for x in xs.values():
        ax.plot([x, x], [top, bottom], **GUIDE)
    for y in alad_ticks + ibd_ticks:
        ax.plot([left, right], [y, y], **GUIDE)
    ax.plot([left, right], [split, split], color="0.3", lw=0.6, zorder=2)
    ax.set(xlim=(left, right), ylim=(bottom, top))
    ax.set_yticks(alad_ticks + ibd_ticks, alad_names + ibd_names, fontsize=TEXT_SIZE)
    ax.set_xticks(
        list(xs.values()),
        [pair.replace("^", "–").replace("&", "+") for pair in xs],
        rotation=90,
        fontsize=style.FONT_SIZE_DENSE - 1,
    )
    width, height = ax.bbox.width, ax.bbox.height
    _key(
        ax,
        width / ax.figure.dpi / style.MM,
        height / ax.figure.dpi / style.MM,
    )


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


def associations(ax, dataset: str) -> None:
    """Each left-over column's strongest association; the box also lists the
    cccRISE components significant in each."""
    t = load("associations", dataset=dataset)
    t = t[t["Association"].isin(ASSOCIATION_NAMES)]
    keys = ["Method", "Labelling", "Component"]
    drawn = t.merge(left_over(dataset)[keys], on=keys)
    programs = t[(t["Method"] == "cccRISE") & (t["Labelling"] == "coarse")]
    best = (
        drawn.sort_values(["PAdj", "EtaSquared"], ascending=[True, False])
        .drop_duplicates(keys)
        .assign(NegLog=lambda d: -np.log10(d["PAdj"]))
    )
    common.source_data(
        "left_over_column_associations",
        pd.concat([drawn, programs])[
            [*keys, "Association", "EtaSquared", "PAdj"]
        ].rename(columns={"EtaSquared": "Eta squared", "PAdj": "q"}),
        "Each left-over column's and cccRISE component's associations; the panel draws "
        "each column's lowest q, and its box lists those below "
        f"{common.SIG_ALPHA}.",
    )
    for method, g in best.groupby("Method"):
        ax.scatter(
            g["EtaSquared"],
            g["NegLog"],
            s=11,
            color=METHOD_COLORS[method],
            ec="white",
            lw=0.3,
            zorder=3,
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
    panel_title(ax, DATASET_LABELS[dataset])
    for r in best[best["PAdj"] < common.SIG_ALPHA].itertuples():
        ax.annotate(
            _name(r),
            (r.EtaSquared, r.NegLog),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=style.FONT_SIZE_DENSE - 1,
            zorder=5,
        )
    _box(ax, _box_rows(pd.concat([drawn, programs])))
