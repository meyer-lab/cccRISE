"""Figure 4's panels about particular LR columns and their contents: reach,
containment in cccRISE, capture of a reference column's top pairs, the
strongest associations, sample profiles, and how columns shift from coarse
to fine labels; and the breadth and input-tensor panels."""

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from ...comparators import scores
from .. import common, style
from ._common import COMPARATORS, DATASET, METHOD_COLORS, METHOD_LABELS
from ._data import (
    LABELLINGS,
    NO_MATCH,
    abs_cos,
    both,
    capture_margin,
    load,
    lr_columns,
    short,
    source_data,
)
from ._layout import LABELLING_HANDLES, LABELLING_MARKERS, method_handles
from ._panels import _frame

#: Each comparator fit, coarse then fine per method.
GROUPS = [(m, lab) for m in COMPARATORS for lab in LABELLINGS]
#: How many of a reference column's heaviest pairs a capture panel shows.
CAPTURE_PAIRS = 10
#: The sample-profile panel's columns (method, labelling, column) and names,
#: and the order of its outcome groups.
PROFILED = [
    (("cccRISE", "coarse", 4), "C4"),
    (("PB", "fine", 5), "P5 fine"),
    (("STAC", "coarse", 2), "S2 coarse"),
]
OUTCOME_ORDER = ["No CLAD/Stable", "No CLAD/Recovered", "CLAD", "Died"]
#: Coarse-to-fine matches drawn from this LR |cos|.
FLOW_FROM = 0.3


def _method_names_below(ax, xs, offset=-12) -> None:
    for m, x in zip(COMPARATORS, xs, strict=True):
        ax.annotate(
            METHOD_LABELS[m],
            (x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )


#: Source data names of the containment table's columns.
CONTAINMENT_COLUMNS = {
    "Containment": "LR containment",
    "NullQ95": "Null 95th pct.",
    "Contained": "Contained",
}
CONTAINMENT_TEXT = (
    "LR containment is the squared norm of the column's projection onto the "
    "span of the other fit's LR columns (Null 95th pct.: its row-permutation "
    "null's); Contained (framed) is LR containment of at least "
    f"{scores.CONTAINED} and above the null."
)


def lr_containment(
    of: str | None, in_: str | None, dataset: str = DATASET
) -> pd.DataFrame:
    """Each comparator fit's (`GROUPS`) containment rows of LR columns of
    `of` in the span of `in_`, the comparator's place given as None; with
    the comparator fit as Method and Labelling."""
    frames = []
    for method, lab in GROUPS:
        t = load("containment", lab, dataset)
        t = t[
            (t["Of"] == (of or method))
            & (t["In"] == (in_ or method))
            & (t["Mode"] == "lr")
        ].sort_values("Component")
        frames.append(t.assign(Method=method, Labelling=lab))
    return pd.concat(frames, ignore_index=True)


def reach(axs, dataset: str = DATASET) -> None:
    """cccRISE's programs' LR containment in each comparator fit; contained
    cells framed."""
    ax = axs[0]
    t = lr_containment("cccRISE", None, dataset)
    common.source_data(
        "lr_containment_of_cccrise_programs",
        t.assign(Program="C" + t["Component"].astype(str))[
            ["Program", "Method", "Labelling", *CONTAINMENT_COLUMNS]
        ].rename(columns=CONTAINMENT_COLUMNS),
        "Each cccRISE program's LR column (Program) in each comparator fit "
        "(Method, Labelling). " + CONTAINMENT_TEXT,
    )
    grid = t.pivot(index="Component", columns=["Method", "Labelling"])
    values = grid["Containment"][GROUPS].to_numpy(float)
    contained = grid["Contained"][GROUPS].to_numpy(bool)
    image = ax.imshow(values, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if float(v) > 0.7 else style.COLORS["ink"],
        )
        if contained[i, j]:
            ax.add_patch(_frame(j, i))
    ax.set_yticks(range(values.shape[0]), [f"C{k + 1}" for k in range(values.shape[0])])
    ax.set_xticks(
        range(len(GROUPS)),
        [lab[0].upper() for _, lab in GROUPS],
        fontsize=style.FONT_SIZE_DENSE,
    )
    _method_names_below(ax, [2 * i + 0.5 for i in range(len(COMPARATORS))])
    ax.tick_params(length=0)
    common.colorbar(ax, image, "LR containment")


def _containment_strip(ax, x, t) -> None:
    cmap = style.CMAPS["weight"]
    for i, r in enumerate(t.itertuples()):
        ax.add_patch(
            Rectangle(
                (x, i), 1, 1, facecolor=cmap(r.Containment), edgecolor="white", lw=0.4
            )
        )
        if r.Contained:
            ax.add_patch(Rectangle((x, i), 1, 1, fill=False, edgecolor="black", lw=1))
        ax.text(
            x - 0.08,
            i + 0.5,
            short(r.Of, r.Component),
            ha="right",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 2,
        )


def columns_in_cccrise(axs, dataset: str = DATASET) -> None:
    """One strip per comparator fit: its LR columns coloured by their LR
    containment in cccRISE's span, contained ones framed."""
    ax = axs[0]
    gap, label_room, x, n_max = (
        0.9,
        0.9,
        0.6,
        0,
    )  # in cell widths; x clears the panel letter
    columns = lr_containment(None, "cccRISE", dataset)
    source_data(
        "lr_containment_in_cccrise_by_column",
        columns,
        CONTAINMENT_COLUMNS,
        "Each comparator LR column in cccRISE's LR span. " + CONTAINMENT_TEXT,
    )
    for method, lab in GROUPS:
        t = columns[(columns["Method"] == method) & (columns["Labelling"] == lab)]
        x += label_room
        _containment_strip(ax, x, t)
        ax.text(
            x + 0.5,
            -0.3,
            f"{METHOD_LABELS[method]}\n{lab}",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE - 1,
        )
        n_max = max(n_max, len(t))
        x += 1 + gap
    ax.set(xlim=(0, x - gap), ylim=(n_max, -1.6))
    ax.set_axis_off()
    cax = ax.inset_axes((0.06, 0.24, 0.58, 0.035))
    bar = ax.figure.colorbar(
        ScalarMappable(Normalize(0, 1), style.CMAPS["weight"]),
        cax=cax,
        orientation="horizontal",
        ticks=(0, 0.5, 1),
    )
    bar.ax.tick_params(labelsize=style.FONT_SIZE_DENSE - 1)
    bar.ax.set_title("LR containment in cccRISE", fontsize=style.FONT_SIZE_DENSE - 1)


def _column_name(key) -> list[str]:
    method, lab, c = key
    return ["cccRISE", f"C{c}"] if method == "cccRISE" else [short(method, c), lab]


def _capture_labels(ax, reference, shown, columns) -> None:
    """Under each column: its name, |cos| with `reference` and Δρ, with a
    row header left of the columns."""
    size = style.FONT_SIZE_DENSE - 1
    target = columns[reference].to_numpy()
    rows = [
        [
            *_column_name(k),
            f"{cos:.2f}",
            _margin_text(capture_margin(reference, k, cos)),
        ]
        for k in shown
        for cos in [abs_cos(columns[k].to_numpy(), target)]
    ]
    rows.append(["", "", "|cos|", "Δρ"])
    for j, lines in zip([*range(len(shown)), -0.5], rows, strict=True):
        for i, line in enumerate(lines):
            ax.annotate(
                line,
                (j, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -3 - i * 1.2 * size),
                textcoords="offset points",
                ha="right" if j < 0 else "center",
                va="top",
                fontsize=size,
                fontweight="bold" if j == 0 else "normal",
                annotation_clip=False,
            )


def _margin_text(margin: float) -> str:
    return "—" if np.isnan(margin) else f"{margin:+.2f}"


def _capture_source(reference, shown, pairs, values, columns) -> None:
    names = [" ".join(_column_name(k)) for k in shown]
    common.source_data(
        "relative_lr_loadings_of_top_pairs",
        pd.DataFrame(values, index=[q.replace("^", "–") for q in pairs], columns=names)
        .rename_axis(index="LR pair", columns="Column")
        .stack()
        .rename("|Loading| / heaviest")
        .reset_index(),
        f"{names[0]}'s heaviest LR pairs' |loading| in each column, relative to "
        "that column's heaviest pair.",
    )
    target = columns[reference].to_numpy()
    cos = [abs_cos(columns[k].to_numpy(), target) for k in shown]
    common.source_data(
        "column_match_to_reference",
        pd.DataFrame(
            {
                "Column": names,
                "|cos|": cos,
                "Delta rho": [
                    capture_margin(reference, k, c)
                    for k, c in zip(shown, cos, strict=True)
                ],
            }
        ),
        f"Each column's LR |cosine| with {names[0]}, and Delta rho: the mean "
        f"pairwise Spearman correlation across samples of {names[0]}'s top LR "
        "pairs, scored in the column's sender/receiver context, minus the "
        "stricter of its random and expression-matched nulls' 95th "
        f"percentiles (empty where |cos| < {NO_MATCH}).",
    )


def capture(
    axs, reference=("cccRISE", "coarse", 4), matches=4, colorbar=True, compact=False
) -> None:
    """`reference`'s `CAPTURE_PAIRS` heaviest LR pairs (rows) in it and in
    the `matches` LR columns of other fits (another method, or its own at
    the other labelling) of greatest |cos| with it: each column's |loading|
    relative to its heaviest pair, with |cos| and the Δρ of `reference`'s
    top pairs in its context."""
    ax = axs[0]
    columns = lr_columns()
    target = columns[reference].to_numpy()
    others = sorted(
        (k for k in columns if k[:2] != reference[:2]),
        key=lambda k: -abs_cos(columns[k].to_numpy(), target),
    )
    shown = [reference, *others[:matches]]
    pairs = list(columns[reference].abs().nlargest(CAPTURE_PAIRS).index)
    values = np.column_stack(
        [columns[k].abs().loc[pairs] / columns[k].abs().max() for k in shown]
    )
    _capture_source(reference, shown, pairs, values, columns)
    image = ax.imshow(values, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(
        range(len(pairs)),
        [q.replace("^", "–") for q in pairs],
        fontsize=style.FONT_SIZE_DENSE - (1 if compact else 0),
    )
    _capture_labels(ax, reference, shown, columns)
    ax.set_xticks([])
    ax.tick_params(length=0)
    if colorbar:
        common.colorbar(ax, image, "|Loading| / heaviest")


def _profiles() -> pd.DataFrame:
    """`PROFILED`'s sample profiles (columns) scaled to their largest
    |loading|, samples by outcome group then by |C4|, with X."""
    tables = both("profiles")
    out = {}
    for (method, lab, c), name in PROFILED:
        t = tables[lab]
        g = t[(t["Method"] == method) & (t["Component"] == c)].set_index("Sample")
        out[name] = g["Loading"] / g["Loading"].abs().max()
        outcome = g["Outcome"]
    t = pd.DataFrame(out).assign(Group=outcome)
    first = PROFILED[0][1]
    t = t.assign(
        Rank=t["Group"].map(OUTCOME_ORDER.index), Key=-t[first].abs()
    ).sort_values(["Rank", "Key"])
    return t.assign(X=np.arange(len(t)))


def sample_profiles(axs) -> None:
    """`PROFILED`'s sample profiles, one boxed row each (bars from its
    floor), samples grouped by 6-month outcome."""
    ax = axs[0]
    t = _profiles()
    names = [name for _, name in PROFILED]
    common.source_data(
        "scaled_sample_profiles",
        t.rename_axis("Sample")
        .reset_index()
        .melt(
            id_vars=["Sample", "Group"],
            value_vars=names,
            var_name="Column",
            value_name="Scaled loading",
        )
        .rename(columns={"Group": "6-month outcome"}),
        "Each column's sample profile (cccRISE's sample loadings; a comparator "
        "LR column's as its outcome associations use it) divided by its "
        "largest |loading|; the bars show the positive part.",
    )
    n = len(PROFILED)
    line = {"color": style.COLORS["ink"], "lw": ax.spines["bottom"].get_linewidth()}
    for k, ((key, name), floor) in enumerate(
        zip(PROFILED, range(n - 1, -1, -1), strict=True)
    ):
        color = METHOD_COLORS[key[0]]
        ax.bar(
            t["X"],
            t[name].clip(lower=0) * 0.85,
            bottom=floor,
            width=0.8,
            color=color,
            zorder=2,
        )
        ax.text(
            -1.2,
            floor + 0.5,
            name,
            ha="right",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color=color,
            fontweight="bold",
        )
        if k:
            ax.axhline(floor + 1, **line, zorder=3)
    for k, (group, g) in enumerate(t.groupby("Group", sort=False)):
        if k:
            ax.axvline(g["X"].min() - 0.5, **line, zorder=3)
        ax.text(
            g["X"].mean(),
            n + 0.06,
            group,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE - 1,
        )
    for side in ax.spines.values():
        side.set(visible=True, edgecolor=line["color"], linewidth=line["lw"])
    ax.set(
        xlim=(-0.5, len(t) - 0.5), ylim=(0, n), xticks=[], yticks=[], xlabel="Samples"
    )


def flow_width(cos: float) -> float:
    """A coarse-to-fine line's width: one fixed step per 0.1 of LR |cos|,
    from 0.25 at `FLOW_FROM` to 1.65 at 1."""
    return 0.25 + 0.2 * round((min(cos, 1.0) - FLOW_FROM) / 0.1)


def _shift(method) -> tuple[np.ndarray, list, list]:
    """A comparator's coarse x fine LR |cos|, and its coarse and fine
    columns."""
    columns = lr_columns()
    keys = {lab: [k for k in columns if k[:2] == (method, lab)] for lab in LABELLINGS}
    unit = {
        lab: np.column_stack([columns[k] / np.linalg.norm(columns[k]) for k in ks])
        for lab, ks in keys.items()
    }
    return np.abs(unit["coarse"].T @ unit["fine"]), keys["coarse"], keys["fine"]


def _flow_block(ax, method, x) -> None:
    cos, coarse, fine = _shift(method)
    color = METHOD_COLORS[method]
    yc = np.linspace(1, 0, len(coarse)) if len(coarse) > 1 else np.array([0.5])
    yf = np.empty(len(fine))
    yf[np.lexsort((-cos.max(axis=0), cos.argmax(axis=0)))] = np.linspace(
        1, 0, len(fine)
    )
    for (i, j), v in np.ndenumerate(cos):
        if v >= FLOW_FROM:
            ax.plot(
                [x, x + 1],
                [yc[i], yf[j]],
                color=color,
                lw=flow_width(v),
                alpha=0.75,
                solid_capstyle="butt",
                zorder=2,
            )
    for xs, ys, keys, ha, dx in (
        (x, yc, coarse, "right", -0.06),
        (x + 1, yf, fine, "left", 0.06),
    ):
        ax.scatter([xs] * len(ys), ys, s=6, color=color, zorder=3)
        for y, k in zip(ys, keys, strict=True):
            ax.text(
                xs + dx,
                y,
                short(method, k[2]),
                ha=ha,
                va="center",
                fontsize=style.FONT_SIZE_DENSE - 3,
            )
    ax.text(
        x + 0.5,
        1.08,
        METHOD_LABELS[method],
        ha="center",
        fontsize=style.FONT_SIZE_DENSE,
    )
    for xs, letter in ((x, "C"), (x + 1, "F")):
        ax.text(xs, -0.1, letter, ha="center", fontsize=style.FONT_SIZE_DENSE - 1)


def _flow_source() -> None:
    frames = []
    for m in COMPARATORS:
        cos, coarse, fine = _shift(m)
        frames.append(
            pd.DataFrame(
                cos,
                index=[short(m, k[2]) for k in coarse],
                columns=[short(m, k[2]) for k in fine],
            )
            .rename_axis(index="Coarse column", columns="Fine column")
            .stack()
            .rename("LR |cos|")
            .reset_index()
            .assign(Method=m)
        )
    t = pd.concat(frames, ignore_index=True)
    common.source_data(
        "coarse_to_fine_lr_cosine",
        t.assign(Drawn=t["LR |cos|"] >= FLOW_FROM)[
            ["Method", "Coarse column", "Fine column", "LR |cos|", "Drawn"]
        ],
        "|Cosine| between each comparator's coarse and fine LR columns, on "
        f"cccRISE's LR pairs; Drawn (a line) at {FLOW_FROM} or more.",
    )


def coarse_to_fine(axs) -> None:
    """Each comparator's coarse LR columns (left) joined to its fine ones
    (right) where LR |cos| >= `FLOW_FROM`, line width stepping with |cos|."""
    ax = axs[0]
    _flow_source()
    for i, m in enumerate(COMPARATORS):
        _flow_block(ax, m, 1.75 * i)
    ax.set(xlim=(-0.35, 1.75 * len(COMPARATORS) - 0.4), ylim=(-0.15, 1.15))
    ax.set_axis_off()
    ax.text(
        0.5,
        -0.04,
        "LR |cos| match",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=style.FONT_SIZE_LABEL,
    )


def _inverse_simpson(v) -> float:
    share = v**2 / (v**2).sum()
    return float(1 / (share**2).sum())


def breadth(axs) -> None:
    """Effective LR pairs against effective samples (inverse Simpson indices
    of squared loadings; a column's sample profile), every fit."""
    ax = axs[0]
    drawn = []
    for lab, b in both("breadth").items():
        lr = b[b["Mode"] == "lr"].set_index(["Method", "Component"])["Effective"]
        samples = (
            load("profiles", lab)
            .groupby(["Method", "Component"])["Loading"]
            .agg(_inverse_simpson)
        )
        for method in ("cccRISE", *COMPARATORS):
            if method == "cccRISE" and lab != "coarse":
                continue
            keys = [k for k in samples.index if k[0] == method]
            drawn.append(
                pd.DataFrame(
                    {
                        "Method": method,
                        "Labelling": lab,
                        "Component": [k[1] for k in keys],
                        "EffectiveLR": lr.loc[keys].to_numpy(),
                        "EffectiveSamples": samples.loc[keys].to_numpy(),
                    }
                )
            )
            ax.scatter(
                lr.loc[keys],
                samples.loc[keys],
                marker=LABELLING_MARKERS[lab],
                s=style.POINT_SIZE * (2 if method == "cccRISE" else 1),
                color=METHOD_COLORS[method],
                edgecolor="white",
                lw=0.4,
                zorder=3 if method == "cccRISE" else 2,
            )
    source_data(
        "effective_lr_pairs_and_samples",
        pd.concat(drawn, ignore_index=True),
        {"EffectiveLR": "Effective LR pairs", "EffectiveSamples": "Effective samples"},
        "Each component's effective number of LR pairs and of samples: the "
        "inverse Simpson index of its squared LR loadings and of its squared "
        "sample profile.",
    )
    ax.set_xscale("log")
    ax.set(xlabel="Effective LR pairs", ylabel="Effective samples", ylim=(0, 47))
    ax.legend(
        handles=method_handles(("cccRISE", *COMPARATORS), 5) + LABELLING_HANDLES,
        loc="upper left",
        fontsize=style.FONT_SIZE_DENSE - 1,
    )


def inputs(axs) -> None:
    """Each method's input tensor (coarse labels), its per-sample mean
    (z-scored) against depth, with their Spearman ρ."""
    ax = axs[0]
    t = load("inputs", "coarse")
    t = t.assign(
        Z=t.groupby("Method")["Mean"].transform(lambda v: (v - v.mean()) / v.std())
    )
    common.source_data(
        "input_tensor_mean_by_sample",
        t[["Method", "Sample", "Depth", "Mean", "Z"]].rename(
            columns={
                "Depth": "Median genes detected per cell",
                "Mean": "Input tensor mean",
                "Z": "z-score",
            }
        ),
        "Each method's input tensor (coarse labels) averaged per sample over "
        "its observed entries, z-scored within the method, against the "
        "sample's depth; the legend gives their Spearman correlation.",
    )
    for m in t["Method"].unique():
        g = t[t["Method"] == m]
        z = g["Z"]
        rho = spearmanr(g["Mean"], g["Depth"]).statistic
        ax.scatter(
            g["Depth"],
            z,
            s=style.POINT_SIZE * 0.6,
            color=METHOD_COLORS[m],
            edgecolor="white",
            lw=0.2,
            label=f"{METHOD_LABELS[m]} (ρ {rho:.2f})",
        )
    ax.set(
        xlabel="Median genes detected per cell (d)",
        ylabel="Input tensor mean\n(z-score)",
    )
    ax.legend(loc="upper left", fontsize=style.FONT_SIZE_DENSE)
