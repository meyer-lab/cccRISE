"""Figure 4's panels about particular LR columns and their contents: reach,
containment in cccRISE, capture of a reference column's top pairs, the
strongest associations, sample profiles, and how columns shift from coarse
to fine labels; and the breadth and input-tensor panels."""

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

from .. import common, style
from ._common import COMPARATORS, METHOD_COLORS, METHOD_LABELS
from ._data import LABELLINGS, abs_cos, both, capture_margin, load, lr_columns, short
from ._layout import LABELLING_HANDLES, LABELLING_MARKERS, SLOTS, method_handles
from ._panels import _frame, label_points
from ._strips import ASSOCIATION_NAMES

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


def reach(axs) -> None:
    """cccRISE's programs' LR containment in each comparator fit; contained
    cells framed."""
    ax = axs[0]
    values, contained = [], []
    for method, lab in GROUPS:
        t = load("containment", lab)
        t = t[
            (t["Of"] == "cccRISE") & (t["In"] == method) & (t["Mode"] == "lr")
        ].sort_values("Component")
        values.append(t["Containment"].to_numpy())
        contained.append(t["Contained"].to_numpy())
    values, contained = np.array(values).T, np.array(contained).T
    image = ax.imshow(values, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if v > 0.7 else style.COLORS["ink"],
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


def columns_in_cccrise(axs) -> None:
    """One strip per comparator fit: its LR columns coloured by their LR
    containment in cccRISE's span, contained ones framed."""
    ax = axs[0]
    gap, label_room, x, n_max = (
        0.9,
        0.9,
        0.6,
        0,
    )  # in cell widths; x clears the panel letter
    for method, lab in GROUPS:
        t = load("containment", lab)
        t = t[
            (t["Of"] == method) & (t["In"] == "cccRISE") & (t["Mode"] == "lr")
        ].sort_values("Component")
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
        [*_column_name(k), f"{cos:.2f}", capture_margin(reference, k, cos)]
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


def _strongest(t) -> pd.DataFrame:
    """Each component's lowest-FDR association (ties to the larger η²)."""
    return (
        t.sort_values(["PAdj", "EtaSquared"], ascending=[True, False])
        .drop_duplicates(["Labelling", "Method", "Component"])
        .assign(NegLog=lambda d: -np.log10(d["PAdj"]))
    )


def _significance_box(ax, t, sig) -> None:
    """A boxed list of every association each of `sig` is significant for."""
    text = {"size": style.FONT_SIZE_DENSE - 2}
    names, lists = [], []
    for r in sig.itertuples():
        hits = t[
            (t["Labelling"] == r.Labelling)
            & (t["Method"] == r.Method)
            & (t["Component"] == r.Component)
            & (t["PAdj"] < common.SIG_ALPHA)
        ].sort_values("PAdj")
        hits = [ASSOCIATION_NAMES[a] for a in hits["Association"]]
        listed = ",\n".join(", ".join(hits[i : i + 2]) for i in range(0, len(hits), 2))
        names.append(
            TextArea(
                short(r.Method, r.Component) + "\n" * listed.count("\n"),
                textprops={**text, "weight": "bold"},
            )
        )
        lists.append(TextArea(listed, textprops=text))
    table = HPacker(
        children=[VPacker(children=names, sep=2), VPacker(children=lists, sep=2)], sep=6
    )
    box = AnchoredOffsetbox(
        loc="upper left",
        child=table,
        pad=0.3,
        borderpad=0.3,
        frameon=True,
        bbox_to_anchor=(0.12, 0.93),
        bbox_transform=ax.transAxes,
    )
    box.patch.set(edgecolor=style.COLORS["ink"], lw=0.6)
    ax.add_artist(box)


def strongest_association(axs) -> None:
    """Each component's strongest association: η² against −log10 FDR,
    cccRISE and every comparator fit; the significant ones labelled and
    their significant associations listed in the box."""
    ax = axs[0]
    t = pd.concat([a.assign(Labelling=lab) for lab, a in both("associations").items()])
    t = t[
        t["Method"].isin(("cccRISE", *COMPARATORS))
        & ((t["Method"] != "cccRISE") | (t["Labelling"] == "coarse"))
    ]
    best = _strongest(t)
    for method, lab, _ in SLOTS:
        g = best[(best["Method"] == method) & (best["Labelling"] == lab)]
        ax.scatter(
            g["EtaSquared"],
            g["NegLog"],
            marker=LABELLING_MARKERS[lab],
            s=style.POINT_SIZE,
            color=METHOD_COLORS[method],
            edgecolor="white",
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
        xlabel=r"$\eta^2$ (strongest association)",
        ylabel=r"$-\log_{10}$ FDR",
        xlim=(0, None),
        ylim=(0, best["NegLog"].max() * 1.3),
    )
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    sig = best[best["PAdj"] < common.SIG_ALPHA].sort_values("PAdj")
    label_points(
        ax,
        sig["EtaSquared"].to_numpy(),
        sig["NegLog"].to_numpy(),
        [short(m, c) for m, c in zip(sig["Method"], sig["Component"], strict=True)],
        leaders=True,
    )
    _significance_box(ax, t, sig)


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


def coarse_to_fine(axs) -> None:
    """Each comparator's coarse LR columns (left) joined to its fine ones
    (right) where LR |cos| >= `FLOW_FROM`, line width stepping with |cos|."""
    ax = axs[0]
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
    for m in t["Method"].unique():
        g = t[t["Method"] == m]
        z = (g["Mean"] - g["Mean"].mean()) / g["Mean"].std()
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
