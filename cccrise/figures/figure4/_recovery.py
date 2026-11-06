"""Figure 4's view of the cccRISE programs against the comparators: where each
sits between its best column and its label ceiling, what each comparator column
is, and how STAC arranges the mixed programs' pairs."""

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .. import common, style
from ._ceiling import _cohort_labels
from ._common import DATASET_COLORS, DATASET_LABELS, METHOD_COLORS, load
from ._data import LABELLINGS, beyond, short
from ._programs import (
    COLUMN_CLASSES,
    FITS,
    best_columns,
    column_census,
    lr_matrix,
    program_homes,
    tested_columns,
)

#: Heatmap cell edges.
GRID = {"ec": "black", "lw": 0.3}
HEAT = style.CMAPS["weight"]
#: Outline of a column tested in the next panel.
LINK = "#d00000"
#: Programs labelled left of their point.
LEFT_LABELS = {("ibd", 3), ("alad", 5), ("ibd", 1), ("ibd", 6), ("alad", 2)}
#: Where the mixed programs' panels look.
MIXED_DATASET, MIXED_PROGRAMS = "alad", (3, 4)
#: Gap between a heatmap's blocks of columns, in columns.
BLOCK_GAP = 0.3
LOADING_LABEL = "|Loading| / max |loading|"
COS2_LABEL = "Max cos$^2$"
LABEL_SIZE = style.FONT_SIZE_DENSE - 0.7


def panel_title(ax, text: str, pad: float = 3) -> None:
    ax.set_title(text, fontsize=style.FONT_SIZE_LABEL, fontweight="bold", pad=pad)


def colorbar_under(ax, label: str, drop: float, x0=0.2, width=0.6) -> None:
    """A horizontal colorbar `drop` of `ax`'s height below it."""
    cax = ax.inset_axes((x0, -drop, width, 0.04))
    ax.figure.colorbar(
        ScalarMappable(Normalize(0, 1), HEAT), cax=cax, orientation="horizontal"
    )
    cax.set_xticks([0, 1])
    cax.tick_params(labelsize=style.FONT_SIZE_DENSE, length=1.5, pad=1)
    cax.set_xlabel(label, fontsize=style.FONT_SIZE_DENSE, labelpad=0)


def _map_rows(labelling: str) -> pd.DataFrame:
    frames = []
    for dataset in DATASET_LABELS:
        best = best_columns(dataset)
        best = best[best["Labelling"] == labelling].groupby("Program")["Cos2"].max()
        ceiling = load("label_ceiling", labelling, dataset)
        sides = ceiling.groupby(["Component", "Side"])["Ceiling"].first().unstack()
        frames.append(
            sides.assign(Dataset=dataset, Cos2=best)
            .rename_axis("Program")
            .reset_index()
        )
    return pd.concat(frames, ignore_index=True)


def program_map(
    ax,
    labelling: str = "fine",
    spots: dict | None = None,
    title: str = "Label-based recovery of cccRISE programs",
) -> None:
    """Each program's best single column of `labelling`'s fits against the
    per-cell weight its labels can hold."""
    t = _map_rows(labelling)
    common.source_data(
        f"program_cos2_and_label_ceiling_{labelling}",
        t.rename(columns={"sender": "Sender R2", "receiver": "Receiver R2"}),
        "Per program: its largest cos^2 with one LR column of the comparator fits, "
        f"and the R2 of its per-cell sender and receiver weights on {labelling} labels.",
    )
    for r in t.itertuples():
        color = DATASET_COLORS[r.Dataset]
        ax.plot([r.Cos2] * 2, [r.sender, r.receiver], color=color, lw=0.8, zorder=1)
        ax.scatter(r.Cos2, r.sender, s=13, color=color, ec="white", lw=0.3, zorder=3)
        ax.scatter(
            r.Cos2,
            r.receiver,
            s=11,
            marker="s",
            color=color,
            ec="white",
            lw=0.3,
            zorder=3,
        )
        left = (r.Dataset, r.Program) in LEFT_LABELS
        x, y, ha = (spots or {}).get(
            (r.Dataset, r.Program),
            (
                r.Cos2 + (-0.02 if left else 0.02),
                (r.sender + r.receiver) / 2,
                "right" if left else "left",
            ),
        )
        ax.text(x, y, f"C{r.Program}", ha=ha, va="center", fontsize=LABEL_SIZE)
    ax.set(
        xlim=(0, 1.1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        xlabel=COS2_LABEL,
        ylabel="Per-cell weight $R^2$",
    )
    panel_title(ax, title)
    ax.legend(
        handles=[
            *[
                Line2D(
                    [], [], marker="o", ls="", color=c, ms=3, label=DATASET_LABELS[d]
                )
                for d, c in DATASET_COLORS.items()
            ],
            Line2D([], [], marker="o", ls="", color="0.4", ms=3, label="Sender"),
            Line2D([], [], marker="s", ls="", color="0.4", ms=3, label="Receiver"),
        ],
        loc="upper left",
        fontsize=LABEL_SIZE,
    )


def _census_bar(ax, y: float, counts: pd.Series) -> None:
    left = 0
    for name, color in COLUMN_CLASSES.items():
        n = int(counts.get(name, 0))
        if n:
            ax.barh(y, n, left=left, height=0.78, color=color, ec="white", lw=0.4)
            light = name in ("Part of a cccRISE program", "Broad", "Single pair")
            ax.text(
                left + n / 2,
                y,
                str(n),
                ha="center",
                va="center",
                fontsize=style.FONT_SIZE_DENSE - 1.5,
                color=style.COLORS["ink"] if light else "white",
            )
            left += n


def census(ax) -> None:
    """How many LR columns of each class each fit has, ALAD over IBD."""
    tables = {dataset: column_census(dataset) for dataset in DATASET_LABELS}
    common.source_data(
        "comparator_column_classes",
        pd.concat(tables.values(), ignore_index=True)[
            [
                "Dataset",
                "Method",
                "Labelling",
                "Component",
                "Containment",
                "Contained",
                "OnProgram",
                "Program",
                "EffectiveLR",
                "HomophilicShare",
                "Class",
            ]
        ],
        "Every comparator LR column: its containment in cccRISE's LR span, its largest "
        "share of squared loading on one program's carrying pairs (OnProgram, Program), "
        "its effective LR pairs, its homophilic share, and its class.",
    )
    y, ticks, names, centers = 0.0, [], [], {}
    for dataset, t in tables.items():
        rows = []
        for method, lab in FITS:
            fit = t[(t["Method"] == method) & (t["Labelling"] == lab)]
            _census_bar(ax, y, fit["Class"].value_counts())
            ticks.append(y)
            names.append(f"{method} {lab}")
            rows.append(y + 0.5)
            y += 1
        centers[dataset] = np.array(rows) - 0.5
        y += 0.7
    ax.set_yticks(ticks, names, fontsize=style.FONT_SIZE_DENSE - 0.8)
    ax.set(xlim=(0, 16.2), ylim=(y - 1.1, -0.6), xlabel="LR columns")
    _cohort_labels(ax, centers)
    ax.legend(
        handles=[Patch(color=c, label=k) for k, c in COLUMN_CLASSES.items()],
        loc="lower left",
        bbox_to_anchor=(-0.02, 1),
        ncol=3,
        fontsize=LABEL_SIZE,
        columnspacing=0.8,
        handlelength=0.9,
        handletextpad=0.3,
    )


def _cell(ax, x: float, y: float, value: float, text: bool = False) -> None:
    ax.add_patch(Rectangle((x, y), 1, 1, fc=HEAT(value), **GRID))
    if text:
        ax.text(
            x + 0.5,
            y + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if value > 0.6 else style.COLORS["ink"],
        )


def _clean_heatmap(ax) -> None:
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def best_column_grid(ax) -> None:
    """Each program's best single column in each fit, ALAD over IBD."""
    t = pd.concat(
        [best_columns(d).assign(Dataset=DATASET_LABELS[d]) for d in DATASET_LABELS]
    )
    common.source_data(
        "program_best_column_cos2",
        t.rename(columns={"Cos2": "Max cos2"}),
        "Each program's largest cos^2 with one LR column of each comparator fit.",
    )
    y, ticks, names, centers = 0.0, [], [], {}
    for dataset, label in DATASET_LABELS.items():
        grid = t[t["Dataset"] == label].pivot_table(
            index="Program", columns=["Method", "Labelling"], values="Cos2"
        )
        centers[dataset] = y + np.arange(len(grid)) + 0.5
        for program, row in grid.iterrows():
            for j, fit in enumerate(FITS):
                _cell(ax, j, y, row[fit], text=True)
            ticks.append(y + 0.5)
            names.append(f"C{program}")
            y += 1
        y += 0.5
    ax.set(xlim=(0, len(FITS)), ylim=(y - 0.5, 0))
    ax.set_yticks(ticks, names)
    ax.set_xticks(np.arange(len(FITS)) + 0.5, [lab[0].upper() for _, lab in FITS])
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    _clean_heatmap(ax)
    for i, method in enumerate(dict.fromkeys(m for m, _ in FITS)):
        ax.text(
            2 * i + 1,
            -0.25,
            method,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    _cohort_labels(ax, centers)
    colorbar_under(ax, COS2_LABEL, drop=0.13)


def _pair_rows(dataset: str, programs) -> tuple[list, dict]:
    """Each program's carrying pairs, heaviest first, at their y, and each
    program's span of y."""
    rows, spans, y = [], {}, 0.0
    for program in programs:
        homes = program_homes(dataset, program, "coarse")
        pairs = homes.sort_values("Share", ascending=False)["LRPair"]
        spans[program] = (y, y + len(pairs))
        rows += [(program, pair, y + i) for i, pair in enumerate(pairs)]
        y += len(pairs) + 0.7
    return rows, spans


def _column_block(ax, x: float, rows, columns: pd.DataFrame, names) -> float:
    """`columns`' |loadings| of `rows`' pairs, each scaled to its heaviest
    pair of all; the x after them."""
    for name in names:
        scale = columns[name].abs().max()
        for _, pair, y in rows:
            value = abs(columns[name].get(pair, 0)) / scale
            _cell(ax, x, y, value)
        x += 1
    return x


def _cccrise_block(ax, dataset: str, rows) -> list:
    """`rows`' pairs in their programs, at x = 0; the values drawn."""
    cccrise = lr_matrix(dataset, "cccRISE")
    values = []
    for program, pair, y in rows:
        value = abs(cccrise.loc[pair, program]) / cccrise[program].abs().max()
        _cell(ax, 0.0, y, value)
        values.append(("cccRISE", "", f"C{program}", pair, value))
    return values


def _held_columns(dataset: str, programs, labelling: str) -> list[int]:
    """The STAC columns holding most of any carrying pair of `programs`."""
    return sorted(
        {
            int(c)
            for p in programs
            for c in program_homes(dataset, p, labelling)["Column"].dropna()
        }
    )


def _column_values(columns: pd.DataFrame, labelling: str, name: int, rows) -> list:
    return [
        (
            "STAC",
            labelling,
            f"S{name}",
            pair,
            abs(columns[name].get(pair, 0)) / columns[name].abs().max(),
        )
        for _, pair, _ in rows
    ]


def _outline_tested(ax, x: float, column: tuple, tested, spans) -> bool:
    """Outline `column` (labelling, name) over each program it is tested
    against; whether it is tested at all."""
    programs = [p for p, lab, name in tested if (lab, name) == column]
    for program in programs:
        y0, y1 = spans[program]
        ax.add_patch(
            Rectangle(
                (x, y0),
                1,
                y1 - y0,
                fc="none",
                ec=LINK,
                lw=1.0,
                zorder=6,
                clip_on=False,
            )
        )
    return bool(programs)


def pair_columns(
    ax,
    dataset: str = MIXED_DATASET,
    programs=MIXED_PROGRAMS,
    title: str | None = "ALAD",
    outline: bool = True,
) -> None:
    """Each program's carrying pairs in the program and in the STAC columns
    holding them most, every column scaled to its own heaviest pair; the
    columns the next panel tests bold, outlined over their program's pairs."""
    rows, spans = _pair_rows(dataset, programs)
    tested = tested_columns(dataset, programs) if outline else []
    values = _cccrise_block(ax, dataset, rows)
    ticks, labels, heads, bold = [0.5], [""], {"cccRISE": 0.5}, []
    x = 1 + BLOCK_GAP
    for lab in LABELLINGS:
        columns = lr_matrix(dataset, "STAC", lab)
        names = _held_columns(dataset, programs, lab)
        start = x
        x = _column_block(ax, x, rows, columns, names)
        heads[f"STAC {lab}"] = (start + x) / 2
        for i, name in enumerate(names):
            ticks.append(start + i + 0.5)
            labels.append(f"S{name}")
            values += _column_values(columns, lab, name, rows)
            if _outline_tested(ax, start + i, (lab, name), tested, spans):
                bold.append(len(ticks) - 1)
        x += BLOCK_GAP
    common.source_data(
        "program_pairs_in_stac_columns",
        pd.DataFrame(
            values, columns=["Method", "Labelling", "Column", "LR pair", LOADING_LABEL]
        ),
        "Each program's carrying pairs in the program and in the STAC columns holding "
        "them most, each column's |loading| relative to its heaviest pair.",
    )
    _pair_axes(ax, rows, spans, x - BLOCK_GAP, (ticks, labels, bold), heads)
    if title:
        panel_title(ax, title, pad=11)
    colorbar_under(ax, LOADING_LABEL, drop=0.19, x0=0.17, width=0.66)


def _pair_axes(ax, rows, spans, end: float, ticks, heads) -> None:
    positions, labels, bold = ticks
    ax.set(xlim=(0, end), ylim=(max(s[1] for s in spans.values()), 0))
    ax.set_yticks(
        [y + 0.5 for *_, y in rows],
        [pair.replace("^", "–").replace("&", "+") for _, pair, _ in rows],
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set_xticks(positions, labels, fontsize=style.FONT_SIZE_DENSE - 0.7)
    for i in set(bold):
        ax.get_xticklabels()[i].set_fontweight("bold")
    _clean_heatmap(ax)
    for name, x in heads.items():
        ax.text(x, -0.4, name, ha="center", va="bottom", fontsize=style.FONT_SIZE_DENSE)
    for program, (y0, y1) in spans.items():
        ax.plot(
            [end + 0.25] * 2, [y0 + 0.1, y1 - 0.1], color="0.5", lw=0.6, clip_on=False
        )
        ax.text(
            end + 0.5,
            (y0 + y1) / 2,
            f"C{program}",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            clip_on=False,
        )


def _covariation_rows(dataset: str, programs) -> pd.DataFrame:
    """Each program and the STAC columns `pair_columns` outlines, in order."""
    tables = {lab: load("covariation", lab, dataset) for lab in LABELLINGS}
    tested = tested_columns(dataset, programs)
    rows, x = [], 0.0
    for program in programs:
        t = tables["coarse"]
        chosen = [
            ("cccRISE", t[(t["Method"] == "cccRISE") & (t["Component"] == program)], "")
        ]
        for _, lab, column in (c for c in tested if c[0] == program):
            s = tables[lab]
            chosen.append(
                ("STAC", s[(s["Method"] == "STAC") & (s["Component"] == column)], lab)
            )
        for method, row, lab in chosen:
            component = int(row["Component"].iloc[0])
            suffix = f" {lab[0].upper()}" if lab else ""
            rows.append(
                row.assign(
                    X=x, Name=f"{short(method, component)}{suffix}", Labelling=lab
                )
            )
            x += 1
        x += 0.7
    return pd.concat(rows, ignore_index=True)


def covariation_lollipops(
    ax, t: pd.DataFrame, suffix: str, ylabel: str, names: bool
) -> None:
    """Each row's ρ̄ against its stricter null's 95th percentile, solid above it."""
    nulls = t[[f"random_q95{suffix}", f"expression_matched_q95{suffix}"]].max(axis=1)
    above = beyond(t, suffix)
    for r, null, solid in zip(t.itertuples(), nulls, above, strict=True):
        observed = getattr(r, f"Observed{suffix}")
        color = METHOD_COLORS[r.Method]
        ax.plot([r.X, r.X], [observed, null], color="0.55", lw=0.7, zorder=1)
        ax.plot([r.X - 0.3, r.X + 0.3], [null, null], color="0.4", lw=0.9, zorder=2)
        ax.scatter(
            r.X,
            observed,
            s=13,
            lw=0.8,
            zorder=3,
            facecolor=color if solid else "white",
            edgecolor=color,
        )
    ax.set(
        ylim=(0, 1),
        yticks=[0, 0.5, 1],
        ylabel=ylabel,
        xlim=(t["X"].min() - 0.7, t["X"].max() + 0.7),
    )
    ax.set_xticks(t["X"], t["Name"] if names else [])
    ax.tick_params(axis="x", labelrotation=55, labelsize=style.FONT_SIZE_DENSE - 0.7)
    if names:
        for label in ax.get_xticklabels():
            label.set(ha="right", rotation_mode="anchor")


def program_covariation(axs, dataset=MIXED_DATASET, programs=MIXED_PROGRAMS) -> None:
    """The programs and their STAC columns, over all pairs and gene-disjoint ones."""
    t = _covariation_rows(dataset, programs)
    common.source_data(
        "program_and_stac_column_covariation",
        t[
            [
                "Name",
                "Method",
                "Labelling",
                "PairsUsed",
                "Observed",
                "random_q95",
                "expression_matched_q95",
                "Observed_disjoint",
                "random_q95_disjoint",
                "expression_matched_q95_disjoint",
            ]
        ],
        "Mean pairwise rank correlation across samples of each component's carrying "
        "pairs (the fewest carrying 90% of squared loading, at most 10), over all pairs "
        "and gene-disjoint pairs, with both nulls' 95th percentiles.",
    )
    covariation_lollipops(axs[0], t, "", r"$\bar{\rho}$, all pairs", False)
    covariation_lollipops(axs[1], t, "_disjoint", r"$\bar{\rho}$, gene-disjoint", True)
