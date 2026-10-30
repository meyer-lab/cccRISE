"""Result panels (pre-registration §4C-F, §5) and one sensitivity figure per
threshold (§6)."""

import numpy as np
import pandas as pd

from .. import style
from . import _common as C

QUANTITIES = ("loading", "contribution", "expressed")
Q_COLORS = dict(
    zip(
        QUANTITIES,
        (style.COLORS["ink"], style.COLORS["emphasis"], style.CATEGORICAL[2]),
    )
)
SIDE_ROWS = C.side_rows()
N_PROGRAMS = len(C.programs())
IDENTITY_COLORS = dict(
    zip(
        ("mislabelled-carrier-like", "other state", "doublet-like", "ambient-like"),
        (
            style.CATEGORICAL[0],
            style.CATEGORICAL[2],
            style.CATEGORICAL[3],
            style.CATEGORICAL[7],
        ),
    )
)


def side_label(k, s):
    return f"C{k} {s[:4]}. ({'/'.join(c.split()[0] for c in C.carriers(k, s))})"


def label_side_rows(ax, y) -> None:
    """Program-side names down the first axes, first program on top."""
    ax.set_yticks(
        y, [side_label(k, s) for k, s in SIDE_ROWS], fontsize=style.FONT_SIZE_DENSE
    )
    ax.invert_yaxis()
    ax.legend(fontsize=style.FONT_SIZE_DENSE, frameon=False, loc="lower right")


def shares_figure():
    t = C.table("shares")
    groups = [
        ("noncarrier", "pos", "Outside label, positive\n(pre-registered)"),
        ("noncarrier", "net", "Outside label, net"),
        (
            "noncarrier_beyond_null",
            "pos",
            "Outside label, beyond\nper-cell null (p ≤ 0.01)",
        ),
        ("same_lineage", "pos", "Same lineage as label,\npositive"),
    ]
    fig = C.figure(180, 120)
    axs = fig.subplots(1, len(groups), sharey=True)
    y = np.arange(len(SIDE_ROWS))
    for ax, (group, kind, title) in zip(axs, groups, strict=True):
        for j, q in enumerate(QUANTITIES):
            sel = (
                t[(t.group == group) & (t.quantity == q)]
                .set_index(["component", "side"])
                .reindex(SIDE_ROWS)
            )
            off = (j - 1) * 0.25
            ax.errorbar(
                sel[kind],
                y + off,
                xerr=[sel[kind] - sel[f"{kind}_lo"], sel[f"{kind}_hi"] - sel[kind]],
                fmt="o",
                ms=2,
                lw=0.6,
                color=Q_COLORS[q],
                label=C.QUANTITY_LABELS[q],
            )
        for x in (0.1, 0.2):
            ax.axvline(x, color=style.COLORS["reference"], ls="--", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(title, fontsize=style.FONT_SIZE_DENSE)
        ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
        ax.set_xlabel(
            "Share of the program side's weight", fontsize=style.FONT_SIZE_DENSE
        )
    label_side_rows(axs[0], y)
    return fig


def types_figure(quantity: str):
    t = C.table("types")
    t = t[t.quantity == quantity]
    order = C.fine_order()
    cols = [side_label(k, s) for k, s in SIDE_ROWS]
    z = pd.DataFrame(index=order, columns=cols, dtype=float)
    sig = pd.DataFrame(False, index=order, columns=cols)
    for (k, s), col in zip(SIDE_ROWS, cols, strict=True):
        sel = t[(t.component == k) & (t.side == s)].set_index("celltype")
        z[col] = (sel["z_mean"] - sel["null_mean"]) / (
            sel["null_q99"] - sel["null_mean"]
        ).replace(0, np.nan)
        sig[col] = (sel["q"] < 0.05).reindex(order).fillna(False)
    fig = C.figure(150, 190)
    ax = fig.add_subplot()
    lim = np.nanpercentile(np.abs(z.to_numpy(float)), 99)
    C.heatmap(
        ax,
        z,
        style.CMAPS["diverging"],
        -lim,
        lim,
        "type mean z, in units of the\nnull's (99th pct - mean)",
    )
    C.lineage_separators(ax, order)
    for j, (k, s) in enumerate(SIDE_ROWS):
        for i, ct in enumerate(order):
            if ct in C.carriers(k, s):
                ax.text(
                    j,
                    i,
                    "C",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="k",
                    fontweight="bold",
                )
            elif sig.iloc[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=6, color="k")
    ax.set_title(
        f"{C.QUANTITY_LABELS[quantity]}: non-carrier types beyond the random-direction null (* BH q < 0.05; C = carrier label)",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def identity_figure():
    t = C.table("identity")
    classes, colors = list(IDENTITY_COLORS), IDENTITY_COLORS
    fig = C.figure(180, 90)
    axs = fig.subplots(1, 3, sharey=True)
    y = np.arange(len(SIDE_ROWS))
    for ax, q in zip(axs, QUANTITIES, strict=True):
        tot = (
            t[t.quantity == q]
            .groupby(["component", "side", "class"])["n"]
            .sum()
            .unstack(fill_value=0)
            .reindex(SIDE_ROWS)
            .fillna(0)
        )
        n = tot.sum(axis=1)
        left = np.zeros(len(y))
        for c in classes:
            frac = (tot.get(c, 0) / n.replace(0, np.nan)).fillna(0).to_numpy()
            ax.barh(y, frac, left=left, color=colors[c], label=c)
            left += frac
        for yi, ni in zip(y, n, strict=True):
            ax.text(1.01, yi, f"{int(ni):,}", va="center", fontsize=5)
        ax.set_title(
            f"{C.QUANTITY_LABELS[q]}: carrier-like non-carrier cells",
            fontsize=style.FONT_SIZE_DENSE,
        )
        ax.set_xlim(0, 1.15)
        ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    axs[0].set_yticks(
        y, [side_label(k, s) for k, s in SIDE_ROWS], fontsize=style.FONT_SIZE_DENSE
    )
    axs[0].invert_yaxis()
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=4,
        fontsize=style.FONT_SIZE_DENSE,
        frameon=False,
    )
    return fig


def identity_types_figure():
    """Which fine types the carrier-like cells come from (contribution)."""
    t = C.table("identity")
    t = t[t.quantity == "contribution"]
    fig = C.figure(180, 120)
    axs = fig.subplots(2, N_PROGRAMS)
    for ax, (k, s) in zip(axs.T.ravel(), SIDE_ROWS, strict=True):
        sel = (
            t[(t.component == k) & (t.side == s)]
            .groupby("celltype")["n"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        ax.barh(range(len(sel)), sel.to_numpy(), color=style.COLORS["emphasis"])
        ax.set_yticks(range(len(sel)), [x[:22] for x in sel.index], fontsize=4.5)
        ax.invert_yaxis()
        ax.set_title(side_label(k, s), fontsize=5)
        ax.tick_params(axis="x", labelsize=5)
    fig.suptitle(
        "Fine types of carrier-like cells (contribution; top 8 per program side)",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def patients_figure():
    p = C.table("patients")
    fig = C.figure(180, 100)
    axs = fig.subplots(2, N_PROGRAMS)
    for j, k in enumerate(C.programs()):
        sel = p[p.component == k]
        a2 = sel["A"] ** 2 / (sel["A"] ** 2).sum()
        ax = axs[0, j]
        ax.scatter(
            sel["n_carrier_receiver"] + 0.5,
            a2,
            s=3,
            c=style.COLORS["emphasis"],
            linewidths=0,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(5.5, color=style.COLORS["reference"], ls="--", lw=0.5)
        ax.set_title(C.program_label(k), fontsize=style.FONT_SIZE_DENSE)
        ax.set_xlabel("receiver carrier cells + 0.5", fontsize=5)
        ax.tick_params(labelsize=5)
        ax2 = axs[1, j]
        for q in QUANTITIES:
            f = sel[f"f_out_receiver_{q}"]
            ax2.scatter(
                f, a2, s=3, c=Q_COLORS[q], linewidths=0, label=C.QUANTITY_LABELS[q]
            )
        ax2.set_yscale("log")
        ax2.axvline(0.5, color=style.COLORS["reference"], ls="--", lw=0.5)
        ax2.set_xlabel("receiver f_out (non-carrier share)", fontsize=5)
        ax2.tick_params(labelsize=5)
    axs[0, 0].set_ylabel("sample share of ΣA²", fontsize=style.FONT_SIZE_DENSE)
    axs[1, 0].set_ylabel("sample share of ΣA²", fontsize=style.FONT_SIZE_DENSE)
    axs[1, 0].legend(fontsize=5, frameon=False)
    return fig


def representability_figure():
    t = C.table("representability")
    fig = C.figure(180, 80)
    axs = fig.subplots(1, 3, sharey=True)
    y = np.arange(len(SIDE_ROWS))
    cols = [
        ("r2_sample_fine", "sample × fine", style.COLORS["emphasis"]),
        ("r2_sample_shuffled_mean", "sample × shuffled fine", style.COLORS["muted"]),
        ("r2_sample_lineage", "sample × lineage", style.CATEGORICAL[1]),
    ]
    for ax, q in zip(axs, QUANTITIES, strict=True):
        sel = t[t.quantity == q].set_index(["component", "side"]).reindex(SIDE_ROWS)
        for i, (col, label, color) in enumerate(cols):
            ax.barh(y + (i - 1) * 0.27, sel[col], height=0.27, color=color, label=label)
        ax.set_title(
            f"{C.QUANTITY_LABELS[q]}: R² of a label-mean model",
            fontsize=style.FONT_SIZE_DENSE,
        )
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    label_side_rows(axs[0], y)
    return fig


LEVEL_COLUMNS = [
    (q, c) for q in QUANTITIES for c in ("T11 low", "reference", "T11 high")
]


def level_heatmap(ax, frame: pd.DataFrame, rows: list[str], title: str):
    """Levels (rows x quantity/cut columns), each cell labelled with its level."""
    frame = frame.reindex(columns=LEVEL_COLUMNS)
    im = ax.imshow(
        frame.to_numpy(float), cmap=style.CMAPS["weight"], vmin=0, vmax=3, aspect="auto"
    )
    ax.set_yticks(range(len(rows)), rows, fontsize=5)
    ax.set_xticks(
        range(frame.shape[1]),
        [f"{q[:4]} {c.replace('T11 ', '')[:4]}" for q, c in frame.columns],
        fontsize=5,
        rotation=90,
    )
    for (i, j), v in np.ndenumerate(frame.to_numpy(float)):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5)
    ax.set_title(title, fontsize=style.FONT_SIZE_DENSE)
    return im


def levels_figure():
    t = C.table("levels")
    fig = C.figure(180, 75)
    axs = fig.subplots(1, 3, width_ratios=[3, 3, 1.3])
    side_rows = [side_label(k, s) for k, s in SIDE_ROWS]
    for ax, variant in zip(
        axs[:2], ("pos (pre-registered)", "beyond null"), strict=True
    ):
        sel = t[(t.hypothesis == "H1") & (t.share_variant == variant)]
        frame = sel.pivot_table(
            index=["component", "side"], columns=["quantity", "cuts"], values="level"
        ).reindex(SIDE_ROWS)
        im = level_heatmap(ax, frame, side_rows, f"H1 level, share = {variant}")
    h2 = t[t.hypothesis == "H2"].pivot_table(
        index="component", columns=["quantity", "cuts"], values="level"
    )
    level_heatmap(axs[2], h2, [C.program_label(k) for k in h2.index], "H2 level")
    cb = fig.colorbar(im, ax=axs, fraction=0.015, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(C.LEVEL_NAMES, fontsize=5)
    return fig


# --- §6 sensitivity ------------------------------------------------------------------


def sweep_panel(ax, rows: pd.DataFrame, quantities) -> None:
    """One program's sweep curves: sender solid, receiver dashed, one colour
    per quantity."""
    for side, ls in (("sender", "-"), ("receiver", "--")):
        for q in quantities:
            sel = rows[(rows.side == side) & (rows.quantity == q)]
            if sel.empty:
                continue
            ax.plot(
                pd.to_numeric(sel["x"], errors="coerce"),
                sel["value"],
                ls=ls,
                marker="o",
                ms=1.5,
                lw=0.7,
                color=Q_COLORS.get(q, style.COLORS["ink"]),
                label=f"{side}, {C.QUANTITY_LABELS.get(q, q)}",
            )


def sweep_grid(
    threshold: str,
    metrics: list[str],
    xlabel: str,
    quantities=QUANTITIES,
    refs=(),
    log_x=False,
    ylabel=None,
):
    t = C.table("sweeps")
    t = t[t.threshold == threshold]
    fig = C.figure(180, 45 * len(metrics) + 15)
    axs = np.atleast_2d(
        fig.subplots(len(metrics), N_PROGRAMS, sharex=True, squeeze=False)
    )
    for i, metric in enumerate(metrics):
        for j, k in enumerate(C.programs()):
            ax = axs[i, j]
            sweep_panel(ax, t[(t.metric == metric) & (t.component == k)], quantities)
            for r in refs:
                ax.axvline(r, color=style.COLORS["reference"], ls=":", lw=0.6)
            if log_x:
                ax.set_xscale("log")
            ax.tick_params(labelsize=5)
            if i == 0:
                ax.set_title(C.program_label(k), fontsize=style.FONT_SIZE_DENSE)
            if j == 0:
                ax.set_ylabel(ylabel or metric, fontsize=5)
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel, fontsize=5)
    handles, labels = axs[0, 2].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncols=3, fontsize=5, frameon=False
    )
    fig.suptitle(
        f"{threshold}: dotted line = pre-registered reference",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def t01():
    return sweep_grid(
        "T1",
        [
            "S_out_pos_power2",
            "S_out_net_power2",
            "S_out_pos_power1",
            "r2_sample_fine_power2",
        ],
        "LR-pair cut ρ (|D| ≥ ρ·max); 0 = all pairs",
        quantities=("contribution",),
        refs=(0.0, 0.4),
    )


def t02():
    return sweep_grid(
        "T2",
        ["S_out_pos_topn"],
        "carrier set = top-n fine types by positive share",
        refs=(1,),
    )


def t03():
    return sweep_grid(
        "T3",
        ["n_carrier_like", "share_carrier_like"],
        "carrier-like cut: percentile of labelled carriers",
        refs=(50,),
    )


def t04():
    return sweep_grid(
        "T4", ["energy_outside_Vm"], "m (≥ m carrier cells on both sides)", refs=(5,)
    )


def t04b():
    return sweep_grid(
        "T4",
        ["frac_top_f_out_ge0.25", "frac_top_f_out_ge0.5", "frac_top_f_out_ge0.75"],
        "top-effective set (share of ΣA²)",
        refs=(0.8,),
    )


def t05():
    t = C.table("sweeps")
    t = t[t.threshold == "T5"]
    fig = C.figure(180, 70)
    ax = fig.add_subplot()
    for (k, side), sel in t.groupby(["component", "side"]):
        x = pd.to_numeric(sel["x"])
        ax.plot(
            x,
            sel["value"],
            marker="o",
            ms=2,
            lw=0.8,
            ls="-" if side == "sender" else "--",
            color=style.CATEGORICAL[k - 1],
            label=f"C{k} {side}",
        )
        last = sel[pd.to_numeric(sel["x"]) == 0.4]["detail"].iloc[0]
        ax.text(
            0.41,
            float(sel[pd.to_numeric(sel["x"]) == 0.4]["value"].iloc[0]),
            f"C{k}{side[0]}: {last}",
            fontsize=4.5,
            color=style.CATEGORICAL[k - 1],
        )
    ax.axvline(0.4, color=style.COLORS["reference"], ls=":", lw=0.6)
    ax.set_xlabel(
        "eigenstate cut (|factor| ≥ cut · max)", fontsize=style.FONT_SIZE_DENSE
    )
    ax.set_ylabel("eigenstates selected", fontsize=style.FONT_SIZE_DENSE)
    ax.legend(fontsize=5, ncols=4, frameon=False)
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    return fig


def t06():
    return sweep_grid(
        "T6",
        ["share_beyond_null", "n_beyond_null"],
        "per-cell null percentile",
        refs=(99,),
    )


def t06b():
    """Number of significant non-carrier types at each q cut."""
    t = C.table("types")
    t = t[~t.carrier]
    rows = []
    for qcut in (0.01, 0.05, 0.1):
        g = (
            t.assign(sig=t["q"] < qcut)
            .groupby(["component", "side", "quantity"])["sig"]
            .sum()
            .rename("value")
            .reset_index()
        )
        rows.append(g.assign(x=qcut))
    d = pd.concat(rows)
    fig = C.figure(180, 55)
    axs = fig.subplots(1, N_PROGRAMS, sharey=True)
    for ax, k in zip(axs, C.programs(), strict=True):
        for side, ls in (("sender", "-"), ("receiver", "--")):
            for q in QUANTITIES:
                sel = d[(d.component == k) & (d.side == side) & (d.quantity == q)]
                ax.plot(
                    sel["x"],
                    sel["value"],
                    ls=ls,
                    marker="o",
                    ms=2,
                    lw=0.7,
                    color=Q_COLORS[q],
                )
        ax.axvline(0.05, color=style.COLORS["reference"], ls=":", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(C.program_label(k), fontsize=style.FONT_SIZE_DENSE)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("BH q cut", fontsize=5)
    axs[0].set_ylabel("non-carrier types significant", fontsize=5)
    return fig


def t07():
    return sweep_grid(
        "T7",
        ["n_max", "carrier_frac_max", "n_q99.9", "carrier_frac_q99.9"],
        "GROUP_TAU (Fig 2L split)",
        refs=(0.3,),
    )


def t09():
    t = C.table("paint")
    fig = C.figure(140, 60)
    ax = fig.add_subplot()
    t = t.assign(
        row=t["component"].map(lambda k: f"C{k}")
        + " "
        + t["method"].map(C.METHOD_SHORT)
    )
    frame = t.pivot(index="row", columns="cut", values="labelling").fillna("none")
    labs = sorted(set(frame.to_numpy().ravel()))
    codes = frame.apply(lambda c: c.map({v: i for i, v in enumerate(labs)}))
    ax.imshow(codes.to_numpy(float), cmap="tab20", aspect="auto")
    for (i, j), v in np.ndenumerate(frame.to_numpy()):
        ax.text(j, i, v, ha="center", va="center", fontsize=5)
    ax.set_yticks(range(len(frame)), frame.index, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xticks(
        range(frame.shape[1]),
        [f"≥ {c}" for c in frame.columns],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_xlabel(
        "LR containment cut for 'fit contains the program'",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_title(
        "T9: fitted-comparator labelling selected (reference 0.85)",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def t10():
    t = C.table("sweeps")
    t = t[(t.threshold == "T10") & (t.quantity == "contribution")]
    fig = C.figure(180, 110)
    axs = fig.subplots(2, N_PROGRAMS, sharey=True)
    classes, colors = list(IDENTITY_COLORS), IDENTITY_COLORS
    for ax, (k, s) in zip(axs.T.ravel(), SIDE_ROWS, strict=True):
        sel = t[(t.component == k) & (t.side == s)].pivot(
            index="x", columns="metric", values="value"
        )
        left = np.zeros(len(sel))
        for c in classes:
            ax.barh(range(len(sel)), sel[c], left=left, color=colors[c], label=c)
            left += sel[c].to_numpy()
        ax.set_yticks(range(len(sel)), sel.index, fontsize=4)
        ax.set_title(side_label(k, s), fontsize=5)
        ax.tick_params(axis="x", labelsize=5)
    axs[0, 0].set_ylabel("hi pct / lo pct / n_genes pct\n(ref 25/5/95)", fontsize=5)
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=4,
        fontsize=5,
        frameon=False,
    )
    fig.suptitle(
        "T10: identity class composition of carrier-like cells (contribution) across cuts",
        fontsize=style.FONT_SIZE_DENSE,
    )
    return fig


def t11():
    t = C.table("levels")
    fig = C.figure(180, 60)
    axs = fig.subplots(1, 2)
    for ax, hyp in zip(axs, ("H1", "H2"), strict=True):
        sel = t[t.hypothesis == hyp]
        frame = (
            sel.groupby(["cuts", "level"])
            .size()
            .unstack(fill_value=0)
            .reindex(["T11 low", "reference", "T11 high"])
        )
        left = np.zeros(len(frame))
        for lvl in frame.columns:
            ax.barh(
                range(len(frame)),
                frame[lvl],
                left=left,
                color=style.CMAPS["weight"](lvl / 3),
                label=C.LEVEL_NAMES[int(lvl)],
            )
            left += frame[lvl].to_numpy()
        ax.set_yticks(range(len(frame)), frame.index, fontsize=style.FONT_SIZE_DENSE)
        ax.set_title(
            f"T11: {hyp} level counts as the level boundaries move ± half",
            fontsize=style.FONT_SIZE_DENSE,
        )
        ax.legend(fontsize=5, frameon=False)
        ax.tick_params(labelsize=5)
    return fig


def t12():
    t = C.table("sweeps")
    t = t[t.threshold == "T12"].pivot(
        index="component", columns="metric", values="value"
    )
    fig = C.figure(120, 50)
    axs = fig.subplots(1, 2)
    for ax, (a, b, title) in zip(
        axs,
        (
            ("effective_pairs", "pairs_ge_0.4max", "LR pairs"),
            ("effective_samples", "samples_ge_0.4max", "Samples"),
        ),
        strict=True,
    ):
        x = np.arange(len(t))
        ax.bar(
            x - 0.2,
            t[a],
            0.4,
            label="effective (inverse Simpson)",
            color=style.COLORS["emphasis"],
        )
        ax.bar(
            x + 0.2, t[b], 0.4, label="count ≥ 0.4 × max", color=style.COLORS["muted"]
        )
        ax.set_xticks(x, [f"C{k}" for k in t.index], fontsize=style.FONT_SIZE_DENSE)
        ax.set_title(title, fontsize=style.FONT_SIZE_DENSE)
        ax.set_yscale("log")
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5, frameon=False)
    return fig
