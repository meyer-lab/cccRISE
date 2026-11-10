"""How far each component's sample loadings track detection depth, every method,
and the cohort bars Figure 5's per-fit strips sit over (`cohort_bars`)."""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, kruskal, mannwhitneyu

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, METHOD_COLORS, load
from ._programs import FITS

BAR_DROP = 10.5
BOX_WIDTH = 0.16
#: Stars sit just above the y range, at this height.
STAR_Y = 1.02
#: Each cohort's Kruskal-Wallis p, at its top left.
KW_Y = 1.22


def cohort_bars(ax, spans: dict, bar_drop: float = BAR_DROP) -> None:
    """Each cohort's bar and name `bar_drop` mm under `ax`, `spans` its first and
    last x."""
    height = ax.bbox.height / ax.figure.dpi
    drop = bar_drop * style.MM / height
    transform = ax.get_xaxis_transform()
    for dataset, (first, last) in spans.items():
        ax.plot(
            [first - 0.4, last + 0.4],
            [-drop, -drop],
            color=DATASET_COLORS[dataset],
            lw=2.5,
            transform=transform,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            (first + last) / 2,
            -drop - style.MM / height,
            DATASET_LABELS[dataset],
            transform=transform,
            ha="center",
            va="top",
            fontsize=style.FONT_SIZE_DENSE,
        )


def _fit_name(method: str, labelling: str) -> str:
    return "CCC-RISE" if method == "cccRISE" else f"{method} {labelling[0].upper()}"


def _values(dataset: str) -> dict:
    """Each fit's components' |rho| with depth, cccRISE first."""
    out = {}
    for method, lab in [("cccRISE", "coarse"), *FITS]:
        t = load("depth", lab, dataset)
        out[(method, lab)] = t.loc[t["Method"] == method, "AbsRho"].dropna().to_numpy()
    return out


def _tests(groups: dict) -> tuple[float, dict]:
    """Kruskal-Wallis p across `groups`, and each group's BH q against the
    cohort's other components (one-sided Mann-Whitney: tracks depth more)."""
    _, p = kruskal(*groups.values())
    keys = list(groups)
    ps = [
        mannwhitneyu(
            groups[k],
            np.concatenate([groups[j] for j in keys if j != k]),
            alternative="greater",
        ).pvalue
        for k in keys
    ]
    return float(p), dict(zip(keys, false_discovery_control(ps), strict=True))


def _box(ax, x: float, values) -> None:
    edge = {"color": "0.35", "lw": 0.6}
    ax.boxplot(
        values,
        positions=[x],
        widths=BOX_WIDTH,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "0.35", "lw": 0.6},
        whiskerprops=edge,
        capprops=edge,
        medianprops={"color": style.COLORS["ink"], "lw": 0.9},
        zorder=2,
    )


def depth_tracking(ax) -> None:
    """|rho| of each component's condition loadings with median genes detected
    per cell, cccRISE then each comparator fit, ALAD beside IBD: a strip over
    a thin box (quartiles, whiskers to the range) per fit. Per cohort, a
    Kruskal-Wallis test across fits; a fit is starred where its components
    track depth more than the cohort's other components (one-sided
    Mann-Whitney, BH within the cohort) at q < `common.SIG_ALPHA` and the
    Kruskal-Wallis p < `common.SIG_ALPHA`."""
    rng = np.random.default_rng(0)
    x, positions, names, spans, frames = 0.0, [], [], {}, []
    for dataset, label in DATASET_LABELS.items():
        groups = _values(dataset)
        kw, qs = _tests(groups)
        start = x
        ax.text(
            x - 0.4,
            KW_Y,
            common.p_text(kw, "Kruskal–Wallis p"),
            ha="left",
            va="top",
            fontsize=style.FONT_SIZE_DENSE,
        )
        for (method, lab), values in groups.items():
            _box(ax, x, values)
            ax.scatter(
                x + rng.uniform(-0.2, 0.2, len(values)),
                values,
                s=7,
                color=METHOD_COLORS[method],
                lw=0,
                zorder=3,
                marker="^" if lab == "fine" and method != "cccRISE" else "o",
            )
            if kw < common.SIG_ALPHA and qs[(method, lab)] < common.SIG_ALPHA:
                ax.text(
                    x,
                    STAR_Y,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=style.FONT_SIZE_LABEL,
                    fontweight="bold",
                )
            frames.append(
                pd.DataFrame(
                    {
                        "Cohort": label,
                        "Method": method,
                        "Labelling": lab,
                        "|rho|": values,
                        "Kruskal-Wallis p": kw,
                        "q (tracks depth more)": qs[(method, lab)],
                    }
                )
            )
            positions.append(x)
            names.append(_fit_name(method, lab))
            x += 1
        spans[dataset] = (start, x - 1)
        x += 0.7
    common.source_data(
        "depth_tracking_by_component",
        pd.concat(frames, ignore_index=True),
        "|Spearman correlation| between each component's absolute condition loadings "
        "and per-sample depth (median genes detected per cell); per cohort, the "
        "Kruskal-Wallis p across fits, and each fit's BH q (within the cohort) of a "
        "one-sided Mann-Whitney test that its components track depth more than the "
        f"cohort's other components; starred where both are below {common.SIG_ALPHA}.",
    )
    ax.set_xticks(
        positions,
        names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set(ylim=(0, 1.22), yticks=[0, 0.5, 1], ylabel="Depth tracking, |ρ|")
    cohort_bars(ax, spans, BAR_DROP)
