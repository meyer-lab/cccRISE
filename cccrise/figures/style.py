"""
Shared style for every manuscript figure.

Sizes follow the EMBO Press (Molecular Systems Biology) figure guidelines:
figures are 87 mm (one column) or 180 mm (two columns) wide and at most
230 mm tall including the legend; text is Arial/Helvetica at ~7-8 pt at final
size; panels are labeled with bold upper-case letters at the top left. Figures
are therefore drawn at final print size, so every size below is the printed
size.

Colors reference the Shiny explorer's scheme (`app/colors.py`: sender orange,
receiver blue, tab10 cell types) but are defined independently here, so
figure and app styling can change separately.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex

MM = 1 / 25.4  # inches per millimeter

## Fonts
FONT_FAMILY = "sans-serif"
# Arial is EMBO's first choice; the rest are metric-compatible fallbacks for
# machines without it. SVG text keeps the name, so Illustrator shows Arial.
FONT_NAMES = ["Arial", "Helvetica", "Helvetica LT Std", "Liberation Sans"]
FONT_SIZE_PANEL_LABEL = 10
FONT_SIZE_TITLE = 8
FONT_SIZE_LABEL = 7  # axis labels
FONT_SIZE_TICK = 7
FONT_SIZE_LEGEND = 7
# Below EMBO's ~7 pt; only for dense tick labels (e.g. LR pairs on a heatmap).
FONT_SIZE_DENSE = 6

## Lines (pt)
LINE_WIDTH = 1.0  # data lines
LINE_WIDTH_THIN = 0.5  # axes, ticks, box outlines, reference lines
LINE_STYLE_REFERENCE = "--"  # null, chance, and threshold lines
TICK_LENGTH = 2.5

## Markers
MARKER_SIZE = 3  # `plot(marker=...)` markersize, pt
POINT_SIZE_CELL = 1  # `scatter` area (pt^2) for per-cell embeddings
POINT_SIZE_BACKGROUND = 4  # bulk / non-emphasized scatter points
POINT_SIZE = 10
POINT_SIZE_FOREGROUND = 18  # highlighted points

## Legends
LEGEND_FRAMEON = False
LEGEND_BORDERPAD = 0.3
LEGEND_LABELSPACING = 0.3
LEGEND_HANDLELENGTH = 1.2
LEGEND_HANDLETEXTPAD = 0.4
LEGEND_BORDERAXESPAD = 0.3
LEGEND_MARKERSCALE = 0.8

## Figure sizes (inches)
FIGURE_WIDTH_SINGLE = 87 * MM
FIGURE_WIDTH_DOUBLE = 180 * MM
# 230 mm is EMBO's cap including the legend; leave room for a short one.
FIGURE_HEIGHT_MAX = 200 * MM
FIGSIZE = {
    "single_column": (FIGURE_WIDTH_SINGLE, 70 * MM),
    "double_column": (FIGURE_WIDTH_DOUBLE, 90 * MM),
    "full_page": (FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_MAX),
}
# Standalone sizes for single panels, as fractions of a two-column figure.
PANEL_HEIGHT = 50 * MM
PANEL_SIZE = {
    "third": (FIGURE_WIDTH_DOUBLE / 3, PANEL_HEIGHT),
    "half": (FIGURE_WIDTH_DOUBLE / 2, PANEL_HEIGHT),
    "two_thirds": (FIGURE_WIDTH_DOUBLE * 2 / 3, PANEL_HEIGHT),
    "full": (FIGURE_WIDTH_DOUBLE, PANEL_HEIGHT),
}

# Fixed axes sizes for figures composed with `common.compose_fixed`: every
# axes box is one of these whatever its ticks, legend, or colorbar take up,
# and rectangular and square axes share a height so a row's axes line up.
AXIS_HEIGHT = 30 * MM
AXIS_WIDTH = 46 * MM
AXIS_SIZE = {
    "rect": (AXIS_WIDTH, AXIS_HEIGHT),
    "square": (AXIS_HEIGHT, AXIS_HEIGHT),
    "full": (FIGURE_WIDTH_DOUBLE - 14 * MM, AXIS_HEIGHT),
    # Supplementary heatmaps with too many columns for "rect"
    "wide": (2.5 * AXIS_WIDTH, AXIS_HEIGHT),
    "tall": (2.5 * AXIS_WIDTH, 3 * AXIS_HEIGHT),
    # A few groups x many genes along x (dot plots)
    "strip": (1.5 * AXIS_WIDTH, 0.4 * AXIS_HEIGHT),
    # A few rows: stacked bars, small heatmaps
    "short": (AXIS_WIDTH, 0.5 * AXIS_HEIGHT),
}
PAGE_MARGIN = 1 * MM
PANEL_GAP_MIN = 1.5 * MM
PANEL_GAP_MAX = 4 * MM
ROW_GAP = 1 * MM
PANEL_LABEL_SPACE = 4 * MM  # minimum room above a row's axes for its letters
COLORBAR_WIDTH = 2 * MM
COLORBAR_PAD = 1.5 * MM
LEGEND_PAD = 1.5 * MM  # `common.legend` gap beyond an axes' decorations

## Colors
CATEGORICAL = [to_hex(c) for c in sns.color_palette("colorblind")]
_BLUE, _ORANGE, _GREEN, _VERMILLION, _PINK, _TAN, _, _GREY, _YELLOW, _SKY = CATEGORICAL
_DARK_RED = "#b3382c"
_DARK_BLUE = "#3a6b9c"

COLORS = {
    "emphasis": _DARK_BLUE,  # the one series a panel is about
    "muted": "0.75",  # de-emphasized / not significant
    "ink": "0.25",  # black-ish marks: summary points, error bars
    "reference": "0.45",  # null, chance, and threshold lines
    # Direction of an effect; matches the diverging colormap's ends.
    "up": _DARK_RED,
    "down": _DARK_BLUE,
    "not_significant": "0.75",
    "sender": _ORANGE,
    "receiver": _BLUE,
    "highlight": _GREEN,
    "fit": _BLUE,  # in-sample fit metrics
    "heldout": _VERMILLION,  # held-out / cross-validated metrics
    "fit_full": "0.25",  # the full-data fit, beside per-repeat fit metrics
}

# Component-stability axes; the seed-only ceiling is a grey reference rather
# than a result of its own.
STABILITY_AXIS_COLORS = {
    "seed_only": "0.55",
    "jackknife": _PINK,
}
STABILITY_AXIS_LABELS = {
    "seed_only": "Seed only",
    "jackknife": "Jackknife",
}

CMAPS = {
    "diverging": sns.diverging_palette(240, 10, as_cmap=True),  # signed loadings
    # Containment and |cosine| in the comparator-benchmark figures.
    "sequential": sns.color_palette("ch:s=-.2,r=.6", as_cmap=True),
    # Magnitudes and relative weights: light until substantial, so only heavy
    # items read dark.
    "weight": LinearSegmentedColormap.from_list(
        "weight",
        [
            (0, "#eef2f7"),
            (0.45, "#cde2fb"),
            (0.7, "#86b6ef"),
            (0.85, "#2a78d6"),
            (1, "#0d366b"),
        ],
    ),
}

# Fixed per dataset, so a cell type keeps its color wherever it is plotted.
CELL_TYPES = {
    "alad": [
        "Macrophages",
        "Monocytes",
        "Dendritic Cells",
        "CD4 T cells",
        "CD8 T cells",
        "NK cells",
        "Epithelial cells",
        # Last, so this cell state rather than type takes tab10's grey.
        "Proliferating cells",
    ],
    "ibd": [
        "B cell",
        "Plasma B cell",
        "T cell",
        "Myeloid",
        "Enterocyte",
        "Secretory",
        "Stem cell",
        "Mesenchymal",
    ],
}
_CELL_TYPE_PALETTE = [to_hex(c) for c in sns.color_palette("tab10")]
CELL_TYPE_COLORS = {
    dataset: dict(zip(types, _CELL_TYPE_PALETTE, strict=False))
    for dataset, types in CELL_TYPES.items()
}


def apply_style() -> None:
    """Apply the shared rcParams. Called once per figure build."""
    sns.reset_orig()
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            f"font.{FONT_FAMILY}": FONT_NAMES,
            "font.size": FONT_SIZE_LABEL,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "axes.linewidth": LINE_WIDTH_THIN,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
            "figure.titlesize": FONT_SIZE_PANEL_LABEL,
            "figure.titleweight": "bold",
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "xtick.major.width": LINE_WIDTH_THIN,
            "ytick.major.width": LINE_WIDTH_THIN,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "patch.linewidth": LINE_WIDTH_THIN,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.title_fontsize": FONT_SIZE_LEGEND,
            "legend.frameon": LEGEND_FRAMEON,
            "legend.borderpad": LEGEND_BORDERPAD,
            "legend.labelspacing": LEGEND_LABELSPACING,
            "legend.handlelength": LEGEND_HANDLELENGTH,
            "legend.handletextpad": LEGEND_HANDLETEXTPAD,
            "legend.borderaxespad": LEGEND_BORDERAXESPAD,
            "legend.markerscale": LEGEND_MARKERSCALE,
            "savefig.dpi": 300,
        }
    )
