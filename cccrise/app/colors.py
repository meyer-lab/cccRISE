"""Shared color scheme for the explorer app.

Categorical colors come from seaborn's Set2 (the ANOVA box plots' palette),
extended with Set3's similar pastels when more categories are needed. Cell
types take the manuscript figures' fixed colors (`figures.style`), as do
continuous colormaps (heatmaps, weighted PaCMAPs).
"""

import seaborn as sns
from matplotlib.colors import to_hex

TEAL, ORANGE, BLUE, PINK, GREEN, YELLOW, TAN, _ = (
    to_hex(c) for c in sns.color_palette("Set2")
)
LIGHT_GREY = "#dddddd"

SENDER = ORANGE
RECEIVER = BLUE
HIGHLIGHT = TEAL
MUTED = LIGHT_GREY

# Set2 without its grey, which is reserved for "none"/"neither" categories.
_CATEGORICAL = [TEAL, ORANGE, BLUE, PINK, GREEN, YELLOW, TAN] + [
    to_hex(c) for c in sns.color_palette("Set3")
]


def categorical_colors(n: int) -> list[str]:
    """`n` categorical colors, repeating only past the Set2 + Set3 pool."""
    return [_CATEGORICAL[i % len(_CATEGORICAL)] for i in range(n)]
