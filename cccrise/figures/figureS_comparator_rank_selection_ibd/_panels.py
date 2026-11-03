"""IBD's comparator rank-selection panels: Figure 4's (ALAD) panels, drawn on
IBD."""

from ..figure4.figureS_comparator_rank_selection._panels import (
    pb_collinearity,
    pb_fit,
    pb_minima,
    stac_shares,
    tc2c_congruence,
    tc2c_elbow,
    tc2c_minima,
)

DATASET = "ibd"

__all__ = [
    "DATASET",
    "pb_collinearity",
    "pb_fit",
    "pb_minima",
    "stac_shares",
    "tc2c_congruence",
    "tc2c_elbow",
    "tc2c_minima",
]
