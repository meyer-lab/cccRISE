"""Supplementary figure: Tensor-cell2cell as published against Tensor-cell2cell
as Figure 4 runs it, on ALAD. The paper keeps only the cell types present in
every sample (``how="inner"``, no mask); Figure 4 keeps every type, masking
the samples a type is absent from. What the published construction drops
from the tensor (A), how much of each cccRISE program lives in the dropped
types (B, C), and which cccRISE programs each version contains (D, E)."""

from ... import common, style
from . import (
    losses,
    program_losses_coarse,
    program_losses_fine,
    reach_coarse,
    reach_fine,
)

LAYOUT = [
    [losses, program_losses_coarse, program_losses_fine],
    [reach_coarse, reach_fine],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
