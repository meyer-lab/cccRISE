"""Supplementary figure: NicheNet v2 vs SecAct ligand-target priors in the
downstream-target validation, under this project's and SecAct's pipelines
(`uv run cccrise-prior-comparison`)."""

from .. import common, style
from . import agreement, confound, coverage, ligand_boxes, top_ligand_tests

PANELS = {
    "coverage": coverage,
    "nichenet_boxes": ligand_boxes.NICHENET,
    "secact_boxes": ligand_boxes.SECACT,
    "tests": top_ligand_tests,
    "agreement": agreement,
    "confound": confound,
}
LAYOUT = [
    ["coverage"] * 2 + ["confound"] * 4,
    ["nichenet_boxes"] * 3 + ["secact_boxes"] * 3,
    ["tests"] * 6,
    ["agreement"] * 6,
]


def makeFigure():
    return common.compose(
        style.FIGSIZE["full_page"], LAYOUT, PANELS, height_ratios=[1, 1, 1, 1.3]
    )
