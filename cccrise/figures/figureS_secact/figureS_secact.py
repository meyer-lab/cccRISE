"""Supplementary figure: NicheNet v2 vs SecAct ligand-target priors in the
downstream-target validation, under this project's and SecAct's pipelines
(`uv run cccrise-prior-comparison`)."""

from .. import common, style
from . import agreement, confound, coverage, top_ligand_tests

PANELS = {
    "coverage": coverage,
    "tests": top_ligand_tests,
    "agreement": agreement,
    "confound": confound,
}
LAYOUT = [
    ["coverage", "confound", "confound"],
    ["tests", "tests", "tests"],
    ["agreement", "agreement", "agreement"],
]


def makeFigure():
    return common.compose(
        style.FIGSIZE["full_page"], LAYOUT, PANELS, height_ratios=[1, 1, 1.3]
    )
