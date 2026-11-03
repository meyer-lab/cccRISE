"""Helpers shared by the rcheck scripts."""

from cccrise.comparators.inputs import Inputs, load_inputs
from cccrise.runners.datasets import DATASETS


def dataset_inputs(dataset: str) -> Inputs:
    """The registry dataset `dataset` (``alad``/``ibd``) as the comparators
    see it, against its production cccRISE fit."""
    spec = DATASETS[dataset]
    return load_inputs(
        spec.factors_path(), spec.load(), spec.condition_column, spec.celltype_column
    )
