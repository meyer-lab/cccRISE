"""Helpers shared by the rcheck scripts."""

from cccrise.comparators.inputs import Inputs, load_inputs
from cccrise.runners.datasets import DATASETS


def dataset_inputs(dataset: str, adata=None) -> Inputs:
    """The registry dataset `dataset` (``alad``/``ibd``) as the comparators
    see it, against its production cccRISE fit; `adata` replaces the
    registry's loaded data (e.g. S4's depth-thinned counts, same genes)."""
    spec = DATASETS[dataset]
    return load_inputs(
        spec.factors_path(),
        spec.load() if adata is None else adata,
        spec.condition_column,
        spec.celltype_column,
    )
