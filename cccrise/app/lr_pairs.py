"""Helpers for parsing CCC-RISE ligand-receptor pair labels.

Pairs are stored as strings of the form ``LIGAND^REC1&REC2&...``: a single
ligand gene, a ``^`` separator, then one or more receptor-complex subunit
genes joined by ``&``. Ligand complexes are not encoded in this string (they
are handled upstream), so the ligand side is always a single gene.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LRPair:
    label: str
    ligand: str
    receptor_subunits: tuple[str, ...]

    @property
    def is_simple_receptor(self) -> bool:
        """True if the receptor side is a single gene (no complex)."""
        return len(self.receptor_subunits) == 1


def parse_lr_pair(label: str) -> LRPair:
    ligand, _, receptor_part = label.partition("^")
    receptor_subunits = tuple(receptor_part.split("&")) if receptor_part else ()
    return LRPair(label=label, ligand=ligand, receptor_subunits=receptor_subunits)


def parse_lr_pairs(labels: np.ndarray) -> list[LRPair]:
    return [parse_lr_pair(label) for label in labels]
