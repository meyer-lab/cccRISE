"""Python re-implementation of NICHES (msraredon/NICHES, R; pinned at
``5d932b8``).

NICHES (Raredon et al., Bioinformatics 2023) has no model: it writes the
per-cell ligand x receptor products out as a new matrix whose columns are
cell pairs or cells, and then analyses that matrix like expression data.
Ported modes:

- `cell_to_cell` / `cell_to_cell_pairs` (``RunCellToCell``): for every ordered
  pair of cell types (autocrine included), ``min(n_a, n_b)`` sender and
  receiver cells drawn without replacement and paired by position; the
  column is ``L(sender) * R(receiver)``. This is NICHES' only random step.
- `system_to_cell` (``RunSystemToCell``): every cell as a receiver, against
  the whole system's ligand (``blend``: mean (default), sum, or mean over
  expressing cells, ``mean.adj``), i.e. ``op_i L(i) * R(j)``.
- `cell_to_system` (``RunCellToSystem``): the mirror, ``L(i) * op_j R(j)``.
- `filter_ground_truth` / `filter_cells_per_type` (``FilterGroundTruth``,
  ``PrepNodeObject``'s ``min.cells.per.ident``).

Outputs are the raw products, as upstream returns them: under Seurat v5 its
``data`` layer is overwritten with ``counts``, so no normalization is applied.
NICHES runs per condition, as upstream recommends: cells are paired with, and
the "system" spans, cells of the same sample only.

Differences from upstream, all deliberate (see `cccrise.comparators.inputs`):

- the expression is cccRISE's normalized ``adata.X`` (upstream:
  log-normalized or ALRA-imputed ``data``); there is no imputation, since
  cccRISE does none;
- the LR table is the cccRISE fit's (upstream: FANTOM5);
- complexes use cccRISE's ``min`` aggregation (upstream multiplies subunits,
  available as ``agg="product"`` in `inputs.lr_expression` and used by the R
  checks);
- random draws are NumPy's; `cell_to_cell` takes the pairs as an argument so
  the R checks can score R's own draws;
- ``min.cells.per.gene`` (counted on raw counts) is not applied, since the
  gene universe is cccRISE's.

Comparing against cccRISE: NICHES enters as a model-free per-cell reference.
A cccRISE program's per-cell signal is compared with the NICHES
SystemToCell / CellToSystem scores over that program's LR pairs (weighted by
its LR loadings), within cell type. Per-cell-pair (edge) concordance is not
used: the NICHES and Scriabin edges both rank cell pairs by L * R, so they are
one reference rather than two, and cccRISE's per-edge reconstruction does not
recover per-cell expression (`notes/ANALYSIS_RECORD.md` §4.5, E16/E17).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

BLENDS = ("mean", "sum", "mean.adj")


def _blend(values: np.ndarray, blend: str) -> np.ndarray:
    """Combine a (cells x LR) matrix over cells: mean, sum, or ``mean.adj``
    (mean divided by the fraction of expressing cells, so NaN for an entity no
    cell expresses, as upstream)."""
    if blend == "mean":
        return values.mean(axis=0)
    if blend == "sum":
        return values.sum(axis=0)
    if blend == "mean.adj":
        with np.errstate(invalid="ignore", divide="ignore"):
            return values.mean(axis=0) / (values > 0).mean(axis=0)
    raise ValueError(f"blend must be one of {BLENDS}, not {blend!r}")


def system_to_cell(lig: np.ndarray, rec: np.ndarray, blend: str = "mean") -> np.ndarray:
    """``RunSystemToCell``: (cells x LR), the system's blended ligand times each
    cell's receptor."""
    return _blend(lig, blend)[None, :] * rec


def cell_to_system(lig: np.ndarray, rec: np.ndarray, blend: str = "mean") -> np.ndarray:
    """``RunCellToSystem``: (cells x LR), each cell's ligand times the system's
    blended receptor."""
    return lig * _blend(rec, blend)[None, :]


def cell_to_cell_pairs(
    cell_types: Sequence[str] | np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """``RunCellToCell``'s sampling: for each sender type, then each receiver
    type (sorted, as ``table(Idents)`` orders them), ``min(n_send, n_recv)``
    sender cells and, independently, as many receiver cells, each drawn
    without replacement and paired by position. Returns one row per edge:
    ``sender``, ``receiver`` (cell indices), ``sending_type``,
    ``receiving_type``."""
    cell_types = np.asarray(cell_types)
    types = np.unique(cell_types)
    pools = {t: np.nonzero(cell_types == t)[0] for t in types}
    blocks = []
    for s in types:
        n = {r: min(len(pools[s]), len(pools[r])) for r in types}
        senders = [rng.choice(pools[s], size=n[r], replace=False) for r in types]
        receivers = [rng.choice(pools[r], size=n[r], replace=False) for r in types]
        for r, snd, rcv in zip(types, senders, receivers, strict=True):
            blocks.append(
                pd.DataFrame(
                    {
                        "sender": snd,
                        "receiver": rcv,
                        "sending_type": s,
                        "receiving_type": r,
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True)


def cell_to_cell(lig: np.ndarray, rec: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    """(edges x LR) CellToCell matrix for the sampled `pairs`:
    ``L(sender) * R(receiver)``."""
    return lig[pairs["sender"].to_numpy()] * rec[pairs["receiver"].to_numpy()]


def filter_ground_truth(
    lr_pairs: pd.DataFrame, gene_names: Sequence[str], complex_sep: str = "&"
) -> pd.DataFrame:
    """``FilterGroundTruth``: keep mechanisms whose every ligand and receptor
    subunit is a gene of the dataset."""
    genes = set(gene_names)

    def present(entity: str) -> bool:
        return all(g in genes for g in entity.split(complex_sep))

    keep = lr_pairs["ligand"].map(present) & lr_pairs["receptor"].map(present)
    return lr_pairs[keep.to_numpy()].reset_index(drop=True)


def filter_cells_per_type(
    cell_types: Sequence[str] | np.ndarray, min_cells_per_ident: int | None
) -> np.ndarray:
    """``PrepNodeObject``'s ``min.cells.per.ident``: mask of cells whose type
    has strictly more than that many cells (all cells when None)."""
    cell_types = np.asarray(cell_types)
    if min_cells_per_ident is None:
        return np.ones(len(cell_types), dtype=bool)
    types, counts = np.unique(cell_types, return_counts=True)
    return np.isin(cell_types, types[counts > min_cells_per_ident])
