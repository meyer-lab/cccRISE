from __future__ import annotations

import numpy as np
import pandas as pd

# Cytosolic ribosomal proteins that `gene_group__ribo_protein` does not mark.
MISSED_CYTOSOLIC_RIBOSOMAL = ("RPL13", "RPL18", "RPSA")

# Nuclear-encoded MT-RNR2 pseudogenes.
MTRNR2L_PATTERN = r"^MTRNR2L"

# Mitochondrial ribosomal proteins, which `gene_group__ribo_protein` lumps in
# with the cytosolic ribosome. `DAP3` is MRPS29 and belongs with them.
MITO_RIBOSOMAL_PATTERN = r"^MRP[SL]"
MITO_RIBOSOMAL_EXTRA = ("DAP3",)

# `var` columns the Sanger pipeline ships.
MITO_FLAG_COLUMNS = ("gene_group__mito_transcript", "gene_group__mito_protein")
RIBO_FLAG_COLUMNS = ("gene_group__ribo_protein", "gene_group__ribo_rna")


def _symbols(var: pd.DataFrame, symbol_column: str) -> pd.Series:
    """Gene symbols, falling back to the index when the column is absent."""
    if symbol_column in var:
        return var[symbol_column].astype(str)
    return pd.Series(var.index.astype(str), index=var.index)


def _bools(series: pd.Series) -> np.ndarray:
    """A writable bool array from a pandas boolean result."""
    return np.asarray(series, dtype=bool).copy()


def _flag(var: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """OR together whichever of `columns` the table actually has."""
    out = np.zeros(len(var), dtype=bool)
    for column in columns:
        if column in var:
            out |= _bools(var[column])
    return out


def mito_ribo_mask(
    var: pd.DataFrame,
    *,
    symbol_column: str = "gene_symbols",
    exclude_mito_ribosomal: bool = True,
    exclude_mtrnr2l: bool = True,
) -> np.ndarray:
    symbols = _symbols(var, symbol_column)

    # `.to_numpy()` on a pandas boolean result can be read-only; copy before
    # accumulating into it.
    is_mito_ribosomal = _bools(symbols.str.match(MITO_RIBOSOMAL_PATTERN, na=False))
    is_mito_ribosomal |= _bools(symbols.isin(MITO_RIBOSOMAL_EXTRA))

    exclude = _flag(var, MITO_FLAG_COLUMNS)

    # The ribo flags cover both ribosomes. The cytosolic half always goes; the
    # mitochondrial half is separable, since it is a different kind of gene.
    ribo = _flag(var, RIBO_FLAG_COLUMNS)
    exclude |= ribo & ~is_mito_ribosomal
    exclude |= _bools(symbols.isin(MISSED_CYTOSOLIC_RIBOSOMAL))

    if exclude_mito_ribosomal:
        exclude |= ribo & is_mito_ribosomal
    if exclude_mtrnr2l:
        exclude |= _bools(symbols.str.match(MTRNR2L_PATTERN, na=False))

    return exclude
