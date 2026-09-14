"""Mitochondrial and ribosomal gene exclusion.

The IBDverse terminal-ileum cohort spends 27.9% of its counts on 177 genes --
0.53% of the gene set -- and that fraction is confounded with both disease
status (Healthy 27.9% vs CD-inflamed 22.5%) and cell type (Stem cell 37.7% vs
Plasma B cell 14.5%). None of those genes is a ligand or receptor, so removing
them costs no communication signal; leaving them in puts a technical gradient
into the per-cell normalization denominator, and so into every other gene.
`IBD_GENE_FILTER_AUDIT.md` has the full measurement.

The source file ships four `gene_group__*` boolean columns from the Sanger
pipeline. They are the right starting point -- a symbol regex is not, because
`^RP[SL]` also matches the ribosomal protein S6 *kinases* (`RPS6KA1`-`RPS6KA6`,
`RPS6KB1`/`RPS6KB2`, ...), which are signaling genes -- but they need two
corrections, both measured in the audit:

* they miss `RPL13`, `RPL18` and `RPSA`, three genuine cytosolic ribosomal
  proteins carrying 0.43% of counts between them (not duplicate-symbol
  artifacts; each has one unique Ensembl ID), and
* `gene_group__ribo_protein` conflates the cytosolic ribosome with the
  mitochondrial one, which is 79 ordinary nuclear-encoded housekeeping genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Cytosolic ribosomal proteins that `gene_group__ribo_protein` does not mark.
MISSED_CYTOSOLIC_RIBOSOMAL = ("RPL13", "RPL18", "RPSA")

#: Nuclear-encoded MT-RNR2 pseudogenes. A recognized ambient-RNA artifact that
#: no `MT-` rule catches, since they are not mitochondrially encoded. 0.24% of
#: IBDverse counts, nearly all of it `MTRNR2L12` and `MTRNR2L8`.
MTRNR2L_PATTERN = r"^MTRNR2L"

#: Mitochondrial ribosomal proteins, which `gene_group__ribo_protein` lumps in
#: with the cytosolic ribosome. `DAP3` is MRPS29 and belongs with them.
MITO_RIBOSOMAL_PATTERN = r"^MRP[SL]"
MITO_RIBOSOMAL_EXTRA = ("DAP3",)

#: `var` columns the Sanger pipeline ships. `gene_group__mito_protein` selects
#: exactly the same 13 genes as `gene_group__mito_transcript`, and
#: `gene_group__ribo_rna` is empty on this reference, but both are read so the
#: mask stays correct on a cohort where they differ.
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
    exclude_mito_ribosomal: bool = False,
    exclude_mtrnr2l: bool = True,
) -> np.ndarray:
    """Boolean mask over `var` rows selecting genes to *exclude*.

    Returns all-False when the table carries none of the `gene_group__*`
    columns and no symbol matches, so a cohort without these annotations passes
    through untouched rather than being silently half-filtered.

    Parameters
    ----------
    var
        A `.var` table; `symbol_column` supplies symbols, otherwise the index.
    exclude_mito_ribosomal
        Also drop `MRPS*`/`MRPL*`/`DAP3`. Off by default: they are
        nuclear-encoded housekeeping genes rather than a technical artifact,
        and they carry 0.28% of IBDverse counts, so the choice is close to free
        either way. See the audit's section 4.
    exclude_mtrnr2l
        Drop the `MTRNR2L*` pseudogenes. On by default.
    """
    symbols = _symbols(var, symbol_column)

    # `.to_numpy()` on a pandas boolean result can be read-only; copy before
    # accumulating into it.
    is_mito_ribosomal = _bools(symbols.str.match(MITO_RIBOSOMAL_PATTERN, na=False))
    is_mito_ribosomal |= _bools(symbols.isin(MITO_RIBOSOMAL_EXTRA))

    exclude = _flag(var, MITO_FLAG_COLUMNS)

    # The ribo flags cover both ribosomes; keep the mitochondrial one unless
    # asked otherwise.
    ribo = _flag(var, RIBO_FLAG_COLUMNS)
    exclude |= ribo & ~is_mito_ribosomal
    exclude |= _bools(symbols.isin(MISSED_CYTOSOLIC_RIBOSOMAL))

    if exclude_mito_ribosomal:
        exclude |= ribo & is_mito_ribosomal
    if exclude_mtrnr2l:
        exclude |= _bools(symbols.str.match(MTRNR2L_PATTERN, na=False))

    return exclude


def describe_mask(
    var: pd.DataFrame, mask: np.ndarray, symbol_column: str = "gene_symbols"
) -> str:
    """One-line summary of what a mask removes, for runner logs."""
    symbols = _symbols(var, symbol_column)
    n = int(mask.sum())
    if n == 0:
        return "gene filter: nothing excluded"
    shown = sorted(symbols[mask])[:6]
    return f"gene filter: excluding {n} of {len(var)} genes ({', '.join(shown)}, ...)"
