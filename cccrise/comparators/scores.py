"""How cccRISE's components and the comparators' compare (Figure 5).

A fit's components are read as `Components`: per mode, a matrix with one
column per component, rows labelled. cccRISE and the comparators share the
sample and LR modes outright; cccRISE's sender and receiver modes index RISE
eigen-states, so they are put on the comparators' cell types as each type's
mean |per-cell loading| (`celltype_profiles`). A Tucker (STACCato) fit has no
per-component sender, receiver or sample pattern, so its sender and receiver
modes are left out and its sample and LR columns are scored on their own.

- `containment`: the share of a unit column's squared norm in another fit's
  column span, ``||P_S c||^2`` (the fraction of a cccRISE component's LR pair
  loadings a comparator fit reconstructs, Figure 5B).
- `inner_losses`: what Tensor-cell2cell's published construction (only the
  cell types present in every sample) drops from its masked tensor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .methods import MODES


@dataclass(frozen=True)
class Components:
    """One fit's components: per mode (of `MODES`) present, a frame with one
    row per label and one column per component."""

    method: str
    modes: dict[str, pd.DataFrame]


def inner_losses(masked, kept) -> pd.DataFrame:
    """The share of `masked` (a masked `PooledTensor` of every cell type)
    that keeping only the cell types `kept` drops: its cell types, its
    observed entries, and its signal (sum of squares over observed entries)
    ('Measure', 'Dropped', 'Total', 'Share')."""
    types = pd.Index(masked.celltypes)
    gone = ~types.isin(list(kept))
    involved = gone[None, :, None, None] | gone[None, None, :, None]
    observed = masked.mask.astype(bool)
    dropped = observed & involved
    signal = masked.values**2 * observed
    rows = [
        ("cell types", gone.sum(), len(types)),
        ("observed entries", dropped.sum(), observed.sum()),
        ("signal", (signal * involved).sum(), signal.sum()),
    ]
    return pd.DataFrame(
        [{"Measure": m, "Dropped": d, "Total": t, "Share": d / t} for m, d, t in rows]
    )


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    """Columns scaled to unit L2 norm (zero columns left at zero)."""
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def _projected(span: np.ndarray, columns: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(span)
    return np.sum((basis.T @ unit_columns(columns)) ** 2, axis=0)


def containment(span: pd.DataFrame, columns: pd.DataFrame) -> pd.DataFrame:
    """Per column of `columns`: its containment in `span`'s column span (rows
    matched by label)."""
    span = span.loc[columns.index].to_numpy(float)
    return pd.DataFrame(
        {
            "Component": columns.columns,
            "Containment": _projected(span, columns.to_numpy(float)),
        }
    )


def oriented(columns: pd.DataFrame) -> pd.DataFrame:
    """Columns signed so their largest-magnitude entry is positive."""
    values = columns.to_numpy(float)
    signs = np.sign(values[np.abs(values).argmax(axis=0), np.arange(values.shape[1])])
    return columns * np.where(signs == 0, 1, signs)


def celltype_profiles(
    loadings: np.ndarray, celltypes: pd.Series, order
) -> pd.DataFrame:
    """Each type of `order`'s mean |per-cell loading|, one column per
    component: a cccRISE sender or receiver mode on the comparators' cell
    types."""
    means = pd.DataFrame(np.abs(loadings)).groupby(celltypes.to_numpy()).mean()
    return means.loc[list(order)].set_axis(list(order))


def cccrise_components(
    factors: xr.Dataset, cells=None, celltype_column: str = "", celltypes=()
) -> Components:
    """cccRISE's CP components, its sender and receiver on `celltypes` from
    `cells`' per-cell loadings (``obsm["sc_B"]``, ``obsm["rc_C"]``); without
    `cells`, only its sample and LR modes, which need no dataset."""
    names = factors["cp_component"].to_numpy()
    modes = {
        "sample": pd.DataFrame(
            factors["cp_A"].to_numpy(), factors["condition"].to_numpy()
        ),
        "lr": pd.DataFrame(factors["cp_D"].to_numpy(), factors["lr_pair"].to_numpy()),
    }
    if cells is not None:
        labels = cells.obs[celltype_column].astype(str)
        modes["sender"] = celltype_profiles(cells.obsm["sc_B"], labels, celltypes)
        modes["receiver"] = celltype_profiles(cells.obsm["rc_C"], labels, celltypes)
    return Components(
        "cccRISE", {m: frame.set_axis(names, axis=1) for m, frame in modes.items()}
    )


def comparator_components(method: str, fit: xr.Dataset) -> Components:
    """A comparator fit's (`methods.fit_dataset`'s) components; a Tucker fit
    keeps only its sample and LR modes, each with its own columns."""
    tucker = "core" in fit
    modes = ("sample", "lr") if tucker else MODES
    return Components(
        method,
        {
            mode: pd.DataFrame(
                fit[f"{mode}_factor"].to_numpy(),
                index=fit[mode].to_numpy().astype(str),
                columns=np.arange(1, fit[f"{mode}_factor"].shape[1] + 1),
            )
            for mode in modes
        },
    )


def tucker_column_profiles(fit: xr.Dataset, column: int) -> dict[str, np.ndarray]:
    """A Tucker LR column's sample, sender and receiver profiles: the
    leading singular vector of each unfolding of its core slice, mapped
    through that mode's factor (`column` counts from 1)."""
    core = fit["core"].isel(lr_component=column - 1).to_numpy()
    profiles = {}
    for axis, mode in enumerate(("sample", "sender", "receiver")):
        unfolded = np.moveaxis(core, axis, 0).reshape(core.shape[axis], -1)
        leading = np.linalg.svd(unfolded, full_matrices=False)[0][:, 0]
        profiles[mode] = fit[f"{mode}_factor"].to_numpy() @ leading
    return profiles
