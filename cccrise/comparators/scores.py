"""How cccRISE's components and the comparators' compare (Figure 5).

A fit's components are read as `Components`: per mode, a matrix with one
column per component, rows labelled. cccRISE and the comparators share the
sample and LR modes outright; cccRISE's sender and receiver modes index RISE
eigen-states, so they are put on the comparators' cell types as each type's
mean |per-cell loading| (`celltype_profiles`). A Tucker (STACCato) fit has no
per-component sender, receiver or sample pattern, so its sender and receiver
modes are left out and its sample and LR columns are scored on their own.

- `containment`: the share of a unit column's squared norm in another fit's
  column span, ``||P_S c||^2``, called contained at `CONTAINED` or above when
  it is also above the 95th percentile of a row-permutation null.
- `effective_number`: the inverse Simpson index of a column's squared
  loadings, ``(sum x^2)^2 / sum x^4``; k equal entries give k. For samples it
  is the participation ratio.
- `inner_losses`: what Tensor-cell2cell's published construction (only the
  cell types present in every sample) drops from its masked tensor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .methods import MODES

#: Containment at or above this, and above the null, calls a column contained.
CONTAINED = 0.85


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


def containment(
    span: pd.DataFrame, columns: pd.DataFrame, n_null: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Per column of `columns`: its containment in `span`'s column span (rows
    matched by label), the 95th percentile of that with `span`'s rows
    permuted, and whether it is contained."""
    span = span.loc[columns.index].to_numpy(float)
    values = columns.to_numpy(float)
    observed = _projected(span, values)
    null = np.array(
        [_projected(span[rng.permutation(len(span))], values) for _ in range(n_null)]
    )
    q95 = np.quantile(null, 0.95, axis=0)
    return pd.DataFrame(
        {
            "Component": columns.columns,
            "Containment": observed,
            "NullQ95": q95,
            "Contained": (observed >= CONTAINED) & (observed > q95),
        }
    )


def group_r2(values: np.ndarray, groups: np.ndarray) -> float:
    """The share of `values`' variance its `groups`' means hold."""
    means = pd.Series(values).groupby(groups).transform("mean").to_numpy()
    return float(
        1 - np.sum((values - means) ** 2) / np.sum((values - values.mean()) ** 2)
    )


def paint_r2(values: np.ndarray, painted: np.ndarray) -> float:
    """The squared correlation of `values` with `painted`: 0 for a constant
    `painted`, which holds none of their variance."""
    return float(np.corrcoef(values, painted)[0, 1] ** 2) if painted.std() > 0 else 0.0


def effective_number(columns: pd.DataFrame) -> pd.Series:
    """Per column, the inverse Simpson index of its squared loadings."""
    squared = columns.to_numpy(float) ** 2
    return pd.Series(
        squared.sum(axis=0) ** 2 / (squared**2).sum(axis=0), index=columns.columns
    )


def top_share(columns: pd.DataFrame) -> pd.Series:
    """Per column, its largest entry's share of the squared loading."""
    squared = columns.to_numpy(float) ** 2
    return pd.Series(squared.max(axis=0) / squared.sum(axis=0), index=columns.columns)


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
    factors: xr.Dataset, cells, celltype_column: str, celltypes
) -> Components:
    """cccRISE's CP components, its sender and receiver on `celltypes` from
    `cells`' per-cell loadings (``obsm["sc_B"]``, ``obsm["rc_C"]``)."""
    names = factors["cp_component"].to_numpy()
    labels = cells.obs[celltype_column].astype(str)
    modes = {
        "sample": pd.DataFrame(
            factors["cp_A"].to_numpy(), factors["condition"].to_numpy()
        ),
        "sender": celltype_profiles(cells.obsm["sc_B"], labels, celltypes),
        "receiver": celltype_profiles(cells.obsm["rc_C"], labels, celltypes),
        "lr": pd.DataFrame(factors["cp_D"].to_numpy(), factors["lr_pair"].to_numpy()),
    }
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
