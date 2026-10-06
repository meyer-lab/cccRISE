"""How general ("basal") or specific a communication program is.

Each statistic describes one mode of one component, so that CP fits (one
component across every mode) and Tucker fits (unpaired columns per mode) are
described the same way:

- **LR mode -- prevalence**: the |loading|^2-weighted mean, over the
  component's LR pairs, of each pair's *prevalence*: the fraction of
  (sample, sender type, receiver type) blocks in which the ligand is detected
  in at least `MIN_DETECTED` of the sender type's cells and the receptor in
  at least `MIN_DETECTED` of the receiver type's cells (complex subunits: the
  least-detected one). High prevalence = an interaction present almost
  everywhere.
- **Condition mode -- breadth and variability**: the participation ratio of
  the loading (the effective number of samples carrying it) and the
  coefficient of variation of its absolute value.
- **Sender / receiver modes -- cell-type breadth**: the inverse Simpson index
  of the squared loadings over cell types (the effective number of types).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

MIN_DETECTED = 0.1


def detection_table(
    X, genes: Sequence[str], samples: Sequence[str], types: Sequence[str]
) -> pd.DataFrame:
    """Fraction of cells with nonzero counts per (sample, type) row and gene
    column; `X` is cells x `genes`."""
    keys = pd.MultiIndex.from_arrays(
        [np.asarray(samples).astype(str), np.asarray(types).astype(str)],
        names=["sample", "type"],
    )
    codes, groups = pd.factorize(keys)
    indicator = sparse.csr_array(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    detected = sparse.csr_array(X) > 0
    counts = np.asarray((indicator @ detected.astype(float)).todense())
    sizes = np.bincount(codes)[:, None]
    index = pd.MultiIndex.from_tuples(list(groups), names=keys.names)
    return pd.DataFrame(counts / sizes, index=index, columns=list(genes))


def _complex_detection(det: pd.DataFrame, names: pd.Series, sep: str) -> np.ndarray:
    """(sample-type rows x pairs) detection of each complex: its least-detected
    subunit (NaN when a subunit is missing from the data)."""
    out = np.full((len(det), len(names)), np.nan)
    for j, name in enumerate(names):
        subunits = [g for g in str(name).split(sep) if g]
        if all(g in det.columns for g in subunits):
            out[:, j] = det[subunits].to_numpy().min(axis=1)
    return out


def lr_prevalence(
    det: pd.DataFrame,
    ligands: pd.Series,
    receptors: pd.Series,
    sep: str = "&",
    min_detected: float = MIN_DETECTED,
) -> np.ndarray:
    """Per LR pair: the fraction of (sample, sender type, receiver type)
    blocks, over the types present in each sample, where both are detected."""
    lig = _complex_detection(det, ligands, sep) >= min_detected
    rec = _complex_detection(det, receptors, sep) >= min_detected
    samples = det.index.get_level_values("sample")
    hits = np.zeros(len(ligands))
    blocks = 0
    for s in np.unique(samples):
        rows = samples == s
        hits += lig[rows].sum(axis=0) * rec[rows].sum(axis=0)
        blocks += int(rows.sum()) ** 2
    return hits / blocks


def weighted_prevalence(loadings: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """Per LR column: the |loading|^2-weighted mean prevalence."""
    w = np.asarray(loadings, float) ** 2
    return (w * prevalence[:, None]).sum(axis=0) / w.sum(axis=0)


def participation_ratio(loadings: np.ndarray) -> np.ndarray:
    """Per column: (sum a^2)^2 / sum a^4, the effective number of rows."""
    sq = np.asarray(loadings, float) ** 2
    return sq.sum(axis=0) ** 2 / (sq**2).sum(axis=0)


def abs_cv(loadings: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(loadings, float))
    return a.std(axis=0) / a.mean(axis=0)


def top_pathways(
    loadings: np.ndarray, pathways: Sequence[str], n_top: int = 5
) -> list[str]:
    """Per LR column: the distinct pathways of its `n_top` largest |loadings|."""
    pathways = np.asarray(pathways).astype(str)
    order = np.argsort(-np.abs(loadings), axis=0)[:n_top]
    return [
        ", ".join(dict.fromkeys(pathways[order[:, k]])) for k in range(order.shape[1])
    ]


def mode_statistics(
    modes: dict[str, np.ndarray],
    prevalence: np.ndarray,
    pathways: Sequence[str],
) -> pd.DataFrame:
    """Long table (mode, component, statistic, value) for one fit's modes:
    condition, lr, sender, receiver (LR rows in `prevalence`'s order)."""
    frames = []
    stats = {
        "condition": {
            "sample_breadth": participation_ratio,
            "sample_abs_cv": abs_cv,
        },
        "lr": {"lr_prevalence": lambda m: weighted_prevalence(m, prevalence)},
        "sender": {"type_breadth": participation_ratio},
        "receiver": {"type_breadth": participation_ratio},
    }
    for mode, functions in stats.items():
        for name, function in functions.items():
            values = function(modes[mode])
            frames.append(
                pd.DataFrame(
                    {
                        "mode": mode,
                        "component": np.arange(1, len(values) + 1),
                        "statistic": name,
                        "value": values,
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True).assign(top_pathways="")
    lr_rows = table["mode"] == "lr"
    paths = top_pathways(modes["lr"], pathways)
    table.loc[lr_rows, "top_pathways"] = [
        paths[k - 1] for k in table.loc[lr_rows, "component"]
    ]
    return table
