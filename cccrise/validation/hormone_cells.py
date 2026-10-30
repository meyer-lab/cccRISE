"""Which cells carry cccRISE's programs, against the fine cell type labels
(`HORMONE_CELLS_PREREGISTRATION.md`, frozen 2026-10-20, for IBD; the ALAD
addendum `HORMONE_CELLS_PREREGISTRATION_ALAD.md`, frozen 2026-10-21).

A pooled tensor method gives every cell of a label one value; cccRISE gives
each cell its own sender (`sc_B = projections @ B`) and receiver
(`rc_C = projections @ C`) loading. Two per-cell quantities are compared
throughout:

- the **loading** itself, as the model outputs it;
- the **contribution**: the loading times the cell's centred expression of
  the program's ligands (sender) or receptors (receiver), weighted by the
  LR mode. Before the tensor's clip at zero, a sample's eigenstate-level
  expression is ``P_s^T (X_s - mu)``, a sum over cells, so the contributions
  of a sample's cells add up exactly to that sample's program-level ligand
  (receptor) signal (checked by `run_hormone_cells cells`).

Every analysis reads the cell table `run_hormone_cells cells` writes, so the
cohort is loaded once.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

COMPLEX_SEP = "&"
PAIR_SEP = "^"
SIDES = ("sender", "receiver")

Carriers = dict[int, tuple[tuple[str, ...], tuple[str, ...]]]


@dataclass(frozen=True)
class DatasetConfig:
    """What the analysis fixes per dataset before it runs.

    `carriers` holds the a-priori carrier labels per CP component
    (1-indexed): (sender labels, receiver labels). Marker sets are the
    starting lists of the pre-registration (§4D); a literature review extends
    them. `qc` maps the cell table's QC fields to the dataset's obs columns
    (None: absent). Cells of `ambient_reference` lineages give each sample's
    ambient level of a program gene."""

    carriers: Carriers
    program_names: dict[int, str]
    carrier_markers: dict[str, list[str]]
    lineage_markers: dict[str, list[str]]
    qc: dict[str, str | None]
    group_column: str
    ambient_reference: dict[str, set[str]]
    tensor_file: str
    preregistration: str
    #: Candidate labellings for the fitted-comparator paint, in order (§4B)
    paint_labellings: dict[int, list[str]] = field(default_factory=dict)

    def carrier_labels(self, k: int, side: str) -> tuple[str, ...]:
        return self.carriers[k][SIDES.index(side)]

    def all_carriers(self) -> list[str]:
        return sorted({c for pair in self.carriers.values() for s in pair for c in s})

    def reference_lineages(self, carrier_lineages: set[str]) -> set[str]:
        """Ambient-reference lineages for a program whose carriers sit in
        `carrier_lineages`: the 'epithelial' reference when every carrier is
        epithelial, else the 'immune' one."""
        key = (
            "epithelial"
            if carrier_lineages <= self.ambient_reference["epithelial_lineages"]
            else "other"
        )
        return self.ambient_reference[key]


IBD = DatasetConfig(
    carriers={
        1: (("Monocytes SOD2+",), ("Monocytes SOD2+",)),
        2: (("ILCs", "Mast cells"), ("ILCs", "Mast cells")),
        3: (("Enteroendocrine cells",), ("Enterochromaffin cells",)),
        4: (("Enteroendocrine cells",), ("Tuft cells",)),
        5: (("Monocytes SOD2+",), ("Monocytes SOD2+",)),
        6: (("ILCs",), ("ILCs",)),
    },
    program_names={
        1: "C1 IL-1",
        2: "C2 MAG",
        3: "C3 PYY–NPY1R",
        4: "C4 GCG–GLP1R",
        5: "C5 CCL3L1/CXCL8",
        6: "C6 CDH4",
    },
    carrier_markers={
        "Enteroendocrine cells": ["CHGA", "CHGB", "PYY", "GCG", "INSL5", "NTS", "SCG5"],
        "Enterochromaffin cells": ["TPH1", "CHGA", "CES1", "ADGRG4"],
        "Tuft cells": ["POU2F3", "TRPM5", "SH2D6", "LRMP", "AVIL"],
        "Monocytes SOD2+": ["SOD2", "IL1B", "S100A8", "FCN1"],
        "ILCs": ["KIT", "IL7R", "KLRB1", "RORC"],
        "Mast cells": ["TPSAB1", "CPA3", "KIT"],
    },
    lineage_markers={
        "B cell": ["MS4A1", "CD79A", "CD19"],
        "Plasma B cell": ["JCHAIN", "MZB1", "IGHA1"],
        "T cell": ["CD3E", "CD3D", "CD2"],
        "Myeloid": ["LYZ", "CD68", "AIF1"],
        "Enterocyte": ["APOA4", "FABP2", "RBP2", "ALPI"],
        "Secretory": ["MUC2", "TFF3", "SPINK4"],
        "Stem cell": ["OLFM4", "LGR5", "SMOC2"],
    },
    qc={
        "conf_score": "conf_score",
        "n_genes": "n_genes_by_counts",
        "total_counts": "total_counts",
        "doublet_score": None,
    },
    group_column="inflammation_status",
    ambient_reference={
        "epithelial_lineages": {"Secretory", "Enterocyte", "Stem cell"},
        "epithelial": {"B cell", "T cell", "Plasma B cell"},
        "other": {"Enterocyte"},
    },
    tensor_file="ibd_tensor_rise60.npy",
    preregistration="HORMONE_CELLS_PREREGISTRATION.md (frozen r2, 2026-10-20)",
    paint_labellings={
        1: ["_fine", "_lin+mono"],
        2: ["_fine", "_lin+ilc", "_lin+mast"],
        3: ["_fine", "_lin+ee+ec", "_lin+ee+ec+tuft"],
        4: ["_fine", "_lin+ee+tuft", "_lin+ee+ec+tuft"],
        5: ["_fine", "_lin+mono"],
        6: ["_fine", "_lin+ilc"],
    },
)

ALAD = DatasetConfig(
    # The fine type with the largest active fraction per side (ANALYSIS_REPORT
    # §2.4; `lr_concordance_cell_activity_alad`, fine annotation)
    carriers={
        1: (("Lung epithelium",), ("Lung epithelium",)),
        2: (("NK cells",), ("cDC1",)),
        3: (("Activated DCs",), ("NK cells",)),
        4: (("NK cells",), ("Activated CD8",)),
        5: (("Lung epithelium",), ("Lung epithelium",)),
    },
    program_names={
        1: "C1 CEACAM1–CEACAM5",
        2: "C2 XCL1/2–XCR1",
        3: "C3 CCL17/22–CCR4",
        4: "C4 GZMA/CCL5 (lymphocyte)",
        5: "C5 BTC–EGFR",
    },
    carrier_markers={
        "Lung epithelium": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
        "NK cells": ["KLRF1", "NCAM1", "GNLY", "NKG7"],
        "cDC1": ["CLEC9A", "XCR1", "BATF3", "IDO1"],
        "Activated DCs": ["CCR7", "LAMP3", "CCL19", "FSCN1"],
        "Activated CD8": ["CD8A", "CD8B", "GZMB", "PRF1"],
    },
    # Figure 2's literature markers of the eight broad types (Travaglini 2020)
    lineage_markers={
        "Macrophages": ["MARCO", "FABP4", "C1QB", "CD68"],
        "Monocytes": ["CD14", "FCN1", "VCAN", "S100A8"],
        "Dendritic Cells": ["CD1C", "FCER1A", "CLEC10A"],
        "CD4 T cells": ["CD4", "IL7R", "CD40LG"],
        "CD8 T cells": ["CD8A", "CD8B", "GZMK"],
        "NK cells": ["KLRF1", "NCAM1", "GNLY"],
        "Epithelial cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
        "Proliferating cells": ["MKI67", "TOP2A"],
    },
    qc={
        "conf_score": None,
        "n_genes": "nFeature_RNA",
        "total_counts": "nCount_RNA",
        "doublet_score": "doubletFinderScore",
    },
    group_column="ALADstatus",
    ambient_reference={
        "epithelial_lineages": {"Epithelial cells"},
        "epithelial": {"Macrophages"},
        "other": {"Macrophages"},
    },
    tensor_file="alad_tensor_rise17.npy",
    preregistration="HORMONE_CELLS_PREREGISTRATION_ALAD.md (frozen, 2026-10-21)",
    paint_labellings={k: ["_fine", "_coarse"] for k in range(1, 6)},
)

CONFIGS = {"ibd": IBD, "alad": ALAD}


def pair_tokens(pair: str) -> tuple[str, str]:
    """(ligand, receptor) of an ``"L^R"`` pair name; a complex stays one
    token (``"IL1R1&IL1RAP"``)."""
    ligand, receptor = pair.split(PAIR_SEP)
    return ligand, receptor


def token_genes(token: str) -> list[str]:
    return token.split(COMPLEX_SEP)


def side_token_weights(
    D: np.ndarray,
    pairs: Sequence[str],
    component: int,
    side: str,
    rho: float = 0.0,
    power: int = 2,
) -> pd.Series:
    """{ligand (sender) or receptor (receiver) token: weight} of a component:
    pair weights ``|D|**power`` over the pairs with ``|D| >= rho * max|D|``,
    normalised to sum to 1 and summed per token."""
    d = np.abs(np.asarray(D)[:, component - 1])
    keep = d >= rho * d.max()
    weights = d[keep] ** power
    tokens = [pair_tokens(p)[SIDES.index(side)] for p in np.asarray(pairs)[keep]]
    return pd.Series(weights / weights.sum(), index=tokens).groupby(level=0).sum()


def token_matrix(
    expression: sparse.csr_matrix, genes: Sequence[str], tokens: Sequence[str]
) -> np.ndarray:
    """(cells, tokens) raw expression of each token: a gene's expression, or
    the min over a complex's subunits (as the tensor's complex aggregation),
    computed per cell. Tokens with a missing subunit are NaN columns."""
    index = {g: i for i, g in enumerate(genes)}
    out = np.full((expression.shape[0], len(tokens)), np.nan, dtype=np.float32)
    for j, token in enumerate(tokens):
        cols = [index.get(g) for g in token_genes(token)]
        if any(c is None for c in cols):
            continue
        block = expression[:, cols].toarray()
        out[:, j] = block.min(axis=1)
    return out


def centred_signal(
    expression: sparse.spmatrix,
    genes: Sequence[str],
    gene_means: pd.Series,
    weights: pd.Series,
    centre: bool = True,
) -> np.ndarray:
    """Per cell, ``sum_token w (x_token - mu_token)``: single genes centred on
    the fit's gene means (`var["means"]`), complexes (min over subunits per
    cell) on their own cell mean. Tokens with a gene outside the table are
    dropped and the weights renormalised. `centre=False` gives the expressed
    part ``sum_token w x_token`` alone."""
    index = {g: i for i, g in enumerate(genes)}
    present = [t for t in weights.index if set(token_genes(t)) <= index.keys()]
    w = weights[present] / weights[present].sum()
    is_complex = w.index.str.contains(COMPLEX_SEP, regex=False)
    out = _singles_signal(expression, index, gene_means, w[~is_complex], centre)
    return out + _complexes_signal(expression, genes, w[is_complex], centre)


def _complexes_signal(expression, genes, w: pd.Series, centre: bool):
    """The complex part of `centred_signal`: min over subunits per cell,
    centred on its own cell mean."""
    if w.empty:
        return 0.0
    raw = token_matrix(expression, genes, list(w.index))
    centred = raw - raw.mean(axis=0) if centre else raw
    return centred @ w.to_numpy()


def _singles_signal(expression, index, gene_means, w: pd.Series, centre: bool):
    """The single-gene part of `centred_signal`, as one sparse product."""
    vec = np.zeros(len(index))
    vec[[index[t] for t in w.index]] = w.to_numpy()
    out = np.asarray(expression @ vec).ravel()
    if centre:
        out -= float((w * gene_means[w.index]).sum())
    return out


def sample_sizes(samples: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(samples).astype(str))
    return s.map(s.value_counts()).to_numpy(float)


# --- A. Eigenstate composition ----------------------------------------------


def eigen_mixture(projections: np.ndarray) -> np.ndarray:
    """Per cell, its share of squared projection on each eigenstate (rows sum
    to 1): what the cell is made of."""
    p2 = np.asarray(projections, float) ** 2
    return p2 / p2.sum(axis=1, keepdims=True)


def type_means(values: np.ndarray, labels: np.ndarray, order: Sequence[str]):
    """Mean of each column of `values` over the cells of each label."""
    return (
        pd.DataFrame(np.asarray(values))
        .groupby(np.asarray(labels))
        .mean()
        .reindex(order)
    )


def sample_equal_shares(
    weights: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
    order: Sequence[str],
    within: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per column of `weights` (cells x k, non-negative), each label's share
    of the column's total within a sample, averaged over samples with equal
    weight (a label absent from a sample counts 0). With `within` (a coarser
    label per cell), the share is of the label's `within` group's total
    instead, averaged over the samples where that group is present."""
    labels = np.asarray(labels)
    sums = (
        pd.DataFrame(np.asarray(weights, float))
        .groupby([np.asarray(samples), labels])
        .sum()
    )
    sample_of = sums.index.get_level_values(0)
    if within is None:
        parent_of = np.zeros(len(sums), dtype=int)
    else:
        parent = pd.Series(np.asarray(within), index=labels).groupby(level=0).first()
        parent_of = parent.reindex(sums.index.get_level_values(1)).to_numpy()
    totals = sums.groupby([sample_of, parent_of]).transform("sum")
    shares = (sums / totals.where(totals > 0)).fillna(0.0)
    n_present = pd.Series(sample_of).groupby(parent_of).nunique()
    per_label = shares.groupby(level=1).sum()
    label_parent = (
        np.zeros(len(per_label), dtype=int)
        if within is None
        else parent.reindex(per_label.index).to_numpy()
    )
    return (
        per_label.div(n_present.reindex(label_parent).to_numpy(), axis=0)
        .reindex(order)
        .fillna(0.0)
    )


def pooled_shares(weights: np.ndarray, labels: np.ndarray, order, within=None):
    """As `sample_equal_shares`, pooling cells over samples (cell-count
    weighted) instead."""
    frame = pd.DataFrame(np.asarray(weights, float)).groupby(np.asarray(labels)).sum()
    if within is None:
        return (frame / frame.sum()).reindex(order).fillna(0.0)
    parent = (
        pd.Series(np.asarray(within), index=np.asarray(labels)).groupby(level=0).first()
    )
    totals = frame.groupby(parent.reindex(frame.index).to_numpy()).transform("sum")
    return (frame / totals).reindex(order).fillna(0.0)


# --- C. Weight outside the carrier labels --------------------------------------


def part_sums(
    values: np.ndarray, samples: np.ndarray, groups: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Per sample, the net and positive-part sums of `values` over all cells
    and over each group mask: columns ``<group>_net`` and ``<group>_pos``."""
    v = np.asarray(values, float)
    cols = {"all_net": v, "all_pos": np.clip(v, 0, None)}
    for name, mask in groups.items():
        cols[f"{name}_net"] = np.where(mask, v, 0.0)
        cols[f"{name}_pos"] = np.where(mask, np.clip(v, 0, None), 0.0)
    return pd.DataFrame(cols).groupby(np.asarray(samples)).sum()


def shares_with_ci(
    sums: pd.DataFrame, group: str, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """A group's net and positive-part share of the total, pooled over
    samples, with a patient-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    out = {}
    for kind in ("net", "pos"):
        num, den = sums[f"{group}_{kind}"].to_numpy(), sums[f"all_{kind}"].to_numpy()
        boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
        out[kind] = num.sum() / den.sum()
        out[f"{kind}_lo"], out[f"{kind}_hi"] = np.percentile(boot, [2.5, 97.5])
    return out


def random_directions(n_dims: int, n: int, seed: int = 0) -> np.ndarray:
    d = np.random.default_rng(seed).standard_normal((n_dims, n))
    return d / np.linalg.norm(d, axis=0)


def direction_null(
    projections: np.ndarray,
    observed: np.ndarray,
    signal: np.ndarray | None,
    onehot: sparse.csr_matrix,
    directions: np.ndarray,
    chunk: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-direction null for one program side. Scores are a unit
    direction's per-cell value (times `signal` for contributions); the
    observed column and the random directions are all unit vectors, so they
    compare without rescaling (rescaling by the SD over cells would penalise
    a direction that concentrates on few cells). Returns (observed per-type
    mean score, null per-type means (types x directions), per-cell count of
    directions scoring at least the observed)."""
    sig = 1.0 if signal is None else signal[:, None]

    def zscores(dirs):
        return (projections @ dirs) * sig

    sizes = np.asarray(onehot.sum(axis=1)).ravel()
    z_obs = zscores((observed / np.linalg.norm(observed))[:, None])[:, 0]
    nulls, exceed = [], np.zeros(len(z_obs), dtype=np.int32)
    for start in range(0, directions.shape[1], chunk):
        z = zscores(directions[:, start : start + chunk]).astype(np.float32)
        nulls.append(np.asarray(onehot @ z) / sizes[:, None])
        exceed += (z >= z_obs[:, None]).sum(axis=1).astype(np.int32)
    return np.asarray(onehot @ z_obs) / sizes, np.hstack(nulls), exceed


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


# --- F. Representability -------------------------------------------------------


def representability(values: np.ndarray, *groupings: np.ndarray) -> float:
    """Share of the variance of `values` explained by the means of the groups
    the given label arrays jointly define (R^2 of a cell-mean model)."""
    v = pd.Series(np.asarray(values, float))
    means = v.groupby(list(groupings)).transform("mean")
    return float(1 - ((v - means) ** 2).sum() / ((v - v.mean()) ** 2).sum())


def shuffled_within(
    labels: np.ndarray, *blocks: np.ndarray, seed: int = 0
) -> np.ndarray:
    """`labels` permuted among the cells of each block (same counts per
    block): a size-matched random split."""
    rng = np.random.default_rng(seed)
    out = np.asarray(labels).copy()
    key = pd.Series(range(len(out))).groupby(list(blocks)).indices
    for idx in key.values():
        out[idx] = out[rng.permutation(idx)]
    return out


# --- Support levels (pre-registration §5) ------------------------------------------


def h1_level(
    n_types_beyond: int,
    n_cells_beyond: int,
    share: float,
    lo: float,
    cuts=(0.10, 0.05, 0.20, 0.10),
) -> int:
    """H1 level: 0 none, 1 present, 2 material, 3 substantial."""
    material, material_lo, substantial, substantial_lo = cuts
    if n_types_beyond == 0 and n_cells_beyond == 0:
        return 0
    if share >= substantial and lo >= substantial_lo:
        return 3
    if share >= material and lo >= material_lo:
        return 2
    return 1


def h2_level(
    energy_out: float,
    frac_top_out: float,
    any_top_out: bool,
    cuts=(0.01, 0.05, 0.10, 0.10, 0.25),
) -> int:
    present, material_e, substantial_e, material_f, substantial_f = cuts
    if energy_out >= substantial_e and frac_top_out >= substantial_f:
        return 3
    if energy_out >= material_e and frac_top_out >= material_f:
        return 2
    if any_top_out or energy_out >= present:
        return 1
    return 0
