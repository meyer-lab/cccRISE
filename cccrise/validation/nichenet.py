"""
NicheNet-style downstream-target validation of cccRISE / CPD components.

For each CPD component, this module tests whether the literature-curated
downstream targets of the component's top-loading ligands are differentially
expressed in the receiver cells the component implicates, reporting a
per-LR-pair/per-component AUROC of the receiver DE statistic for recovering
the ligand's top predicted targets (a NicheNet-inspired variant of its
ligand-activity score, which instead treats the DE genes as the positive
class and the ligand's regulatory potential as the score) and a two-sided
empirical permutation p-value (distance of the AUROC from 0.5) against a
null built from the same receiver population.

The ligand-target prior is the NicheNet v2 ligand-target matrix (Zenodo
record 7074291), which is only published as an R `.rds` object; it is read
here with `pyreadr` (a pure-Python RDS parser) rather than adding an R/rpy2
dependency. `fetch_ligand_target_matrix` downloads the full matrix once,
restricts it to the ligand vocabulary actually used by this pipeline's LR
resource, and caches a small long-format parquet (mirroring how
`Human-2020-Jin-LR-pairs.parquet` is cached in `import_data.py`) so that
neither the network nor `pyreadr` is required afterwards.
"""

import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data"
NICHENET_LT_URL = (
    "https://zenodo.org/records/7074291/files/ligand_target_matrix_nsga2r_final.rds"
)
NICHENET_LT_CACHE = DATA_DIR / "nichenet_ligand_target_matrix.parquet"

ReceiverMode = Literal["within_celltype", "vs_rest"]
DeMethod = Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"]


def fetch_ligand_target_matrix(
    ligands: list[str] | None = None,
    top_n_targets: int = 250,
    cache_path: str | Path = NICHENET_LT_CACHE,
    url: str = NICHENET_LT_URL,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download the NicheNet v2 ligand-target matrix and cache a filtered,
    long-format subset of it.

    The raw matrix (`ligand_target_matrix_nsga2r_final.rds`) is ~260MB and
    dense (every ligand x every target gene), which is far more than this
    pipeline needs and too large to commit to the repository. This function
    downloads it once, keeps only the requested `ligands` (typically the
    ligand vocabulary of `import_ligand_receptor_pairs()`), keeps only the
    top `top_n_targets` targets per ligand by regulatory potential, and
    caches the result as a small long-format parquet
    (columns: ligand, target, weight).

    Parameters
    ----------
    ligands : list[str], optional
        Ligand gene symbols to keep. If None, keeps every ligand in the
        matrix (much larger cache).
    top_n_targets : int, default=250
        Number of top-scoring targets to retain per ligand, matching the
        default of NicheNet's `get_weighted_ligand_target_links`
        (active-target inference).
    cache_path : str or Path, default=NICHENET_LT_CACHE
        Where to write/read the cached long-format parquet.
    url : str, default=NICHENET_LT_URL
        Source URL for the raw `.rds` ligand-target matrix.
    force : bool, default=False
        Re-download and rebuild the cache even if `cache_path` exists.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ["ligand", "target", "weight"].
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    import tempfile

    import pyreadr

    with tempfile.TemporaryDirectory() as tmpdir:
        rds_path = Path(tmpdir) / "ligand_target_matrix.rds"
        urllib.request.urlretrieve(url, rds_path)
        result = pyreadr.read_r(rds_path)

    matrix = next(iter(result.values()))
    # The R matrix has targets as rownames and ligands as column names;
    # pyreadr surfaces the rownames as the DataFrame index.
    if ligands is not None:
        available = [lig for lig in ligands if lig in matrix.columns]
        matrix = matrix[available]

    records = []
    for ligand in matrix.columns:
        col = matrix[ligand].astype(np.float32)
        top = col.nlargest(top_n_targets)
        records.append(
            pd.DataFrame(
                {
                    "ligand": ligand,
                    "target": top.index,
                    "weight": top.to_numpy(),
                }
            )
        )
    long_df = pd.concat(records, ignore_index=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(cache_path, index=False)

    return long_df


def load_ligand_target_matrix(
    cache_path: str | Path = NICHENET_LT_CACHE,
) -> pd.DataFrame:
    """Load the cached long-format ligand-target matrix produced by `fetch_ligand_target_matrix`."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist. Run `fetch_ligand_target_matrix` once "
            "(requires network access and `pyreadr`) to build the cache."
        )
    return pd.read_parquet(cache_path)


def top_lr_pairs_for_component(
    D: np.ndarray,
    lr_pairs: pd.DataFrame,
    cmp: int,
    top_m: int = 10,
) -> pd.DataFrame:
    """
    Return the top `top_m` ligand-receptor pairs loading onto CPD component `cmp`.

    Ranks by absolute loading magnitude rather than signed loading.

    Parameters
    ----------
    D : np.ndarray
        LR-pair CPD factor matrix, shape (n_lr_pairs, cp_rank) (`adata.uns["D"]`).
    lr_pairs : pd.DataFrame
        LR pairs corresponding to the rows of `D`, with "ligand"/"receptor" columns
        (the `filtered_lr_pairs` returned alongside `D` from the cccRISE workflow).
    cmp : int
        1-indexed component number.
    top_m : int, default=10
        Number of top LR pairs to return.

    Returns
    -------
    pd.DataFrame
        Columns ["ligand", "receptor", "loading"], sorted by descending
        `abs(loading)`, of length `min(top_m, n_lr_pairs)`.
    """
    loadings = np.asarray(D)[:, cmp - 1]
    order = np.argsort(-np.abs(loadings))[:top_m]

    out = lr_pairs.iloc[order][["ligand", "receptor"]].reset_index(drop=True)
    out["loading"] = loadings[order]
    return out


def receiver_celltypes_for_component(
    adata: anndata.AnnData,
    cmp: int,
    celltype_col: str = "broad_cell_type",
    pool_threshold: float = 0.3,
) -> list[str]:
    """
    Return the receiver cell type(s) implicated by CPD component `cmp`.

    Cell types are ranked by mean receiver-eigenstate loading
    (`adata.obsm["rc_C"][:, cmp - 1]`). The top-ranked cell type is always
    included; any other cell type whose mean loading is at least
    `pool_threshold` of the top cell type's mean loading is pooled in too.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with `obsm["rc_C"]` populated.
    cmp : int
        1-indexed component number.
    celltype_col : str, default="broad_cell_type"
    pool_threshold : float, default=0.3
        Minimum mean loading, as a fraction of the top cell type's mean
        loading, for another cell type to be pooled in.

    Returns
    -------
    list[str]
        Cell types, descending by mean loading; always length >= 1.
    """
    loadings = np.asarray(adata.obsm["rc_C"])[:, cmp - 1]
    means = (
        pd.Series(loadings, index=adata.obs.index)
        .groupby(adata.obs[celltype_col])
        .mean()
        .sort_values(ascending=False)
    )
    top_mean = means.iloc[0]
    if top_mean <= 0:
        # No cell type has positive mean loading for this component -- fall
        # back to just the (least-negative) top one rather than pooling.
        return [cast(str, means.index[0])]
    pooled = means[means >= pool_threshold * top_mean]
    return cast(list[str], pooled.index.tolist())


def _receiver_population(
    adata: anndata.AnnData,
    cmp: int,
    celltype_col: str,
    celltype: str | list[str] | None,
    top_perc: float,
) -> tuple[anndata.AnnData, np.ndarray, int, int]:
    """
    Restrict `adata` to `celltype` (if given) and split it into the cells
    above the `100 - top_perc` percentile ("high") and below the `top_perc`
    percentile ("low") of component `cmp`'s receiver-eigenstate loading
    (`adata.obsm["rc_C"][:, cmp - 1]`), discarding every other cell.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with `obsm["rc_C"]` populated.
    cmp : int
        1-indexed component number.
    celltype_col : str
        Column in `adata.obs` naming cell types.
    celltype : str, list[str], or None
        If given, `adata` is restricted to this cell type (or these cell
        types) before splitting into groups. If None, all of `adata` is used.
    top_perc : float
        Percentile threshold defining the "high" and "low" groups.

    Returns
    -------
    tuple[anndata.AnnData, np.ndarray, int, int]
        (kept, pos_mask, n_pos, n_neg): `kept` is the cells in the "high" or
        "low" group (union), `pos_mask` is a boolean array aligned to `kept`
        marking the "high" group, and `n_pos`/`n_neg` are the two group sizes.
    """
    if celltype is None:
        X = adata
    else:
        celltypes = [celltype] if isinstance(celltype, str) else list(celltype)
        X = adata[adata.obs[celltype_col].isin(celltypes)]

    loadings = np.asarray(X.obsm["rc_C"])[:, cmp - 1]
    hi_thr = np.percentile(loadings, 100 - top_perc)
    lo_thr = np.percentile(loadings, top_perc)
    pos_idx = loadings > hi_thr
    neg_idx = loadings < lo_thr

    kept_idx = pos_idx | neg_idx
    kept = X[kept_idx].copy()
    pos_mask = pos_idx[kept_idx]
    return kept, pos_mask, int(pos_mask.sum()), int((~pos_mask).sum())


def _run_de(
    kept: anndata.AnnData,
    pos_mask: np.ndarray,
    cmp: int,
    method: DeMethod,
) -> pd.Series:
    """Run `scanpy.tl.rank_genes_groups` for `pos_mask` vs. the rest of `kept`."""
    kept.obs["Label"] = np.where(pos_mask, f"Cmp{cmp}", "NoLabel")
    sc.tl.rank_genes_groups(kept, "Label", groups=[f"Cmp{cmp}"], method=method)
    results = kept.uns["rank_genes_groups"]
    names = results["names"][f"Cmp{cmp}"]
    scores = results["scores"][f"Cmp{cmp}"]
    return pd.Series(scores, index=names).sort_values(ascending=False)


def receiver_de_ranking(
    adata: anndata.AnnData,
    cmp: int,
    receiver_mode: ReceiverMode = "within_celltype",
    celltype_col: str = "broad_cell_type",
    celltype: str | list[str] | None = None,
    top_perc: float = 10,
    method: DeMethod = "wilcoxon",
) -> pd.Series:
    """
    Rank genes by a differential-expression statistic between the "high" and
    "low" component-loading groups defined by `_receiver_population`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with `obsm["rc_C"]` populated (i.e. already processed by the
        cccRISE workflow / `load_factors_to_anndata`).
    cmp : int
        1-indexed component number.
    receiver_mode : {"within_celltype", "vs_rest"}, default="within_celltype"
        "within_celltype" restricts to `celltype` before grouping;
        "vs_rest" uses all of `adata`.
    celltype_col : str, default="broad_cell_type"
        Column in `adata.obs` naming cell types. Used only when
        `receiver_mode == "within_celltype"`.
    celltype : str or list[str], optional
        Required when `receiver_mode == "within_celltype"`.
    top_perc : float, default=10
        Percentile threshold defining the "high"/"low" groups; see
        `_receiver_population`.
    method : str, default="wilcoxon"
        `scanpy.tl.rank_genes_groups` method.

    Returns
    -------
    pd.Series
        Gene -> DE test statistic, sorted descending (most "high-group-up" first).
    """
    if receiver_mode == "within_celltype":
        if celltype is None:
            raise ValueError(
                "`celltype` is required when receiver_mode='within_celltype'"
            )
    elif receiver_mode == "vs_rest":
        celltype = None
    else:
        raise ValueError(f"Unknown receiver_mode: {receiver_mode!r}")

    kept, pos_mask, n_pos, n_neg = _receiver_population(
        adata, cmp, celltype_col, celltype, top_perc
    )
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Too few cells to test (n_pos={n_pos}, n_neg={n_neg}) for "
            f"cmp={cmp}, receiver_mode={receiver_mode!r}, celltype={celltype!r}."
        )
    return _run_de(kept, pos_mask, cmp, method)


def ligand_activity_auroc(
    de_ranking: pd.Series,
    ligand: str,
    lt_df: pd.DataFrame,
) -> float | None:
    """
    Compute the NicheNet-style ligand-activity AUROC for one ligand.

    Genes in `de_ranking` that are among `ligand`'s cached top targets
    (`lt_df`) are treated as the positive class; every other gene in
    `de_ranking` is the negative/background class. `de_ranking`'s DE test
    statistic is used as the classifier score.

    Parameters
    ----------
    de_ranking : pd.Series
        Gene -> DE test statistic (from `receiver_de_ranking`), covering the
        receiver population's expressed/tested genes.
    ligand : str
        Ligand gene symbol.
    lt_df : pd.DataFrame
        Long-format ligand-target matrix (columns "ligand", "target", "weight"),
        from `load_ligand_target_matrix`/`fetch_ligand_target_matrix`.

    Returns
    -------
    float or None
        AUROC, or None if `ligand` has no cached targets overlapping
        `de_ranking`'s genes, or if the overlap is entirely positive or
        entirely negative (AUROC undefined).
    """
    targets = set(lt_df.loc[lt_df["ligand"] == ligand, "target"])
    if not targets:
        return None

    genes = de_ranking.index
    y_true = genes.isin(targets).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    return float(roc_auc_score(y_true, de_ranking.to_numpy()))


def permutation_null_aurocs(
    kept: anndata.AnnData,
    pos_mask: np.ndarray,
    lt_df: pd.DataFrame,
    ligands: list[str],
    n_perm: int = 1000,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """
    Compute a null distribution of ligand-activity AUROCs for `ligands`, by
    repeatedly reassigning which cells in `kept` are in the positive group.

    Each of `n_perm` iterations draws `pos_mask.sum()` cells (without
    replacement, from all cells in `kept`) as the positive group and treats
    the remaining cells as the negative group. Per-gene ranks across cells
    are computed once and reused for every iteration; each iteration's
    per-gene score is the sum of positive-group ranks for that gene, and
    each ligand's AUROC is computed from that score against its target set,
    as in `ligand_activity_auroc`.

    Parameters
    ----------
    kept : anndata.AnnData
        Cell population to draw the positive/negative relabeling from.
    pos_mask : np.ndarray[bool]
        Aligned to `kept`; only `pos_mask.sum()` (the positive group size)
        is used.
    lt_df : pd.DataFrame
        Long-format ligand-target matrix, as in `ligand_activity_auroc`.
    ligands : list[str]
        Ligand gene symbols to compute null distributions for. Duplicates
        are computed once and reused.
    n_perm : int, default=1000
        Number of relabelings.
    random_state : int, optional
        Seed for the relabeling draws.

    Returns
    -------
    dict[str, np.ndarray]
        ligand -> array of `n_perm` null AUROCs. Ligands with no targets
        overlapping `kept.var_names`, or whose overlap is entirely positive
        or negative, are omitted (matching `ligand_activity_auroc`).
    """
    rng = np.random.default_rng(random_state)
    genes = kept.var_names
    X = cast(Any, kept.X)
    dense = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    ranks = rankdata(dense, axis=0).astype(np.float32)

    n_cells = kept.n_obs
    n_pos = int(pos_mask.sum())
    perm_pos = np.zeros((n_perm, n_cells), dtype=np.float32)
    for i in range(n_perm):
        idx = rng.choice(n_cells, size=n_pos, replace=False)
        perm_pos[i, idx] = 1.0
    null_scores = perm_pos @ ranks  # (n_perm, n_genes)
    score_ranks = rankdata(null_scores, axis=1)  # (n_perm, n_genes)

    out: dict[str, np.ndarray] = {}
    for ligand in dict.fromkeys(ligands):  # dedupe, preserve order
        targets = set(lt_df.loc[lt_df["ligand"] == ligand, "target"])
        y_true = genes.isin(targets)
        n_target = int(y_true.sum())
        n_bg = len(y_true) - n_target
        if not targets or n_target == 0 or n_bg == 0:
            continue
        r_pos = score_ranks[:, y_true].sum(axis=1)
        out[ligand] = (r_pos - n_target * (n_target + 1) / 2) / (n_target * n_bg)
    return out


def component_accuracy_scores(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    lt_df: pd.DataFrame,
    components: list[int],
    top_m_lr: int = 10,
    receiver_mode: ReceiverMode = "within_celltype",
    celltype_col: str = "broad_cell_type",
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    pool_threshold: float = 0.3,
    n_perm: int = 1000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Compute per-LR-pair NicheNet-style AUROCs, with a permutation null, for
    each of `components`.

    For each component: take its top `top_m_lr` LR pairs
    (`top_lr_pairs_for_component`); build one receiver DE ranking
    (`_receiver_population` + `_run_de`) -- restricted, when `receiver_mode
    == "within_celltype"`, to the cell type(s) implicated by that component
    (`receiver_celltypes_for_component`) -- then score every top ligand's
    downstream targets (`ligand_activity_auroc`) against that single
    ranking, and a two-sided empirical p-value (`permutation_null_aurocs`)
    against `n_perm` random relabelings of the same receiver population. The
    null always scores genes by a Wilcoxon rank sum, so `p_value` is
    calibrated only for `de_method="wilcoxon"`. A ligand paired with several
    top receptors repeats its AUROC/p-value on each of its rows.

    Returns
    -------
    pd.DataFrame
        Columns: ["component", "ligand", "receptor", "loading", "celltype",
        "auroc", "n_pos", "n_neg", "null_mean_auroc", "null_sd_auroc",
        "p_value"]. `auroc`/`p_value` are NaN where they could not be
        computed (see `ligand_activity_auroc`); `celltype` is a single name,
        or several joined by " + " when `receiver_mode == "within_celltype"`
        pools more than one cell type, and NaN under `vs_rest`. Components
        with fewer than 2 cells in either group are omitted.
    """
    D = adata.uns["D"]
    rows = []

    for cmp in components:
        top_pairs = top_lr_pairs_for_component(D, lr_pairs, cmp, top_m=top_m_lr)

        celltype = (
            receiver_celltypes_for_component(
                adata, cmp, celltype_col=celltype_col, pool_threshold=pool_threshold
            )
            if receiver_mode == "within_celltype"
            else None
        )

        try:
            kept, pos_mask, n_pos, n_neg = _receiver_population(
                adata, cmp, celltype_col, celltype, top_perc
            )
            if n_pos < 2 or n_neg < 2:
                raise ValueError(
                    f"Too few cells to test (n_pos={n_pos}, n_neg={n_neg}) for "
                    f"cmp={cmp}, receiver_mode={receiver_mode!r}, celltype={celltype!r}."
                )
        except ValueError:
            continue

        de_ranking = _run_de(kept, pos_mask, cmp, de_method)
        ligands = top_pairs["ligand"].tolist()
        null_aurocs = permutation_null_aurocs(
            kept,
            pos_mask,
            lt_df,
            ligands,
            n_perm=n_perm,
            random_state=None if random_state is None else random_state + cmp,
        )
        celltype_label = (
            " + ".join(celltype) if isinstance(celltype, list) else celltype
        )

        for _, pair in top_pairs.iterrows():
            ligand = cast(str, pair["ligand"])
            receptor = cast(str, pair["receptor"])
            auroc = ligand_activity_auroc(de_ranking, ligand, lt_df)
            null = null_aurocs.get(ligand)
            if auroc is not None and null is not None:
                null_mean = float(np.mean(null))
                null_sd = float(np.std(null))
                p_value = float(np.mean(np.abs(null - 0.5) >= abs(auroc - 0.5)))
            else:
                null_mean = null_sd = p_value = None
            rows.append(
                {
                    "component": cmp,
                    "ligand": ligand,
                    "receptor": receptor,
                    "loading": pair["loading"],
                    "celltype": celltype_label,
                    "auroc": auroc,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "null_mean_auroc": null_mean,
                    "null_sd_auroc": null_sd,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)
