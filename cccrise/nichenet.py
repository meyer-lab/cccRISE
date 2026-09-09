"""
NicheNet-style downstream-target validation of cccRISE / CPD components.

For each CPD component, this module tests whether the literature-curated
downstream targets of the component's top-loading ligands are actually
differentially expressed in the receiver cells the component implicates,
reporting a per-ligand/per-component AUROC (the standard NicheNet
ligand-activity statistic). This is a real-data accuracy metric, in place of
a synthetic ground truth (see GitHub issue #136).

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
from typing import Literal, cast

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent / "data"
NICHENET_LT_URL = (
    "https://zenodo.org/records/7074291/files/ligand_target_matrix_nsga2r_final.rds"
)
NICHENET_LT_CACHE = DATA_DIR / "nichenet_ligand_target_matrix.parquet"

ReceiverMode = Literal["within_celltype", "vs_rest"]


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
        Number of top-scoring targets to retain per ligand, matching
        NicheNet's own convention for its ligand-activity analysis.
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
    TODO: revisit once the communication-score sign ambiguity (#139) is
    resolved -- at that point it may be correct to only take
    positive-signed loadings.

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


def receiver_de_ranking(
    adata: anndata.AnnData,
    cmp: int,
    receiver_mode: ReceiverMode = "within_celltype",
    celltype_col: str = "broad_cell_type",
    celltype: str | None = None,
    top_perc: float = 10,
    method: Literal[
        "logreg", "t-test", "wilcoxon", "t-test_overestim_var"
    ] = "wilcoxon",
) -> pd.Series:
    """
    Rank genes by a differential-expression statistic between receiver cells
    with high vs. low loading on CPD component `cmp`.

    Two configurations for defining the comparison groups (see #136 plan):

    - "within_celltype": restrict to a single `celltype` first, then split
      that population into its top/bottom `top_perc` by receiver-eigenstate
      loading (`adata.obsm["rc_C"][:, cmp - 1]`) and compare those two
      groups to each other. Controls for cell-type identity, matching the
      existing `figureS11a.py` analysis.
    - "vs_rest": take the top `top_perc` of *all* cells (not restricted to
      one cell type) by receiver-eigenstate loading and compare them to
      every other cell. Larger effective sample size / effect size, but
      conflates component activity with cell-type composition.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with `obsm["rc_C"]` populated (i.e. already processed by the
        cccRISE workflow / `load_factors_to_anndata`).
    cmp : int
        1-indexed component number.
    receiver_mode : {"within_celltype", "vs_rest"}, default="within_celltype"
    celltype_col : str, default="broad_cell_type"
        Required (and used) only when `receiver_mode == "within_celltype"`.
    celltype : str, optional
        Required when `receiver_mode == "within_celltype"`.
    top_perc : float, default=10
        Percentile split for defining "high" vs. "low" component loading.
    method : str, default="wilcoxon"
        `scanpy.tl.rank_genes_groups` method.

    Returns
    -------
    pd.Series
        Gene -> DE test statistic, sorted descending (most "high-group-up" first).
    """
    from .utils import add_obs_cmp_label, add_obs_cmp_unique_one

    if receiver_mode == "within_celltype":
        if celltype is None:
            raise ValueError(
                "`celltype` is required when receiver_mode='within_celltype'"
            )
        X = adata[adata.obs[celltype_col] == celltype].copy()
    elif receiver_mode == "vs_rest":
        X = adata.copy()
    else:
        raise ValueError(f"Unknown receiver_mode: {receiver_mode!r}")

    X = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=top_perc, type="receiver")
    X = add_obs_cmp_unique_one(X, cmp=cmp)

    n_pos = int((X.obs["Label"] == f"Cmp{cmp}").sum())
    n_neg = int((X.obs["Label"] == "NoLabel").sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Too few cells to test (n_pos={n_pos}, n_neg={n_neg}) for "
            f"cmp={cmp}, receiver_mode={receiver_mode!r}, celltype={celltype!r}."
        )

    sc.tl.rank_genes_groups(X, "Label", groups=[f"Cmp{cmp}"], method=method)
    results = X.uns["rank_genes_groups"]
    names = results["names"][f"Cmp{cmp}"]
    scores = results["scores"][f"Cmp{cmp}"]

    return pd.Series(scores, index=names).sort_values(ascending=False)


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


def top_receiver_celltype(
    adata: anndata.AnnData,
    cmp: int,
    celltype_col: str = "broad_cell_type",
) -> str:
    """Return the `celltype_col` group with the highest mean receiver-eigenstate loading for `cmp`."""
    loadings = np.asarray(adata.obsm["rc_C"])[:, cmp - 1]
    means = (
        pd.Series(loadings, index=adata.obs.index)
        .groupby(adata.obs[celltype_col])
        .mean()
    )
    return cast(str, means.idxmax())


def component_accuracy_scores(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    lt_df: pd.DataFrame,
    components: list[int],
    top_m_lr: int = 10,
    receiver_mode: ReceiverMode = "within_celltype",
    celltype_col: str = "broad_cell_type",
    top_perc: float = 10,
    de_method: Literal[
        "logreg", "t-test", "wilcoxon", "t-test_overestim_var"
    ] = "wilcoxon",
) -> pd.DataFrame:
    """
    Compute per-ligand NicheNet-style AUROCs for each of `components`.

    For each component: take its top `top_m_lr` LR pairs
    (`top_lr_pairs_for_component`); build one receiver DE ranking
    (`receiver_de_ranking`) -- restricted, when `receiver_mode ==
    "within_celltype"`, to the `celltype_col` group with the highest mean
    receiver-eigenstate loading for that component (`top_receiver_celltype`),
    mirroring how receiver cell types are chosen manually elsewhere (e.g.
    `figures/figure4.py`) -- then score every top ligand's downstream targets
    (`ligand_activity_auroc`) against that single ranking.

    Returns
    -------
    pd.DataFrame
        Columns: ["component", "ligand", "receptor", "loading", "celltype", "auroc"].
        `auroc` is NaN where it could not be computed (see `ligand_activity_auroc`).
    """
    D = adata.uns["D"]
    rows = []

    for cmp in components:
        top_pairs = top_lr_pairs_for_component(D, lr_pairs, cmp, top_m=top_m_lr)

        celltype = (
            top_receiver_celltype(adata, cmp, celltype_col=celltype_col)
            if receiver_mode == "within_celltype"
            else None
        )

        try:
            de_ranking = receiver_de_ranking(
                adata,
                cmp=cmp,
                receiver_mode=receiver_mode,
                celltype_col=celltype_col,
                celltype=celltype,
                top_perc=top_perc,
                method=de_method,
            )
        except ValueError:
            continue

        for _, pair in top_pairs.iterrows():
            ligand = cast(str, pair["ligand"])
            receptor = cast(str, pair["receptor"])
            auroc = ligand_activity_auroc(de_ranking, ligand, lt_df)
            rows.append(
                {
                    "component": cmp,
                    "ligand": ligand,
                    "receptor": receptor,
                    "loading": pair["loading"],
                    "celltype": celltype,
                    "auroc": auroc,
                }
            )

    return pd.DataFrame(rows)
