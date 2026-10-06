"""
NicheNet-style downstream-target validation of cccRISE / CPD components.

For each CPD component, the cells with the highest and lowest
receiver-eigenstate loading (`obsm["rc_C"]`) are contrasted by differential
expression, and every ligand the model sees is scored by how well that
ranking -- restricted to genes expressed in the two groups -- recovers the
ligand's literature-curated downstream targets (a NicheNet-style AUROC;
`component_ligand_scores`). Each component's top ligands (`top_pair_ligands`)
are then compared with all the other ligands on the same ranking
(`top_ligand_tests`): the question is whether the ligands the component
weights most are better supported by downstream evidence than ligands it
barely uses, not whether any ligand beats a shuffled-cell null.

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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data"
NICHENET_LT_URL = (
    "https://zenodo.org/records/7074291/files/ligand_target_matrix_nsga2r_final.rds"
)
NICHENET_LT_CACHE = DATA_DIR / "nichenet_ligand_target_matrix.parquet"

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


def _receiver_population(
    adata: anndata.AnnData, cmp: int, top_perc: float
) -> tuple[anndata.AnnData, np.ndarray]:
    """
    Split `adata` into the cells above the `100 - top_perc` percentile
    ("high") and below the `top_perc` percentile ("low") of component `cmp`'s
    receiver-eigenstate loading (`adata.obsm["rc_C"][:, cmp - 1]`), discarding
    every other cell.

    Returns
    -------
    tuple[anndata.AnnData, np.ndarray]
        (kept, pos_mask): the cells in either group, and a boolean array
        aligned to `kept` marking the "high" group.
    """
    loadings = np.asarray(adata.obsm["rc_C"])[:, cmp - 1]
    pos_idx = loadings > np.percentile(loadings, 100 - top_perc)
    neg_idx = loadings < np.percentile(loadings, top_perc)
    kept_idx = pos_idx | neg_idx
    return adata[kept_idx].copy(), pos_idx[kept_idx]


def _run_de(
    kept: anndata.AnnData,
    pos_mask: np.ndarray,
    cmp: int,
    method: DeMethod,
) -> pd.Series:
    """Rank `kept`'s genes by `scanpy.tl.rank_genes_groups` (`pos_mask` vs. the
    rest of `kept`), highest score first."""
    kept.obs["Label"] = np.where(pos_mask, f"Cmp{cmp}", "NoLabel")
    sc.tl.rank_genes_groups(kept, "Label", groups=[f"Cmp{cmp}"], method=method)
    results = kept.uns["rank_genes_groups"]
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
        Gene -> DE test statistic (from `receiver_ranking`), covering the
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


def expressed_genes(
    kept: anndata.AnnData, pos_mask: np.ndarray, min_frac: float = 0.1
) -> pd.Index:
    """Genes detected (non-zero) in at least `min_frac` of the "high" or of the
    "low" group of `kept` (`_receiver_population`)."""
    detected = cast(Any, kept.X) > 0
    frac_high = np.asarray(detected[pos_mask].mean(axis=0)).ravel()
    frac_low = np.asarray(detected[~pos_mask].mean(axis=0)).ravel()
    return kept.var_names[(frac_high >= min_frac) | (frac_low >= min_frac)]


def ligand_weights(D: np.ndarray, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    """Ligand x component (1-indexed columns) weight: the square root of the
    summed squared loadings of the ligand's LR pairs."""
    D = np.asarray(D)
    return np.sqrt(
        pd.DataFrame(
            D**2, index=lr_pairs["ligand"].to_numpy(), columns=range(1, D.shape[1] + 1)
        )
        .groupby(level=0)
        .sum()
    )


def top_pair_ligands(
    D: np.ndarray,
    lr_pairs: pd.DataFrame,
    cmp: int,
    scorable: set[str],
    rel_cutoff: float = 0.4,
) -> list[str]:
    """
    Ligands of component `cmp`'s LR pairs with |loading| >= `rel_cutoff` x the
    largest |loading| among its scorable pairs (pairs whose ligand is in
    `scorable`), ordered by loading. Unscorable pairs neither set the scale
    nor enter the set.
    """
    loading = np.abs(np.asarray(D)[:, cmp - 1])
    ligand = lr_pairs["ligand"].to_numpy()
    ok = np.isin(ligand, list(scorable))
    if not ok.any():
        return []
    rel = np.where(ok, loading, 0.0) / loading[ok].max()
    order = np.argsort(-rel)
    return list(dict.fromkeys(ligand[i] for i in order if rel[i] >= rel_cutoff))


def receiver_ranking(
    adata: anndata.AnnData,
    cmp: int,
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    min_expr_frac: float = 0.1,
) -> tuple[pd.Series, int, int] | None:
    """
    Component `cmp`'s receiver DE ranking: "high" vs "low" loading cells
    (`_receiver_population`), genes ranked by `_run_de`, restricted to
    `expressed_genes`.

    Returns
    -------
    tuple[pd.Series, int, int] or None
        (ranking, n_high, n_low), or None if either group has fewer than 2
        cells.
    """
    kept, pos_mask = _receiver_population(adata, cmp, top_perc)
    n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
    if n_pos < 2 or n_neg < 2:
        return None
    expressed = expressed_genes(kept, pos_mask, min_expr_frac)
    ranking = _run_de(kept, pos_mask, cmp, de_method)
    return ranking[ranking.index.isin(expressed)], n_pos, n_neg


def component_ligand_scores(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    lt_df: pd.DataFrame,
    components: list[int],
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    min_expr_frac: float = 0.1,
    rel_cutoff: float = 0.4,
) -> pd.DataFrame:
    """
    Score every ligand the model sees on each component's receiver ranking
    (`receiver_ranking`) and flag each component's top ligands.

    A ligand's AUROC (`ligand_activity_auroc`) is NaN where undefined, i.e.
    when none of its prior targets is expressed. The top set is
    `top_pair_ligands` over the scorable ligands.

    Parameters
    ----------
    adata : anndata.AnnData
        Dataset with the finalized fit loaded (`obsm["rc_C"]`, `uns["D"]`).
    lr_pairs : pd.DataFrame
        "ligand"/"receptor" columns in the row order of `uns["D"]`.
    lt_df : pd.DataFrame
        Long-format ligand-target prior (`load_ligand_target_matrix`).
    components : list[int]
        1-indexed CPD components to score.
    top_perc : float, default=10
        Percentile defining the "high"/"low" receiver groups.
    de_method : str, default="wilcoxon"
        `scanpy.tl.rank_genes_groups` method.
    min_expr_frac : float, default=0.1
        Minimum detection fraction in either group for a gene to be kept.
    rel_cutoff : float, default=0.4
        Top-set threshold, relative to the component's largest scorable pair.

    Returns
    -------
    pd.DataFrame
        One row per (component, ligand): ["component", "ligand", "auroc",
        "weight", "rel_weight" (weight / the component's top scorable
        ligand's), "is_top", "n_pos", "n_neg", "n_expressed_genes"].
        Components with fewer than 2 cells in either group are omitted.
    """
    D = adata.uns["D"]
    weights = ligand_weights(D, lr_pairs)
    rows = []
    for cmp in components:
        receivers = receiver_ranking(adata, cmp, top_perc, de_method, min_expr_frac)
        if receivers is None:
            continue
        ranking, n_pos, n_neg = receivers
        auroc = {
            lig: ligand_activity_auroc(ranking, lig, lt_df) for lig in weights.index
        }
        scorable = {lig for lig, a in auroc.items() if a is not None}
        top = set(top_pair_ligands(D, lr_pairs, cmp, scorable, rel_cutoff))
        w = weights[cmp]
        w_max = w[list(scorable)].max() if scorable else np.nan
        rows += [
            {
                "component": cmp,
                "ligand": lig,
                "auroc": np.nan if auroc[lig] is None else auroc[lig],
                "weight": w[lig],
                "rel_weight": w[lig] / w_max,
                "is_top": lig in top,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_expressed_genes": len(ranking),
            }
            for lig in weights.index
        ]
    return pd.DataFrame(rows)


def top_ligand_tests(
    scores: pd.DataFrame, correction: str | None = "fdr_bh", score: str = "auroc"
) -> pd.DataFrame:
    """
    Per component, a two-sided Mann-Whitney test of the top ligands' `score`
    (AUROC by default) against every other scored ligand's
    (`component_ligand_scores` output),
    with p-values adjusted across components (`anova.adjust_pvalues`, as in
    the ANOVA analysis).

    Returns
    -------
    pd.DataFrame
        One row per component: ["component", "top_ligands", "n_top",
        "n_comparison", "top_median", "comparison_median", "direction"
        ("up"/"down"), "p", "q"].
    """
    from ..anova import adjust_pvalues

    rows = []
    for cmp, c in scores.dropna(subset=[score]).groupby("component"):
        top = c[c["is_top"]].sort_values("weight", ascending=False)
        rest = c.loc[~c["is_top"], score]
        rows.append(
            {
                "component": cmp,
                "top_ligands": ", ".join(top["ligand"]),
                "n_top": len(top),
                "n_comparison": len(rest),
                "top_median": top[score].median(),
                "comparison_median": rest.median(),
                "direction": "up" if top[score].median() > rest.median() else "down",
                "p": mannwhitneyu(top[score], rest, alternative="two-sided").pvalue,
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = adjust_pvalues(out["p"].to_numpy(), method=correction)
    return out
