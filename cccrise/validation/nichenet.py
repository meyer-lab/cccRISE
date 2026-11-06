"""
NicheNet-style downstream-target validation of cccRISE / CPD components.

For each CPD component, the cells of its receiver cell types
(`receiver_celltypes`) with the highest receiver-eigenstate loading
(`obsm["rc_C"]`) are contrasted with the lowest-loading cells of the whole
dataset by differential expression, and every ligand the model sees is
scored by how well that ranking -- restricted to genes expressed in the two
groups -- recovers the ligand's literature-curated downstream targets
(`ligand_activities`, a NicheNet-style AUROC in which targets many ligands
share count less; `component_ligand_scores`). The question is whether the
ligands a component weights most are better supported by downstream evidence
than the ligands it barely uses: each component is summarized by the ligands'
score percentiles averaged with the ligand weights as weights, and tested
against the same average on copies of the NicheNet prior with its
ligand-target links shuffled at random (`shuffled_percentiles`,
`weighted_percentile_tests`).

The ligand-target prior is the NicheNet v2 ligand-target matrix (Zenodo
record 7074291), which is only published as an R `.rds` object; it is read
here with `pyreadr` (a pure-Python RDS parser) rather than adding an R/rpy2
dependency. `fetch_ligand_target_matrix` downloads the full matrix once,
restricts it to the ligand vocabulary actually used by this pipeline's LR
resource, and caches a small long-format parquet (mirroring how
`Human-2020-Jin-LR-pairs.parquet` is cached in `import_data.py`) so that
neither the network nor `pyreadr` is required afterwards.
"""

import multiprocessing
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata

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


def receiver_celltypes(
    adata: anndata.AnnData, cmp: int, celltype_key: str, rel_cutoff: float = 0.4
) -> list[str]:
    """Component `cmp`'s receiver cell types: those whose mean
    receiver-eigenstate loading is at least `rel_cutoff` x the largest
    cell-type mean."""
    loadings = np.asarray(adata.obsm["rc_C"])[:, cmp - 1]
    means = (
        pd.Series(loadings)
        .groupby(adata.obs[celltype_key].astype(str).to_numpy())
        .mean()
    )
    return list(means.index[means >= rel_cutoff * means.max()])


def receiver_masks(
    adata: anndata.AnnData, cmp: int, top_perc: float, celltype_key: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    The cells above the `100 - top_perc` percentile ("high") and below the
    `top_perc` percentile ("low") of component `cmp`'s receiver-eigenstate
    loading (`adata.obsm["rc_C"][:, cmp - 1]`). With `celltype_key`, the "high"
    group is taken among the cells of the component's `receiver_celltypes`
    only, so that a rare receiver cell type is not diluted by abundant
    low-loading cells; the "low" group is still taken among all cells.
    """
    loadings = np.asarray(adata.obsm["rc_C"])[:, cmp - 1]
    pool = np.ones(len(loadings), dtype=bool)
    if celltype_key is not None:
        types = receiver_celltypes(adata, cmp, celltype_key)
        pool = adata.obs[celltype_key].astype(str).isin(types).to_numpy()
    pos_idx = pool & (loadings > np.percentile(loadings[pool], 100 - top_perc))
    neg_idx = ~pos_idx & (loadings < np.percentile(loadings, top_perc))
    return pos_idx, neg_idx


def _receiver_population(
    adata: anndata.AnnData, cmp: int, top_perc: float, celltype_key: str | None = None
) -> tuple[anndata.AnnData, np.ndarray]:
    """
    `adata`'s cells in either of `receiver_masks`' groups, and a boolean array
    aligned to them marking the "high" group.
    """
    pos_idx, neg_idx = receiver_masks(adata, cmp, top_perc, celltype_key)
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


def rank_sum_statistics(
    X, masks: np.ndarray, chunk: int = 1500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Every row of `masks` (groups x cells) as `_run_de`'s Wilcoxon test would
    score it, at once: for fixed cells the ranks are fixed, so a group's rank
    sum is a product with the rank matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Each group's per-gene z, and the share of cells each gene is detected
        in within the group and out of it.
    """
    n, n_genes = X.shape
    groups = masks.astype(np.float64)
    inside = masks.sum(axis=1, keepdims=True).astype(float)
    outside = n - inside
    z, frac_in, frac_out = (
        np.empty((len(masks), n_genes), np.float32) for _ in range(3)
    )
    by_gene = X.tocsc()
    for start in range(0, n_genes, chunk):
        block = slice(start, start + chunk)
        values = by_gene[:, block].toarray()
        sums = groups @ rankdata(values, axis=0)
        z[:, block] = (sums - inside * (n + 1) / 2) / np.sqrt(
            inside * outside * (n + 1) / 12
        )
        detected = (values > 0).astype(np.float64)
        hits = groups @ detected
        frac_in[:, block] = hits / inside
        frac_out[:, block] = (detected.sum(axis=0) - hits) / outside
    return z, frac_in, frac_out


def statistic_ranking(
    z: np.ndarray,
    frac_in: np.ndarray,
    frac_out: np.ndarray,
    genes: np.ndarray,
    min_frac: float = 0.1,
) -> pd.Series:
    """One group of `rank_sum_statistics` as a DE ranking of the genes detected
    in at least `min_frac` of either side (`expressed_genes`)."""
    kept = (frac_in >= min_frac) | (frac_out >= min_frac)
    return pd.Series(z[kept], index=genes[kept]).sort_values(ascending=False)


def target_idf(lt_df: pd.DataFrame) -> np.ndarray:
    """Per row of `lt_df`, log(number of ligands / the number of ligands
    listing the target): hub targets that most ligands list count ~0."""
    sharedness = lt_df.groupby("target")["ligand"].transform("size")
    return np.log(lt_df["ligand"].nunique() / sharedness).to_numpy()


def ligand_activities(de_ranking: pd.Series, lt_df: pd.DataFrame) -> pd.Series:
    """
    NicheNet-style activity of every ligand in `lt_df` on one DE ranking: the
    AUROC of `de_ranking`'s statistic for the ligand's cached top targets
    against every other ranked gene, each target counted with its
    `target_idf` weight and each other gene once (ties averaged).

    A target's count of non-targets ranked below it is its rank among all
    ranked genes minus its rank among the ligand's ranked targets.

    Parameters
    ----------
    de_ranking : pd.Series
        Gene -> DE test statistic (from `receiver_ranking`), covering the
        receiver population's expressed/tested genes.
    lt_df : pd.DataFrame
        Long-format ligand-target matrix (columns "ligand", "target", "weight"),
        from `load_ligand_target_matrix`/`fetch_ligand_target_matrix`.

    Returns
    -------
    pd.Series
        Ligand -> activity; NaN where none or all of the ranked genes are the
        ligand's targets (AUROC undefined).
    """
    pos = de_ranking.index.get_indexer(lt_df["target"])
    hit = pos >= 0
    df = pd.DataFrame(
        {
            "ligand": lt_df["ligand"].to_numpy()[hit],
            "rank": rankdata(de_ranking.to_numpy())[pos[hit]],
            "w": target_idf(lt_df)[hit],
        }
    )
    df["below"] = df["w"] * (df["rank"] - df.groupby("ligand")["rank"].rank())
    g = df.groupby("ligand")
    n_neg = len(de_ranking) - g.size()
    auc = g["below"].sum() / (g["w"].sum() * n_neg)
    auc[n_neg == 0] = np.nan
    return auc.reindex(lt_df["ligand"].unique())


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
    Ligands of component `cmp`'s top LR pairs (`top_pairs`), ordered by
    loading.
    """
    ligand = lr_pairs["ligand"].to_numpy()
    top = top_pairs(D, lr_pairs, cmp, scorable, rel_cutoff)
    return list(dict.fromkeys(ligand[top]))


def top_pairs(
    D: np.ndarray,
    lr_pairs: pd.DataFrame,
    cmp: int,
    scorable: set[str],
    rel_cutoff: float = 0.4,
) -> np.ndarray:
    """
    Row indices of component `cmp`'s LR pairs with |loading| >= `rel_cutoff`
    x the largest |loading| among its scorable pairs (pairs whose ligand is in
    `scorable`), ordered by loading. Unscorable pairs neither set the scale
    nor enter the set.
    """
    loading = np.abs(np.asarray(D)[:, cmp - 1])
    ok = np.isin(lr_pairs["ligand"].to_numpy(), list(scorable))
    if not ok.any():
        return np.array([], dtype=int)
    rel = np.where(ok, loading, 0.0) / loading[ok].max()
    order = np.argsort(-rel)
    return order[rel[order] >= rel_cutoff]


def receiver_ranking(
    adata: anndata.AnnData,
    cmp: int,
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    min_expr_frac: float = 0.1,
    celltype_key: str | None = None,
) -> tuple[pd.Series, int, int] | None:
    """
    Component `cmp`'s receiver DE ranking: "high" vs "low" loading cells
    (`_receiver_population`, with `celltype_key`), genes ranked by `_run_de`,
    restricted to `expressed_genes`.

    Returns
    -------
    tuple[pd.Series, int, int] or None
        (ranking, n_high, n_low), or None if either group has fewer than 2
        cells.
    """
    kept, pos_mask = _receiver_population(adata, cmp, top_perc, celltype_key)
    n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
    if n_pos < 2 or n_neg < 2:
        return None
    expressed = expressed_genes(kept, pos_mask, min_expr_frac)
    ranking = _run_de(kept, pos_mask, cmp, de_method)
    return ranking[ranking.index.isin(expressed)], n_pos, n_neg


def receiver_rankings(
    adata: anndata.AnnData,
    components: list[int],
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    min_expr_frac: float = 0.1,
    celltype_key: str | None = None,
) -> dict[int, tuple[pd.Series, int, int]]:
    """`receiver_ranking` of every component in `components`; components with
    fewer than 2 cells in either group are omitted."""
    rankings = {
        cmp: receiver_ranking(
            adata, cmp, top_perc, de_method, min_expr_frac, celltype_key
        )
        for cmp in components
    }
    return {cmp: r for cmp, r in rankings.items() if r is not None}


def component_ligand_scores(
    D: np.ndarray,
    lr_pairs: pd.DataFrame,
    lt_df: pd.DataFrame,
    rankings: dict[int, tuple[pd.Series, int, int]],
    rel_cutoff: float = 0.4,
) -> pd.DataFrame:
    """
    Score every ligand the model sees on each component's receiver ranking
    (`receiver_rankings`) and flag each component's top ligands.

    A ligand's score (`ligand_activities`) is NaN where undefined, i.e.
    when none of its prior targets is expressed. The top set is
    `top_pair_ligands` over the scorable ligands.

    Parameters
    ----------
    D : np.ndarray
        LR-pair x component loadings of the finalized fit (`uns["D"]`).
    lr_pairs : pd.DataFrame
        "ligand"/"receptor" columns in the row order of `D`.
    lt_df : pd.DataFrame
        Long-format ligand-target prior (`load_ligand_target_matrix`).
    rankings : dict
        1-indexed component -> (ranking, n_high, n_low), `receiver_rankings`.
    rel_cutoff : float, default=0.4
        Top-set threshold, relative to the component's largest scorable pair.

    Returns
    -------
    pd.DataFrame
        One row per (component, ligand): ["component", "ligand", "auroc",
        "weight", "rel_weight" (weight / the component's top scorable
        ligand's), "is_top", "n_pos", "n_neg", "n_expressed_genes"].
    """
    weights = ligand_weights(D, lr_pairs)
    rows = []
    for cmp, (ranking, n_pos, n_neg) in rankings.items():
        auroc = ligand_activities(ranking, lt_df).reindex(weights.index)
        scorable = set(auroc.dropna().index)
        top = set(top_pair_ligands(D, lr_pairs, cmp, scorable, rel_cutoff))
        w = weights[cmp]
        w_max = w[list(scorable)].max() if scorable else np.nan
        rows += [
            {
                "component": cmp,
                "ligand": lig,
                "auroc": auroc[lig],
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


def score_percentiles(scores: pd.DataFrame) -> pd.DataFrame:
    """The scored ligands of `component_ligand_scores` output, with each
    ligand's score percentile among its component's ligands ("score_pct",
    average ranks / number of ligands)."""
    scored = scores.dropna(subset=["auroc"]).copy()
    scored["score_pct"] = scored.groupby("component")["auroc"].rank(pct=True)
    return scored


def weighted_percentile(
    percentiles: pd.DataFrame, by=("component",), value: str = "score_pct"
) -> pd.Series:
    """Per group, the ligands' `value` (by default score percentiles) averaged
    with the ligand weights as weights: for percentiles, where the ligands the
    component uses sit in the score ranking, 0.5 if weight is unrelated to
    score."""
    g = percentiles.assign(wp=percentiles["weight"] * percentiles[value])
    sums = g.groupby(list(by))[["wp", "weight"]].sum()
    return sums["wp"] / sums["weight"]


def shuffle_prior(lt_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The long-format prior with its ligand-target links placed uniformly at
    random on the grid of its ligands x targets (as many links, no other
    structure kept)."""
    rng = np.random.default_rng(seed)
    ligands, targets = lt_df["ligand"].unique(), lt_df["target"].unique()
    cells = rng.choice(len(ligands) * len(targets), len(lt_df), replace=False)
    return pd.DataFrame(
        {
            "ligand": ligands[cells // len(targets)],
            "target": targets[cells % len(targets)],
        }
    )


def _shuffled_percentiles(rankings, lt_df, ligands, seed: int) -> pd.DataFrame:
    shuffled = shuffle_prior(lt_df, seed)
    return pd.concat(
        [
            ligand_activities(rankings[cmp][0], shuffled)
            .reindex(ligs)
            .dropna()
            .rank(pct=True)
            .rename("score_pct")
            .rename_axis("ligand")
            .reset_index()
            .assign(component=cmp, replicate=seed)
            for cmp, ligs in ligands.items()
        ],
        ignore_index=True,
    )


def shuffled_percentiles(
    scores: pd.DataFrame,
    rankings: dict[int, tuple[pd.Series, int, int]],
    lt_df: pd.DataFrame,
    n_shuffle: int = 1000,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    The shuffled-prior null of `score_percentiles`: every scored ligand of
    `scores` re-scored on its component's ranking (`rankings`) with each of
    `n_shuffle` `shuffle_prior` copies of `lt_df` (seeds 0..n_shuffle-1), as
    percentiles among the component's scored ligands.

    Returns
    -------
    pd.DataFrame
        ["ligand", "score_pct", "component", "replicate"].
    """
    scored = scores.dropna(subset=["auroc"])
    ligands = {cmp: list(g["ligand"]) for cmp, g in scored.groupby("component")}
    job = partial(_shuffled_percentiles, rankings, lt_df, ligands)
    with ProcessPoolExecutor(n_jobs) as pool:
        return pd.concat(pool.map(job, range(n_shuffle)), ignore_index=True)


def weighted_percentile_tests(
    scores: pd.DataFrame,
    null_percentiles: pd.DataFrame,
    correction: str | None = "fdr_bh",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Per component, the `weighted_percentile` of `component_ligand_scores`
    output against its value on each replicate of `null_percentiles`
    (`shuffled_percentiles`), the ligand weights held fixed: one-sided
    permutation p, adjusted across components (`anova.adjust_pvalues`, as in
    the ANOVA analysis).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        One row per component: ["component", "top_ligands", "weighted_pct",
        "null_mean", "p", "q"]; and the null's weighted percentiles, indexed
        by (component, replicate).
    """
    percentiles = score_percentiles(scores)
    observed = weighted_percentile(percentiles)
    null = weighted_percentile(
        null_weights(null_percentiles, scores), by=("component", "replicate")
    )
    out = _permutation_tests(observed, null, correction).rename(
        columns={"observed": "weighted_pct"}
    )
    out.insert(
        1,
        "top_ligands",
        [
            ", ".join(c[c["is_top"]].sort_values("weight", ascending=False)["ligand"])
            for _, c in percentiles.groupby("component")
        ],
    )
    return out, null


def _permutation_tests(
    observed: pd.Series, null: pd.Series, correction: str | None
) -> pd.DataFrame:
    """Per component, `observed` against its `null` (indexed by (component,
    replicate)): one-sided permutation p, (1 + #null >= observed) / (1 + n),
    adjusted across components (`anova.adjust_pvalues`)."""
    from ..anova import adjust_pvalues

    n_null = null.groupby(level=0).size()
    out = pd.DataFrame(
        {
            "component": observed.index,
            "observed": observed.to_numpy(),
            "null_mean": null.groupby(level=0).mean().loc[observed.index].to_numpy(),
            "p": [
                (1 + (null.loc[k] >= v - 1e-12).sum()) / (1 + n_null[k])
                for k, v in observed.items()
            ],
        }
    )
    out["q"] = adjust_pvalues(out["p"].to_numpy(), method=correction)
    return out


def null_weights(null_percentiles: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """`null_percentiles` with each ligand's weight in its component from
    `scores` ("weight" column)."""
    return null_percentiles.merge(
        scores[["component", "ligand", "weight"]], on=["component", "ligand"]
    )


_CELL_SHUFFLE: dict = {}  # per component (kept, pos_mask), and the run's settings


# No caller yet: `run_nichenet --n-cell-shuffle` takes it if Figure 2 moves to this
# test (#277). `run_comparator_nichenet` shuffles through `rank_sum_statistics`.
def shuffled_cell_scores(
    adata: anndata.AnnData,
    scores: pd.DataFrame,
    lt_df: pd.DataFrame,
    n_shuffle: int = 1000,
    n_jobs: int = 1,
    top_perc: float = 10,
    de_method: DeMethod = "wilcoxon",
    min_expr_frac: float = 0.1,
    celltype_key: str | None = None,
) -> pd.DataFrame:
    """
    The shuffled-cell null of `component_ligand_scores`: each component's
    receiver ranking (`receiver_ranking`) recomputed `n_shuffle` times with
    the "high" label permuted among its high and low cells (seeds
    0..n_shuffle-1; group sizes kept), every scored ligand of `scores`
    re-scored on each. Forked workers share the cells, so `n_jobs` > 1 needs
    a platform with fork.

    Returns
    -------
    pd.DataFrame
        ["component", "ligand", "auroc", "replicate"].
    """
    scored = scores.dropna(subset=["auroc"])
    _CELL_SHUFFLE.clear()
    _CELL_SHUFFLE.update(
        {
            cmp: _receiver_population(adata, cmp, top_perc, celltype_key)
            for cmp in scored["component"].unique()
        },
        settings=(lt_df, de_method, min_expr_frac),
        ligands={cmp: list(g["ligand"]) for cmp, g in scored.groupby("component")},
    )
    jobs = [
        (cmp, seed) for cmp in _CELL_SHUFFLE["ligands"] for seed in range(n_shuffle)
    ]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(n_jobs, mp_context=context) as pool:
        return pd.concat(pool.map(_shuffled_cell_scores, jobs), ignore_index=True)


def _shuffled_cell_scores(job: tuple[int, int]) -> pd.DataFrame:
    cmp, seed = job
    kept, pos_mask = _CELL_SHUFFLE[cmp]
    lt_df, de_method, min_expr_frac = _CELL_SHUFFLE["settings"]
    pos_mask = np.random.default_rng(seed).permutation(pos_mask)
    kept = kept.copy()  # `_run_de` writes its labels into obs
    ranking = _run_de(kept, pos_mask, cmp, de_method)
    ranking = ranking[
        ranking.index.isin(expressed_genes(kept, pos_mask, min_expr_frac))
    ]
    return (
        ligand_activities(ranking, lt_df)
        .reindex(_CELL_SHUFFLE["ligands"][cmp])
        .rename("auroc")
        .rename_axis("ligand")
        .reset_index()
        .assign(component=cmp, replicate=seed)
    )


def weighted_score_tests(
    scores: pd.DataFrame,
    null_scores: pd.DataFrame,
    correction: str | None = "fdr_bh",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Per component, the ligand-weighted mean target score (`weighted_percentile`
    of "auroc") of `component_ligand_scores` output against its value on each
    replicate of `shuffled_cell_scores`, the ligand weights held fixed, over
    the ligands scored on every ranking. Each
    ligand counts once however many LR pairs share it, and the replicate (a
    whole reshuffled ranking) is the permutation unit: one-sided permutation
    p, adjusted across components.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        One row per component: ["component", "weighted_auroc", "null_mean",
        "p", "q"]; and the null's weighted means, indexed by (component,
        replicate).
    """
    # Only ligands scored on every ranking, so both sides average the same set
    null = null_weights(null_scores, scores)
    scored = null.groupby(["component", "ligand"])["auroc"].transform(
        lambda v: v.notna().all()
    )
    null = null[scored]
    kept = scores.merge(null[["component", "ligand"]].drop_duplicates())
    observed = weighted_percentile(kept.dropna(subset=["auroc"]), value="auroc")
    null = weighted_percentile(null, by=("component", "replicate"), value="auroc")
    tests = _permutation_tests(observed, null, correction)
    return tests.rename(columns={"observed": "weighted_auroc"}), null
