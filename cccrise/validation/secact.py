"""
SecAct-style downstream-target validation, and a comparison of the SecAct
and NicheNet v2 ligand-target priors.

SecAct (Ru et al., Nat. Methods 2026, doi:10.1038/s41592-026-03172-0;
github.com/data2intelligence/SecAct) publishes a signed gene x secreted
protein signature matrix (1,170 proteins, spatial colocalization with the
protein's coding gene across 1,258 Visium samples). Unlike NicheNet's
top-target AUROC (`nichenet.ligand_activity_auroc`), SecAct infers a
protein's activity by ridge-regressing a differential expression profile on
all (grouped, standardized) signatures jointly and z-scoring each coefficient
against gene-label permutations (`ridge_activity`, a port of SecAct's
`.ridge_pureR`). Its CCC receiver profile is a pseudobulk log2(TPM + 1)
difference between case and control cells over all genes
(`pseudobulk_log_tpm_diff`, after `SecAct.CCC.scRNAseq`).

`signature_matrix` caches either prior as a dense genes x ligands parquet
(gitignored; both are rebuilt from their public sources), and
`top_targets_long` turns one into the long top-N format `nichenet.py` uses.
"""

import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

import anndata
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .nichenet import DATA_DIR, NICHENET_LT_URL, ligand_weights, top_pair_ligands

Prior = Literal["nichenet", "secact"]

SECACT_URL = (
    "https://raw.githubusercontent.com/data2intelligence/SecAct/main/"
    "inst/extdata/SecAct.tsv.gz"
)
SIGNATURE_DIR = DATA_DIR / "signatures"
PRIOR_URLS: dict[Prior, str] = {"nichenet": NICHENET_LT_URL, "secact": SECACT_URL}
#: Ridge penalties: SecAct's package default, and the value its paper tuned
#: for NicheNet v2's regulatory potentials (Methods, "Benchmark").
PRIOR_LAMBDA: dict[Prior, float] = {"nichenet": 1e7, "secact": 5e5}
GROUP_COR = 0.9
N_PERMUTATIONS = 1000
TPM_SCALE = 1e6


def signature_matrix(prior: Prior, url: str | None = None) -> pd.DataFrame:
    """Dense genes x ligands signature matrix for `prior`, downloaded once
    from `PRIOR_URLS[prior]` (or `url`, e.g. a local ``file://`` copy) and
    cached under `SIGNATURE_DIR`."""
    cache = SIGNATURE_DIR / f"{prior}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw"
        urllib.request.urlretrieve(url or PRIOR_URLS[prior], raw)
        if prior == "secact":
            matrix = pd.read_csv(raw, sep="\t", index_col=0, compression="gzip")
        else:
            import pyreadr

            matrix = next(iter(pyreadr.read_r(raw).values()))
    matrix = matrix.astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache)
    return matrix


def top_targets_long(signatures: pd.DataFrame, top_n: int = 250) -> pd.DataFrame:
    """Each ligand's `top_n` highest-weight targets, in the long
    ["ligand", "target", "weight"] format of `nichenet.load_ligand_target_matrix`.
    For SecAct's signed signatures this keeps the most positively
    colocalized genes."""
    return pd.concat(
        [
            pd.DataFrame({"ligand": lig, "target": top.index, "weight": top.to_numpy()})
            for lig, top in (
                (lig, signatures[lig].nlargest(top_n)) for lig in signatures
            )
        ],
        ignore_index=True,
    )


def group_signatures(
    signatures: pd.DataFrame, group_cor: float = GROUP_COR
) -> tuple[pd.DataFrame, pd.Series]:
    """Average signatures whose complete-linkage Pearson clusters exceed
    `group_cor` (SecAct's `cluster_signatures`), dropping constant ones.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (grouped genes x groups matrix, ligand -> group label).
    """
    signatures = signatures.loc[:, signatures.std() > 0]
    corr = np.corrcoef(signatures.to_numpy().T)
    dist = squareform(1 - corr, checks=False).clip(min=0)
    labels = fcluster(linkage(dist, "complete"), 1 - group_cor, criterion="distance")
    groups = pd.Series(labels, index=signatures.columns)
    return signatures.T.groupby(groups.to_numpy()).mean().T, groups


def ridge_activity(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    lam: float,
    nrand: int = N_PERMUTATIONS,
    seed: int = 0,
    batch: int = 50,
) -> pd.DataFrame:
    """
    Permutation z-scores of ridge coefficients of `Y` (genes x profiles) on
    `X` (genes x signatures), both column-standardized over their shared
    genes (SecAct's `.ridge_pureR`; numpy RNG in place of GSL's MT19937).

    Returns
    -------
    pd.DataFrame
        Signatures x profiles z-scores.
    """
    genes = X.index.intersection(Y.index)
    x, y = (_standardize(m.loc[genes].to_numpy(np.float64)) for m in (X, Y))
    proj = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T)
    beta = proj @ y
    rng = np.random.default_rng(seed)
    total = np.zeros_like(beta)
    total_sq = np.zeros_like(beta)
    for start in range(0, nrand, batch):
        perms = [rng.permutation(len(genes)) for _ in range(min(batch, nrand - start))]
        rand = (proj @ np.hstack([y[p] for p in perms])).reshape(
            len(beta), len(perms), -1
        )
        total += rand.sum(axis=1)
        total_sq += (rand**2).sum(axis=1)
    mean = total / nrand
    sd = np.sqrt(total_sq / nrand - mean**2)
    return pd.DataFrame((beta - mean) / sd, index=X.columns, columns=Y.columns)


def _standardize(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=0)) / m.std(axis=0, ddof=1)


def pseudobulk_log_tpm_diff(counts: anndata.AnnData, pos_mask: np.ndarray) -> pd.Series:
    """log2(TPM + 1) of the summed raw counts of the `pos_mask` cells minus
    that of the rest (SecAct.CCC.scRNAseq's receiver profile), over genes
    detected in either group."""
    log_tpm = [
        np.log2(TPM_SCALE * s / s.sum() + 1)
        for s in (
            np.asarray(cast(Any, counts.X)[mask].sum(axis=0), dtype=np.float64).ravel()
            for mask in (pos_mask, ~pos_mask)
        )
    ]
    diff = pd.Series(log_tpm[0] - log_tpm[1], index=counts.var_names)
    return diff[(log_tpm[0] > 0) | (log_tpm[1] > 0)]


def score_rows(
    cmp: int,
    scores: dict[str, float],
    D: np.ndarray,
    lr_pairs: pd.DataFrame,
    rel_cutoff: float = 0.4,
    universe: set[str] | None = None,
) -> list[dict]:
    """
    `nichenet.component_ligand_scores`-style rows for one component from a
    precomputed ligand -> score map (NaN = unscorable). The top set is
    `top_pair_ligands` over the scorable ligands, intersected with `universe`
    if given (to hold the top set fixed across priors).
    """
    weights = ligand_weights(D, lr_pairs)[cmp]
    scorable = {lig for lig, s in scores.items() if np.isfinite(s)}
    if universe is not None:
        scorable &= universe
    top = set(top_pair_ligands(D, lr_pairs, cmp, scorable, rel_cutoff))
    w_max = weights[list(scorable)].max() if scorable else np.nan
    return [
        {
            "component": cmp,
            "ligand": lig,
            "score": scores.get(lig, np.nan) if lig in scorable else np.nan,
            "weight": weights[lig],
            "rel_weight": weights[lig] / w_max,
            "is_top": lig in top,
        }
        for lig in weights.index
    ]


def ridge_ligand_scores(
    profile: pd.Series,
    grouped: pd.DataFrame,
    groups: pd.Series,
    lam: float,
    nrand: int = N_PERMUTATIONS,
) -> pd.Series:
    """Ligand -> ridge z-score (`ridge_activity`) of `profile` on the
    `group_signatures` output, grouped ligands sharing their group's score."""
    z = ridge_activity(grouped, profile.to_frame(), lam, nrand).iloc[:, 0]
    return pd.Series(z.loc[groups.to_numpy()].to_numpy(), index=groups.index)
