"""
Gene-set enrichment of CPD components over the ligand-receptor (LR) mode (#144).

Reproduces Tensor-cell2cell's LR-GSEA (`cell2cell.external.gseapy`): each term
of a gene-set library is converted to the set of LR interactions whose ligand
and receptor subunits all belong to it (`generate_lr_geneset`), and each
component's LR loadings are then tested against those sets by preranked GSEA
(`lr_gsea`). The one departure is 19,999 permutations instead of 999, since a
permutation p-value cannot fall below ``1 / (permutations + 1)`` and 999 leaves
BH unable to call any single term across ~500 tests.
"""

import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

import gseapy
import pandas as pd

from ..anova import adjust_pvalues
from ..import_data import import_ligand_receptor_pairs

# The MSigDB gene-set libraries Tensor-cell2cell distributes (Zenodo 7593519).
Library = Literal["GOBP", "KEGG", "Reactome"]
LIBRARIES: tuple[Library, ...] = ("GOBP", "KEGG", "Reactome")
LIBRARY_URL = (
    "https://zenodo.org/record/7593519/files/Human-2023-{library}-msigdb-symbols.gmt"
)
DATA_DIR = Path(__file__).parent.parent / "data"

# GSEA settings; all but PERMUTATIONS are cell2cell's `run_gsea` defaults.
PERMUTATIONS = 19_999
MIN_SIZE = 15
WEIGHT = 1.0
SEED = 6

RESULT_COLUMNS = ["component", "term", "nes", "pval", "padj", "leading_edge"]


def _interaction_genes(interaction: str) -> list[str]:
    """Split an LR interaction into its member genes.

    ``"TGFB1^TGFBR1&TGFBR2"`` -> ``["TGFB1", "TGFBR1", "TGFBR2"]``.
    """
    ligand, receptor = interaction.split("^")
    return ligand.split("&") + receptor.split("&")


def lr_resource_interactions() -> list[str]:
    """Return the interaction symbols of the full CellChat LR resource."""
    return import_ligand_receptor_pairs()["interaction_symbol"].astype(str).tolist()


def _library_cache(library: Library) -> Path:
    """Where a library's LR-gene subset is cached."""
    return DATA_DIR / f"{library.lower()}_msigdb_2023_lr_genes.parquet"


def _download_library(library: Library, keep_genes: set[str]) -> pd.DataFrame:
    """Download a library's GMT and return its (term, gene) pairs for `keep_genes`.

    GMT rows are ``term <tab> url <tab> gene1 <tab> gene2 ...``.
    """
    with urllib.request.urlopen(LIBRARY_URL.format(library=library)) as response:
        lines = response.read().decode().splitlines()

    rows = []
    for line in lines:
        term, _url, *genes = line.split("\t")
        rows.extend((term, gene) for gene in genes if gene in keep_genes)
    return pd.DataFrame(rows, columns=["term", "gene"])


def load_pathway_per_gene(library: Library = "GOBP") -> dict[str, set[str]]:
    """Return ``{gene: {terms}}`` of `library` for every gene in the LR resource.

    Parameters
    ----------
    library : {"GOBP", "KEGG", "Reactome"}, default="GOBP"
        MSigDB gene-set library, as in `cell2cell.datasets.gsea_msig`. The GMT
        is downloaded once and cached, keeping only LR-resource genes because
        those are the only ones `generate_lr_geneset` can use.

    Returns
    -------
    dict[str, set[str]]
        ``{gene: {terms}}``.
    """
    if library not in LIBRARIES:
        raise ValueError(f"Unknown library {library!r}; choose from {LIBRARIES}.")

    cache = _library_cache(library)
    if not cache.exists():
        lr_genes = {
            gene
            for interaction in lr_resource_interactions()
            for gene in _interaction_genes(interaction)
        }
        _download_library(library, lr_genes).to_parquet(cache, index=False)

    memberships = pd.read_parquet(cache)
    return {
        str(gene): set(terms) for gene, terms in memberships.groupby("gene")["term"]
    }


def generate_lr_geneset(
    lr_list: list[str],
    pathway_per_gene: dict[str, set[str]],
    min_size: int = MIN_SIZE,
) -> dict[str, set[str]]:
    """Convert gene-level terms into LR-level terms.

    Port of `cell2cell.external.gseapy.generate_lr_geneset`.

    Parameters
    ----------
    lr_list : list[str]
        LR interaction symbols, e.g. from `lr_resource_interactions`.
    pathway_per_gene : dict[str, set[str]]
        ``{gene: {terms}}``, e.g. from `load_pathway_per_gene`.
    min_size : int, default=MIN_SIZE
        Drop terms containing fewer LR interactions than this.

    Returns
    -------
    dict[str, set[str]]
        ``{term: {interactions}}``, where an interaction is included only if
        every one of its genes is annotated to the term.
    """
    interactions_per_term: dict[str, set[str]] = {}
    for interaction in lr_list:
        gene_terms = [
            pathway_per_gene.get(gene, set())
            for gene in _interaction_genes(interaction)
        ]
        for term in set.intersection(*gene_terms):
            interactions_per_term.setdefault(term, set()).add(interaction)

    return {
        term: interactions
        for term, interactions in interactions_per_term.items()
        if len(interactions) >= min_size
    }


def _prerank(
    ranking: pd.Series, lr_set: dict[str, set[str]], permutations: int
) -> pd.DataFrame:
    """Run `gseapy.prerank` on one component's ranking and tidy its output."""
    result = gseapy.prerank(
        rnk=ranking.sort_values(ascending=False),
        gene_sets=cast(Any, {term: sorted(lrs) for term, lrs in lr_set.items()}),
        weight=WEIGHT,
        min_size=MIN_SIZE,
        permutation_num=permutations,
        seed=SEED,
        outdir=None,
        no_plot=True,
    )
    table = cast(pd.DataFrame, result.res2d)
    return pd.DataFrame(
        {
            "term": table["Term"],
            "nes": table["NES"].astype(float),
            "pval": table["NOM p-val"].astype(float),
            "leading_edge": table["Lead_genes"],
        }
    )


def lr_gsea(
    loadings: pd.DataFrame,
    lr_set: dict[str, set[str]],
    permutations: int = PERMUTATIONS,
) -> pd.DataFrame:
    """Run preranked GSEA on each component's LR loadings.

    Port of `cell2cell.external.gseapy.run_gsea`.

    Parameters
    ----------
    loadings : pd.DataFrame
        LR pairs x components (e.g. `cp_D`), indexed by interaction symbol.
    lr_set : dict[str, set[str]]
        LR-level terms from `generate_lr_geneset`.
    permutations : int, default=PERMUTATIONS
        Permutations per component.

    Returns
    -------
    pd.DataFrame
        One row per (component, term) with columns `RESULT_COLUMNS`, sorted
        by ``padj`` (BH across all components and terms).
    """
    per_component = []
    for component in loadings.columns:
        component_result = _prerank(loadings[component], lr_set, permutations)
        per_component.append(component_result.assign(component=component))
    results = pd.concat(per_component, ignore_index=True).dropna(subset=["pval"])

    # A permutation p-value is never below 1 / (permutations + 1).
    results["pval"] = results["pval"].clip(lower=1.0 / (permutations + 1))
    results["padj"] = adjust_pvalues(results["pval"].to_numpy(), method="fdr_bh")

    return (
        results[RESULT_COLUMNS]
        .sort_values("padj", kind="stable")
        .reset_index(drop=True)
    )
