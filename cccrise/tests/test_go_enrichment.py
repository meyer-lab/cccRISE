"""Unit tests for cccrise.validation.go_enrichment (no network access)."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from cccrise.validation.go_enrichment import (
    generate_lr_geneset,
    load_pathway_per_gene,
    lr_gsea,
)


def test_generate_lr_geneset_requires_every_member():
    # An interaction joins a term only if the ligand and every receptor
    # subunit are annotated to it (Tensor-cell2cell's AND rule).
    terms = {"T1": {"A", "B", "C", "D", "E"}, "T2": {"A", "X", "Y"}}
    pathway_per_gene: dict[str, set[str]] = {}
    for term, genes in terms.items():
        for g in genes:
            pathway_per_gene.setdefault(g, set()).add(term)

    lr_list = ["A^B", "A^X", "D^E", "A^B&C", "X^Y", "A^B&Z"]
    assert generate_lr_geneset(lr_list, pathway_per_gene, min_size=1) == {
        "T1": {"A^B", "D^E", "A^B&C"},
        "T2": {"A^X", "X^Y"},
    }
    assert generate_lr_geneset(lr_list, pathway_per_gene, min_size=3) == {
        "T1": {"A^B", "D^E", "A^B&C"}
    }


def test_lr_gsea_reports_enrichment_only():
    lrs = [f"L{i}^R{i}" for i in range(200)]
    loadings = pd.DataFrame({1: np.linspace(1, 0, 200)}, index=lrs)
    lr_set = {"top": set(lrs[:20]), "bottom": set(lrs[-20:])}

    res = lr_gsea(loadings, lr_set, permutations=99).set_index("term")

    # The depleted "bottom" set is dropped from the output...
    assert list(res.index) == ["top"]
    assert res.loc["top", "nes"] > 0
    # ...with the zero p-value floored at 1 / (permutations + 1)...
    assert res.loc["top", "pval"] == 0.01
    # ...but still counted in BH: 0.01 * 2 tests / rank 1.
    assert res.loc["top", "padj"] == pytest.approx(0.02)


def test_load_pathway_per_gene_rejects_unknown_library():
    with pytest.raises(ValueError, match="Unknown library"):
        load_pathway_per_gene(cast(Any, "GOMF"))
