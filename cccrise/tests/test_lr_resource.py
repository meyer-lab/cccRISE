import pandas as pd

from cccrise.ccc import filter_ppi_by_proteins
from cccrise.import_data import import_ligand_receptor_pairs


def test_resource_is_human():
    """The shipped resource names human MHC genes, not mouse H2 genes."""
    lr = import_ligand_receptor_pairs()
    genes = set("&".join(lr["ligand"].astype(str)).split("&"))
    assert {"HLA-A", "HLA-B", "HLA-C", "HLA-DRA"} <= genes
    assert not any(g.startswith("H2-") for g in genes)


def test_partners_read_from_symbols_not_label():
    """Genes come from the structured symbol columns, so display-label tokens
    (FLT3L, CD45, CD94:NKG2A) resolve to the genes they stand for."""
    lr = import_ligand_receptor_pairs()
    pairs = set(lr["interaction_symbol"])
    assert "FLT3LG^FLT3" in pairs
    assert "LGALS9^PTPRC" in pairs
    assert "HLA-E^KLRD1&KLRC1" in pairs
    genes = set(
        "&".join(pd.concat([lr["ligand"], lr["receptor"]]).astype(str)).split("&")
    )
    assert not {"FLT3L", "CD45", "CD94:NKG2A"} & genes


def test_complex_on_both_sides_kept():
    """A complex that is a ligand in one pair and a receptor in another
    passes the filter on both sides."""
    ppi = pd.DataFrame({"A": ["ITGA4&ITGB1", "FN1"], "B": ["VCAM1", "ITGA4&ITGB1"]})
    kept = filter_ppi_by_proteins(
        ppi, ["ITGA4", "ITGB1", "VCAM1", "FN1"], complex_sep="&"
    )
    assert len(kept) == 2
