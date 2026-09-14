import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import vsparse
from scipy.sparse import csr_array, issparse

# Temporary imports until upstream APIs support gene exclusion
from ._vsparse_filter import load_and_normalize_excluding
from .gene_filters import describe_mask, mito_ribo_mask

DATA_DIR = Path(__file__).parent / "data"
LR_PAIRS_PATH = DATA_DIR / "Human-2020-Jin-LR-pairs.parquet"

IBD_DATA_DIR = Path("/opt/IVCSC")
IBD_FULL_COHORT_IVCSC = IBD_DATA_DIR / "full_IBD_cohort.ivcsc.h5ad"


# The below code is taken directly from https://github.com/earmingol/cell2cell/blob/master/cell2cell/datasets/anndata.py
def import_balf_covid(
    filename: str = str(DATA_DIR / "BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"),
    gene_threshold: float = 0.001,
    normalize: bool = True,
) -> anndata.AnnData:
    """BALF samples from COVID-19 patients
    The data consists in 63k immune and epithelial cells in lungs
    from 3 control, 3 moderate COVID-19, and 6 severe COVID-19 patients.

    This dataset was previously published in [1], and this objects contains
    the raw counts for the annotated cell types available in:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926

    References:
    [1] Liao, M., Liu, Y., Yuan, J. et al.
        Single-cell landscape of bronchoalveolar immune cells in patients
        with COVID-19. Nat Med 26, 842–844 (2020).
        https://doi.org/10.1038/s41591-020-0901-9

    Parameters
    ----------
        filename : str, default='BALF-COVID19-Liao_et_al-NatMed-2020.h5ad'
            Path to the h5ad file in case it was manually downloaded.

    Returns
    -------
        Annotated data matrix with sparse X matrix preserved.
    """
    url = "https://ucla.box.com/shared/static/lnirh5vmas4r1fc7xnr7z6ndk2s6tkhq.h5ad"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        local_file = Path("./BALF-COVID19-Liao_et_al-NatMed-2020.h5ad")
        if local_file.exists():
            import shutil

            shutil.copyfile(local_file, filename)
        else:
            print("Downloading data from Box...")
            import shutil

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(filename, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)

    print("Loading BALF COVID data (preserving sparse format)...")
    if normalize:
        adata = vsparse.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
        adata = add_cond_idxs(adata, "sample")
        adata.var["means"] = adata.X.mean(axis=0)
        return adata

    adata = vsparse.load_packed(filename).to_anndata()
    return prepare_dataset(
        adata,
        condition_name="sample",
        geneThreshold=gene_threshold,
        normalize=normalize,
    )


@lru_cache(maxsize=1)
def import_ligand_receptor_pairs(
    filename: str | Path = LR_PAIRS_PATH,
    update_interaction_names: bool = True,
) -> pd.DataFrame:
    """Import ligand-receptor pairs from a parquet file with caching."""
    df = pd.read_parquet(filename)

    if update_interaction_names:
        if "interaction_name_2" in df.columns:
            df["ligand"] = (
                df["interaction_name_2"]
                .apply(lambda x: x.split(" - ")[0].upper())
                .astype("category")
            )
            df["receptor"] = (
                df["interaction_name_2"]
                .apply(
                    lambda x: (
                        x.split(" - ")[1]
                        .upper()
                        .replace("(", "")
                        .replace(")", "")
                        .replace("+", "&")
                    )
                )
                .astype("category")
            )
        # Also update interaction_symbol if present
        if "interaction_symbol" in df.columns:
            df["interaction_symbol"] = (
                df["interaction_symbol"].str.upper().str.replace("_", "&")
            )

    return df


def import_alad(
    filename: str | Path = "/opt/IVCSC/BAL-scRNAseq.raw.h5ad",
    gene_threshold: float = 0.001,
    normalize: bool = True,
):
    """Generate data for ALAD analysis, filtering and scaling as needed.
    patient ids: dsco_id
    """
    if normalize:
        data = vsparse.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
    else:
        data = vsparse.load_packed(filename).to_anndata()

    # Map cell types directly from the metadata column revised_annotations_10_5
    if "revised_annotations_10_5" in data.obs:
        celltype_map = {
            0: "Macrophage 1",
            1: "Macrophage 2",
            2: "Monocyte",
            3: "Macrophage 3",
            4: "Macrophage 4",
            5: "Macrophage 5",
            7: "Macrophage 6",
            8: "Macrophage 7",
            9: "Macrophage",
            10: "Macrophage 8",
            11: "Macrophage Ribo-high",
            12: "CD8 T cells",
            13: "Cytokine prod.macrophage",
            15: "Cytokine prod.macrophage 2",
            16: "CD4 T cells",
            17: "Tissue resident Macrophage",
            18: "Activated Mono/Mac",
            19: "Proliferating cells",
            20: "cDC2",
            21: "NK cells",
            22: "Treg",
            23: "Lung epithelium",
            24: "Activated CD8",
            25: "Activated DCs",
            26: "Anti-Inflammatory Macrophages",
            27: "cDC1",
        }
        broad_map = {
            0: "Macrophages",
            1: "Macrophages",
            2: "Monocytes",
            3: "Macrophages",
            4: "Macrophages",
            5: "Macrophages",
            7: "Macrophages",
            8: "Macrophages",
            9: "Macrophages",
            10: "Macrophages",
            11: "Macrophages",
            12: "CD8 T cells",
            13: "Macrophages",
            15: "Macrophages",
            16: "CD4 T cells",
            17: "Macrophages",
            18: "Macrophages",
            19: "Proliferating cells",
            20: "Dendritic Cells",
            21: "NK cells",
            22: "CD4 T cells",
            23: "Epithelial cells",
            24: "CD8 T cells",
            25: "Dendritic Cells",
            26: "Macrophages",
            27: "Dendritic Cells",
        }
        data.obs["celltype"] = (
            data.obs["revised_annotations_10_5"].map(celltype_map).values
        )
        data.obs["broad_cell_type"] = (
            data.obs["revised_annotations_10_5"].map(broad_map).values
        )

    if normalize:
        data = add_cond_idxs(data, "dsco_id")
        data.var["means"] = data.X.mean(axis=0)
        return data

    return prepare_dataset(
        data, "dsco_id", geneThreshold=gene_threshold, normalize=normalize
    )


def import_alad_brush(
    filename: str | Path = "/opt/IVCSC/alad.brush.sc.h5ad",
    gene_threshold: float = 0.01,
    normalize: bool = True,
) -> anndata.AnnData:
    """Independent ALAD airway brushing dataset for validation.
    patient ids: subj
    """
    print("Loading ALAD brushing data...")
    if normalize:
        data = vsparse.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
        data = add_cond_idxs(data, "subj")
        data.var["means"] = data.X.mean(axis=0)
        return data

    data = vsparse.load_packed(filename).to_anndata()
    return prepare_dataset(
        data, "subj", geneThreshold=gene_threshold, normalize=normalize
    )

# Necessary fix for IBDverse genes
def use_gene_symbols(
    data: anndata.AnnData, symbol_column: str = "gene_symbols"
) -> anndata.AnnData:
    """Re-index an AnnData's genes by HGNC symbol, in place."""
    if symbol_column not in data.var:
        return data

    symbols = data.var[symbol_column].astype(str)
    missing = symbols.isin(["", "nan", "None"])
    data.var["ensembl_id"] = data.var_names
    data.var_names = symbols.where(~missing, data.var_names).astype(str).tolist()
    data.var_names_make_unique()
    return data


def import_ibdverse(
    filename: str | Path = IBD_FULL_COHORT_IVCSC,
    gene_threshold: float = 0.001,
    normalize: bool = True,
    condition_name: str = "sanger_sample_id",
    filter_mito_ribo: bool = True,
) -> anndata.AnnData:
    """IBDverse terminal-ileum biopsies: 343 samples, 232 healthy and 111 Crohn's.

    References:
    [1] Krzak, Alegbe, Taylor, Jones et al. Nat Genet (2026).
        https://doi.org/10.1038/s41588-026-02634-7
    """
    print("Loading IBDverse data...")
    if normalize:
        if filter_mito_ribo:
            data = load_and_normalize_excluding(
                filename,
                min_cell_counts=10.0,
                gene_threshold=gene_threshold,
                exclude_genes=_report_mito_ribo_mask,
            )
        else:
            data = vsparse.load_and_normalize(
                filename, min_cell_counts=10.0, gene_threshold=gene_threshold
            )
        data = use_gene_symbols(data)
        data = add_cond_idxs(data, condition_name)
        data.var["means"] = data.X.mean(axis=0)
        return data

    data = vsparse.load_packed(filename).to_anndata()
    if filter_mito_ribo:
        var = cast(pd.DataFrame, data.var)
        mask = mito_ribo_mask(var)
        print(f"   {describe_mask(var, mask)}")
        data = data[:, ~mask].copy()
    data = use_gene_symbols(data)
    return prepare_dataset(
        data, condition_name, geneThreshold=gene_threshold, normalize=normalize
    )


def _report_mito_ribo_mask(var: pd.DataFrame) -> np.ndarray:
    """`mito_ribo_mask`, announcing what it removed."""
    mask = mito_ribo_mask(var)
    print(f"   {describe_mask(var, mask)}")
    return mask


def add_cond_idxs(X, condition_key):
    """Add unique condition indices to an AnnData object."""
    # Create a copy to avoid modifying a view
    # X = X.copy()

    # Get unique conditions and map to indices
    conditions = np.unique(X.obs[condition_key].unique())
    condition_map = {cond: i for i, cond in enumerate(conditions)}

    # Add indices to obs
    X.obs["condition_unique_idxs"] = X.obs[condition_key].map(condition_map).values

    return X


def condition_matrices(X: anndata.AnnData) -> list[np.ndarray]:
    """Split `X.X` into per-condition, mean-centered dense matrices.

    Centers by `X.var["means"]` (as set by `prepare_dataset`) if present,
    matching the centering `parafac2_nd` applies internally, so that
    projecting these matrices with RISE/PARAFAC2 factors is consistent with
    the fit.
    """
    assert X.X is not None
    X_mat = cast(Any, X.X)
    sg_index = X.obs["condition_unique_idxs"].to_numpy().astype(int)
    means = X.var["means"].to_numpy() if "means" in X.var else np.zeros(X_mat.shape[1])

    matrices = []
    for i in range(int(np.amax(sg_index)) + 1):
        slice_mat = X_mat[sg_index == i]
        dense = slice_mat.toarray() if issparse(slice_mat) else np.asarray(slice_mat)
        matrices.append(dense - means)

    return matrices


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float, normalize=False
) -> anndata.AnnData:
    assert issparse(X.X)
    # Use raw data if available, otherwise use the current X data
    data = cast(Any, csr_array(X.raw.X if X.raw is not None else X.X))
    assert np.amin(data.data) >= 0.0

    # Filter out genes with too few reads, and cells with fewer than 10 counts
    row_mask = data.sum(axis=1) > 10
    col_mask = data.mean(axis=0) > geneThreshold

    # Do the (expensive) sparse row/column filtering ourselves, once. Some source
    # files alias X.layers[None] to the same array as X.X, which would otherwise
    # make AnnData's view/copy machinery redundantly filter and copy the same
    # large matrix twice. Swap in a cheap empty placeholder first so that step
    # only has to filter/copy the (tiny) placeholder to keep obs/var/obsm aligned.
    filtered_data = data[row_mask][:, col_mask]
    X.X = csr_array(X.shape, dtype=data.dtype)
    X = X[row_mask, col_mask]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())
    X.X = filtered_data
    data = filtered_data

    # Convert counts to floats
    if issubclass(data.dtype.type, int | np.integer):
        data.data = data.data.astype(np.float32)

    if normalize:
        ## Normalize total counts per cell
        # Keep the counts on a reasonable scale to avoid accuracy issues
        counts_per_cell = data.sum(axis=1)
        counts_per_cell /= np.median(counts_per_cell)
        # inplace csr row scale
        data.data /= np.repeat(counts_per_cell, np.diff(data.indptr))

        # Scale genes by sum, inplace csr col scale
        data.data /= data.sum(axis=0).take(data.indices, mode="clip")

        # Transform values
        data.data = np.log10((1000.0 * data.data) + 1.0)

    # Get the indices for subsetting the data
    X = add_cond_idxs(X, condition_name)

    # Pre-calculate gene means
    X.var["means"] = data.mean(axis=0)

    return X
