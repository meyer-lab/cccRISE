import io
import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.sparse import csr_array, issparse

DATA_DIR = Path("/opt/andrew/ccc")


# The below code is taken directly from https://github.com/earmingol/cell2cell/blob/master/cell2cell/datasets/anndata.py
def import_balf_covid(
    filename=str(DATA_DIR / "BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"),
    gene_threshold: float = 0.01,
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
    url = "https://zenodo.org/record/7535867/files/BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print("Downloading data from Zenodo...")
        urllib.request.urlretrieve(url, filename)

    print("Loading BALF COVID data (preserving sparse format)...")
    adata = anndata.read_h5ad(filename)

    assert hasattr(adata.X, "nnz"), "adata.X should be a sparse matrix"

    return prepare_dataset(
        adata,
        condition_name="sample",
        geneThreshold=gene_threshold,
        normalize=normalize,
    )


@lru_cache(maxsize=1)
def import_ligand_receptor_pairs(
    filename=str(DATA_DIR / "Human-2020-Jin-LR-pairs.csv.zst"),
    update_interaction_names: bool = True,
) -> pd.DataFrame:
    """Import ligand-receptor pairs from a zstd-compressed CSV with caching."""
    print("Loading ligand-receptor pairs data (zstd compressed)...")
    with open(filename, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            df = pd.read_csv(text_stream)

    print(f"Cached {len(df)} ligand-receptor pairs")

    if update_interaction_names:
        if "interaction_name_2" in df.columns:
            df["ligand"] = df["interaction_name_2"].apply(
                lambda x: x.split(" - ")[0].upper()
            )
            df["receptor"] = df["interaction_name_2"].apply(
                lambda x: (
                    x.split(" - ")[1]
                    .upper()
                    .replace("(", "")
                    .replace(")", "")
                    .replace("+", "&")
                )
            )
        # Also update interaction_symbol if present
        if "interaction_symbol" in df.columns:
            df["interaction_symbol"] = (
                df["interaction_symbol"].str.upper().str.replace("_", "&")
            )

    return df


def import_alad(gene_threshold: float = 0.1, normalize: bool = True):
    """Generate data for ALAD analysis, filtering and scaling as needed.
    patient ids: dsco_id
    """
    # Load data without backing to avoid issues with sparse conversion
    data = anndata.read_h5ad("/opt/BAL-scRNAseq-raw.h5ad")

    # Convert to sparse format if not already sparse
    if not issparse(data.X):
        # Convert dense matrix to sparse CSR format
        from scipy.sparse import csr_matrix

        data.X = csr_matrix(data.X)

    data = prepare_dataset(
        data, "dsco_id", geneThreshold=gene_threshold, normalize=normalize
    )

    return data


def import_alad_brush(
    filename="/opt/alad.brush.sc.h5ad",
    gene_threshold: float = 0.01,
    normalize: bool = True,
) -> anndata.AnnData:
    """Independent ALAD airway brushing dataset for validation.
    patient ids: subj
    """
    print("Loading ALAD brushing data...")
    data = anndata.read_h5ad(filename)

    if not issparse(data.X):
        from scipy.sparse import csr_matrix

        data.X = csr_matrix(data.X)

    return prepare_dataset(
        data, "subj", geneThreshold=gene_threshold, normalize=normalize
    )


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
