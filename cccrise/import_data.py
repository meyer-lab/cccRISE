import zlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import vsparse
from scipy.sparse import csr_array, issparse

from .gene_filters import mito_ribo_mask

DATA_DIR = Path(__file__).parent / "data"
LR_PAIRS_PATH = DATA_DIR / "Human-2020-Jin-LR-pairs.parquet"

IBD_DATA_DIR = Path("/opt/IVCSC")
IBD_FULL_COHORT_IVCSC = IBD_DATA_DIR / "full_IBD_cohort.ivcsc.h5ad"

# Depth-normalize to the median, scale by gene sum, then log10(1 + 1000x)
IBD_NORMALIZATION = vsparse.RECIPES["parafac2"]


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


#: ALAD1-HS63 and ALAD1-HS65 each contributed two dsco_id samples; every other
#: dsco_id is its own patient. Hardcoded rather than parsed out of dsco_id,
#: since these are the only known pairs -- a regex would silently "find"
#: pairs in any future dsco_id that happens to share a prefix.
ALAD_PATIENT_OF: dict[str, str] = {
    "ALAD1-HS63-E1": "HS63",
    "ALAD1-HS63-E2": "HS63",
    "ALAD1-HS65-E1": "HS65",
    "ALAD1-HS65-E2": "HS65",
}


def add_alad_patient_id(data: anndata.AnnData) -> anndata.AnnData:
    """Add `patient_id`, pairing dsco_id's two known multi-sample patients.

    Every dsco_id not in `ALAD_PATIENT_OF` is treated as its own patient (its
    `patient_id` is just its `dsco_id`), so `patient_id` is safe to pass as
    `validation.component_stability`'s `patient_column` unconditionally: a
    singleton dsco_id groups to itself and so never gets paired.
    """
    dsco_id = data.obs["dsco_id"].astype(str)
    data.obs["patient_id"] = dsco_id.map(ALAD_PATIENT_OF).fillna(dsco_id)
    return data


def import_alad(
    filename: str | Path = "/opt/IVCSC/BAL-scRNAseq.raw.h5ad",
    gene_threshold: float = 0.001,
    normalize: bool = True,
):
    """Generate data for ALAD analysis, filtering and scaling as needed.
    patient ids: dsco_id (patient_id pairs its two known multi-sample
    patients; see `ALAD_PATIENT_OF`)
    """
    if normalize:
        data = vsparse.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
    else:
        data = vsparse.load_packed(filename).to_anndata()

    data = add_alad_patient_id(data)

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


def cap_cells_per_condition(
    conditions: pd.Series | np.ndarray,
    max_cells: int,
    seed: int = 0,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Mask keeping at most `max_cells` eligible cells of each condition.

    A condition with no more than `max_cells` eligible cells keeps all of them;
    a larger one keeps a uniform random subset. Each condition's draw is seeded
    by `seed` and the condition's own name, never by its position, so a
    condition's cells do not depend on which other conditions are present. The
    kept cells are a prefix of one fixed permutation, so for the same `seed`
    the cells kept under a smaller cap are a subset of those kept under a
    larger one.

    Returns a boolean mask over all cells; ineligible cells are always False.
    """
    if max_cells < 1:
        raise ValueError(f"max_cells must be at least 1; got {max_cells}.")
    labels = np.asarray(conditions).astype(str)
    eligible = (
        np.ones(labels.size, dtype=bool) if eligible is None else np.asarray(eligible)
    )
    keep = np.zeros(labels.size, dtype=bool)

    codes, names = pd.factorize(labels)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(names) + 1))
    for code, name in enumerate(names):
        members = order[bounds[code] : bounds[code + 1]]
        members = members[eligible[members]]
        if members.size <= max_cells:
            keep[members] = True
            continue
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        keep[members[rng.permutation(members.size)[:max_cells]]] = True
    return keep


def import_ibdverse(
    filename: str | Path = IBD_FULL_COHORT_IVCSC,
    gene_threshold: float = 0.001,
    normalize: bool = True,
    condition_name: str = "sanger_sample_id",
    max_cells_per_condition: int | None = None,
    cell_cap_seed: int = 0,
) -> anndata.AnnData:
    """IBDverse terminal-ileum biopsies: 343 samples, 232 healthy and 111 Crohn's.

    `max_cells_per_condition` caps each sample at that many cells, drawn with
    `cap_cells_per_condition` from the cells passing the count filter. The gene
    filter is measured on the full, uncapped cohort, so capped and uncapped
    loads share one gene set; normalization runs on the capped cells.

    References:
    [1] Krzak, Alegbe, Taylor, Jones et al. Nat Genet (2026).
        https://doi.org/10.1038/s41588-026-02634-7
    """
    print("Loading IBDverse data...")
    packed = vsparse.load_packed(filename)
    counts = cast(Any, packed.X)
    var = cast(pd.DataFrame, packed.var)

    excluded = mito_ribo_mask(var)

    # Thresholds read the raw counts, before anything is dropped, so the
    # surviving set does not depend on the order the masks are applied.
    n_cells = counts.shape[0]
    keep_cells = np.asarray(counts.sum(axis=1)).reshape(-1) > 10.0
    keep_genes = np.asarray(counts.sum(axis=0)).reshape(-1) > gene_threshold * n_cells
    keep_genes &= ~excluded
    if max_cells_per_condition is not None:
        keep_cells = cap_cells_per_condition(
            cast(pd.DataFrame, packed.obs)[condition_name],
            max_cells_per_condition,
            seed=cell_cap_seed,
            eligible=keep_cells,
        )
    kept = packed[keep_cells, keep_genes]
    del packed, counts

    if not normalize:
        data = use_gene_symbols(kept.to_anndata())
        return prepare_dataset(
            data, condition_name, geneThreshold=gene_threshold, normalize=normalize
        )

    # `to_scipy_sparse` leaves the centering term out of `X` so the matrix stays
    # sparse; `means` is what `condition_matrices` subtracts once a condition
    # block is densified.
    view = kept.normalized(IBD_NORMALIZATION)
    X = view.to_scipy_sparse(dtype=np.float32)
    # vsparse hands back its int64 index arrays whatever the size. Once the
    # nonzeros fit int32 (e.g. under a cell cap), narrowing them saves 4 bytes
    # per nonzero here and in every row copy BiCV takes, and is what MKL's
    # default LP64 interface requires.
    if X.nnz <= np.iinfo(np.int32).max:
        X = csr_array(
            (X.data, X.indices.astype(np.int32), X.indptr.astype(np.int32)),
            shape=X.shape,
        )
    data = anndata.AnnData(
        X=X,
        obs=cast(pd.DataFrame, kept.obs).copy(),
        var=cast(pd.DataFrame, kept.var).copy(),
    )
    data.var["means"] = np.asarray(view.means, dtype=np.float64)
    data = use_gene_symbols(data)
    return add_cond_idxs(data, condition_name)


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


def iter_condition_matrices(X: anndata.AnnData) -> Iterator[np.ndarray]:
    """Yield `X.X` one condition at a time, as mean-centered dense matrices.

    Centers by `X.var["means"]` (as set by `prepare_dataset`) if present,
    matching the centering `parafac2_nd` applies internally, so that
    projecting these matrices with RISE/PARAFAC2 factors is consistent with
    the fit. Only one condition is dense at a time, so a single pass costs
    one condition's worth of memory rather than the whole dataset's.
    """
    assert X.X is not None
    X_mat = cast(Any, X.X)
    sg_index = X.obs["condition_unique_idxs"].to_numpy().astype(int)
    means = X.var["means"].to_numpy() if "means" in X.var else np.zeros(X_mat.shape[1])

    for i in range(int(np.amax(sg_index)) + 1):
        slice_mat = X_mat[sg_index == i]
        dense = slice_mat.toarray() if issparse(slice_mat) else np.asarray(slice_mat)
        yield dense - means


def condition_matrices(X: anndata.AnnData) -> list[np.ndarray]:
    """Every condition of `iter_condition_matrices` at once.

    Holds the whole dataset dense; prefer `iter_condition_matrices` for a
    single pass over a large cohort.
    """
    return list(iter_condition_matrices(X))


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
