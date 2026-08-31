import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import vcsc
from scipy.sparse import csr_array, issparse

DATA_DIR = Path(__file__).parent / "data"
LR_PAIRS_PATH = DATA_DIR / "Human-2020-Jin-LR-pairs.parquet"


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
        adata = vcsc.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
        adata = add_cond_idxs(adata, "sample")
        adata.var["means"] = adata.X.mean(axis=0)
        return adata

    adata = vcsc.load_packed(filename).to_anndata()
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
        data = vcsc.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
    else:
        data = vcsc.load_packed(filename).to_anndata()

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
        data = vcsc.load_and_normalize(
            filename, min_cell_counts=10.0, gene_threshold=gene_threshold
        )
        data = add_cond_idxs(data, "subj")
        data.var["means"] = data.X.mean(axis=0)
        return data

    data = vcsc.load_packed(filename).to_anndata()
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


# --- IBDverse -----------------------------------------------------------------
# Alegbe, Harris et al. Nature 2026 (multi-tissue release). See
# https://www.ibdverse.info/#data. The distributed h5ad files are written with a
# Blosc/Zstd filter, so `hdf5plugin` must be imported before h5py opens them.

IBD_DATA_DIR = Path("/home/ehurt/ProjectData/IBD")
IBD_TI_PATH = IBD_DATA_DIR / "ti_only.h5ad"


def load_ibdverse_sample_metadata(
    metadata_dir: str | Path = IBD_DATA_DIR,
) -> pd.DataFrame:
    """Join the three IBDverse metadata TSVs into one per-sample table.

    Returns a frame indexed by ``sanger_sample_id`` carrying ``individual_id``,
    ``biopsy_type``, ``disease_status``, ``sex``, ``age_binned_10``,
    ``ti_ses_inflamed`` and a derived three-level ``inflammation_status``
    (``Healthy`` / ``CD_uninflamed`` / ``CD_inflamed``).

    Note that ``ti_ses_inflamed`` is an *individual*-level SES-CD score for the
    terminal ileum, so a blood sample labelled ``inflamed`` means the donor's
    ileum was inflamed, not the blood.
    """
    metadata_dir = Path(metadata_dir)
    samples = pd.read_csv(metadata_dir / "sample_metadata.tsv", sep="\t")
    mapping = pd.read_csv(metadata_dir / "sample_individual_mapping.tsv", sep="\t")
    individuals = pd.read_csv(metadata_dir / "individual_metadata.tsv", sep="\t")

    meta = samples.merge(mapping, on="sanger_sample_id", how="left").merge(
        individuals, on="individual_id", how="left"
    )

    status = pd.Series("Healthy", index=meta.index, dtype=object)
    is_cd = meta["disease_status"] == "CD"
    status[is_cd & (meta["ti_ses_inflamed"] == "uninflamed")] = "CD_uninflamed"
    status[is_cd & (meta["ti_ses_inflamed"] == "inflamed")] = "CD_inflamed"
    meta["inflammation_status"] = status

    return meta.drop_duplicates("sanger_sample_id").set_index("sanger_sample_id")


def _read_h5ad_obs(path: str | Path) -> pd.DataFrame:
    """Read only ``obs`` from an h5ad, without touching X."""
    import h5py
    import hdf5plugin  # noqa: F401  (registers the Blosc/Zstd filters)

    with h5py.File(path, "r") as handle:
        return anndata.io.read_elem(handle["obs"])


def _coalesce_ranges(
    starts: np.ndarray, stops: np.ndarray, max_gap: int = 1_000_000
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Group sorted [start, stop) ranges into slabs separated by more than `max_gap`.

    Returns ``(slab_start, slab_stop, member_ranges)`` per slab.
    """
    slabs: list[tuple[int, int, list[tuple[int, int]]]] = []
    slab_start, slab_stop = int(starts[0]), int(stops[0])
    members = [(int(starts[0]), int(stops[0]))]
    for start, stop in zip(starts[1:], stops[1:]):
        start, stop = int(start), int(stop)
        if start - slab_stop > max_gap:
            slabs.append((slab_start, slab_stop, members))
            slab_start, slab_stop, members = start, stop, []
        slab_stop = max(slab_stop, stop)
        members.append((start, stop))
    slabs.append((slab_start, slab_stop, members))
    return slabs


def _read_h5ad_rows(path: str | Path, rows: np.ndarray) -> anndata.AnnData:
    """Read a subset of cells from a CSR-backed h5ad without loading all of X.

    `anndata`'s backed mode still round-trips through its own indexing machinery;
    for a file with billions of stored values it is much cheaper to gather the
    requested rows straight out of the CSR arrays.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    rows = np.sort(np.asarray(rows, dtype=np.int64))

    with h5py.File(path, "r") as handle:
        obs = anndata.io.read_elem(handle["obs"]).iloc[rows]
        var = anndata.io.read_elem(handle["var"])

        group = handle["X"]
        indptr = group["indptr"][:]
        starts, stops = indptr[rows], indptr[rows + 1]
        counts = stops - starts

        data = np.empty(int(counts.sum()), dtype=group["data"].dtype)
        indices = np.empty(data.size, dtype=np.int32)
        at = 0
        # One h5py read per requested row would mean tens of thousands of
        # round-trips through the decompression filter. Coalesce rows that land
        # close together into a single slab read and slice locally instead; the
        # gap tolerance bounds how much extra we pull off disk per slab.
        for slab_start, slab_stop, members in _coalesce_ranges(starts, stops):
            slab_data = group["data"][slab_start:slab_stop]
            slab_indices = group["indices"][slab_start:slab_stop]
            for start, stop in members:
                width = int(stop - start)
                if width == 0:
                    continue
                lo = int(start - slab_start)
                data[at : at + width] = slab_data[lo : lo + width]
                indices[at : at + width] = slab_indices[lo : lo + width]
                at += width

    matrix = csr_array(
        (data, indices, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)),
        shape=(rows.size, var.shape[0]),
    )
    return anndata.AnnData(X=matrix, obs=obs, var=var)


def _select_ibdverse_cells(
    obs: pd.DataFrame,
    condition_name: str,
    n_conditions: int | None,
    max_cells_per_condition: int | None,
    min_conf_score: float | None,
    stratify_by: str | None,
    random_state: int,
) -> np.ndarray:
    """Pick the cell indices for a (possibly downsampled) IBDverse run."""
    rng = np.random.default_rng(random_state)
    keep = np.ones(len(obs), dtype=bool)
    if min_conf_score is not None:
        keep &= obs["conf_score"].to_numpy() >= min_conf_score

    conditions = obs.loc[keep, condition_name].astype(str)

    if n_conditions is not None and n_conditions < conditions.nunique():
        per_condition = obs.loc[keep].drop_duplicates(condition_name)
        if stratify_by is not None and stratify_by in per_condition:
            groups = per_condition.groupby(stratify_by, observed=True)[condition_name]
            # Round-robin across strata so small strata are not squeezed out.
            pools = [
                list(rng.permutation(np.asarray(names, dtype=object)))
                for _, names in groups
            ]
            chosen: list[str] = []
            while len(chosen) < n_conditions and any(pools):
                for pool in pools:
                    if pool and len(chosen) < n_conditions:
                        chosen.append(str(pool.pop()))
        else:
            names = per_condition[condition_name].astype(str).to_numpy()
            chosen = list(rng.choice(names, size=n_conditions, replace=False))
        keep &= obs[condition_name].astype(str).isin(set(chosen)).to_numpy()

    rows = np.flatnonzero(keep)
    if max_cells_per_condition is not None:
        codes = obs[condition_name].astype(str).to_numpy()[rows]
        selected = []
        for name in pd.unique(codes):
            block = rows[codes == name]
            if block.size > max_cells_per_condition:
                block = rng.choice(block, size=max_cells_per_condition, replace=False)
            selected.append(block)
        rows = np.sort(np.concatenate(selected))

    return rows


def import_ibdverse(
    filename: str | Path = IBD_TI_PATH,
    metadata_dir: str | Path = IBD_DATA_DIR,
    gene_threshold: float = 0.001,
    normalize: bool = True,
    condition_name: str = "sanger_sample_id",
    n_conditions: int | None = None,
    max_cells_per_condition: int | None = None,
    min_conf_score: float | None = None,
    stratify_by: str | None = "inflammation_status",
    random_state: int = 0,
) -> anndata.AnnData:
    """IBDverse intestinal biopsies, optionally downsampled.

    The full terminal-ileum object is 2.17M cells x 33,538 genes with 4.19e9
    stored values, roughly 30x more conditions and 34x more cells than any
    dataset CCC-RISE has been fit on, so `n_conditions` / `max_cells_per_condition`
    exist to draw a tractable subset without materializing all of X.

    Sample-level phenotype is not in the distributed `obs`; it is joined here
    from the metadata TSVs (see `load_ibdverse_sample_metadata`). `obs` also
    carries only unnamed prediction clusters (`Epithelial_17`, `T_20`, ...), so
    a coarse `lineage` column is derived from their prefix.

    Beware that outside the terminal ileum, tissue is completely separated from
    disease status in this release: every blood sample is from a CD donor and
    every rectal sample from a healthy one. Pool tissues as conditions and the
    leading components are uninterpretable.

    References:
    [1] Alegbe, Harris et al. Nature (2026). https://doi.org/10.1038/s41586-026-10627-z
    """
    obs = _read_h5ad_obs(filename)
    meta = load_ibdverse_sample_metadata(metadata_dir)
    obs = obs.join(meta, on=condition_name)

    rows = _select_ibdverse_cells(
        obs,
        condition_name=condition_name,
        n_conditions=n_conditions,
        max_cells_per_condition=max_cells_per_condition,
        min_conf_score=min_conf_score,
        stratify_by=stratify_by,
        random_state=random_state,
    )
    print(
        f"Loading IBDverse: {rows.size:,} cells across "
        f"{obs.iloc[rows][condition_name].nunique():,} samples..."
    )

    data = _read_h5ad_rows(filename, rows)
    data.obs = obs.iloc[rows]

    if "predicted_labels" in data.obs:
        data.obs["lineage"] = (
            data.obs["predicted_labels"].astype(str).str.split("_").str[0]
        ).astype("category")

    return prepare_dataset(
        data, condition_name, geneThreshold=gene_threshold, normalize=normalize
    )
