"""Raw ALAD UMI counts for the faithful Scriabin run (`preprocess.R`): every
cell and gene of the source file, unnormalized, as a genes x cells CSC
matrix in three little-endian binary buffers R reads with ``readBin``
(``indices`` int32, ``indptr`` int64, ``data`` int32), plus gene names and
the cell metadata the authors' preprocessing uses (sample, cell type,
percent.mt, percent.ribo).

    uv run python rcheck/faithful/export_counts.py rcheck/out/faithful/input
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cccrise.import_data import import_alad
from cccrise.runners.datasets import DATASETS


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spec = DATASETS["alad"]
    adata = import_alad(normalize=False)
    X = adata.X.tocsr()  # cells x genes CSR == genes x cells CSC
    X.sort_indices()
    assert np.all(X.data == np.round(X.data)), "expected integer UMI counts"
    X.indices.astype("<i4").tofile(out / "indices.bin")
    X.indptr.astype("<i8").tofile(out / "indptr.bin")
    X.data.astype("<i4").tofile(out / "data.bin")
    pd.Series(adata.var_names).to_csv(out / "genes.csv", index=False, header=False)
    pd.DataFrame(
        {
            "cell": adata.obs_names,
            "sample": adata.obs[spec.condition_column].astype(str).to_numpy(),
            "celltype": adata.obs[spec.celltype_column].astype(str).to_numpy(),
            "percent.mt": adata.obs["percent.mt"].to_numpy(float),
            "percent.ribo": adata.obs["percent.ribo"].to_numpy(float),
        }
    ).to_csv(out / "meta.csv", index=False)
    print(f"{adata.n_obs} cells x {adata.n_vars} genes, {X.nnz} nonzeros")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
