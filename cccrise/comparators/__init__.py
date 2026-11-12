"""Comparator methods used to benchmark cccRISE.

The label-based comparators (pseudobulk CPD, Tensor-cell2cell, STACCato) are
built and fit by ``methods``, ``tensors`` and ``staccato``; ``staccato`` is a
Python re-implementation of the authors' R code, not a wrapper around it, and
is not guaranteed to be numerically identical to it. ``scores`` compares their
fits with cccRISE's (Figure 5).

The single-cell comparators run the upstream R packages: NICHES through
``niches_system.R`` (``runners/run_niches.py``, scored by ``niches_programs``)
and Scriabin from R runs read by ``runners/run_scriabin_seeds.py`` (pooled by
``scriabin_programs``). ``niches`` and ``scriabin`` are earlier Python
re-implementations, not used by the manuscript and kept for their tests; both
use this project's shared LR table (``import_data.import_ligand_receptor_pairs``)
rather than either method's bundled database.
"""
