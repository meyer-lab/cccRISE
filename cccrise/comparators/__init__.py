"""The comparators behind the manuscript's figures (main, supplementary and
reviewer): only the code that made their data.

The label-based comparators (pseudobulk CPD, Tensor-cell2cell, STACCato) are
built and fit by ``methods``, ``tensors`` and ``staccato``; ``staccato`` is a
Python re-implementation of the authors' R code, not a wrapper around it, and
is not guaranteed to be numerically identical to it. ``scores`` compares their
fits with cccRISE's (Figure 5).

The single-cell comparators run the authors' own R code, sourced from pinned
clones rather than installed: NICHES through ``niches_system.R``
(``runners/run_niches.py``, scored by ``niches_programs``), and Scriabin
through the scripts in ``scriabin/`` (its README has the R setup), whose
per-sample programs ``scriabin_programs`` merges as upstream does
(``runners/run_scriabin_seeds.py``).
"""
