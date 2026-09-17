"""Cross-method benchmarking of CCC-RISE against published CCC methods.

This package exists because a benchmark table needs *one* definition of each
number it reports, shared by every method in it. The comparators themselves live
in `cccrise.comparators`; what lives here is the scoring that turns any of their
output into a directly comparable figure.

Two axes, following the plan in `.deferred/BENCHMARK_PLAN.md`:

* **tensor axis** -- methods that fit a multi-condition model and expose a
  per-condition score matrix (CCC-RISE, Tensor-cell2cell, STACCato, the
  pseudobulk CP baseline). Scored by ALAD-status AUC (`auc`).
* **single-cell axis** -- methods that emit a per-cell-pair score and have no
  condition factor at all (Scriabin, NICHES). Scored by concordance against
  CCC-RISE's reconstructed per-cell-pair score, and by an AUC computed from a
  stated aggregation of those scores to the sample level.

The contract every method on the tensor axis satisfies is deliberately narrow: it
reduces to a `(n_conditions, n_components)` score matrix, and `auc.cv_auc` never
learns which method produced one. A method that needs special handling to look
good is a method whose advantage is in the handling.
"""
