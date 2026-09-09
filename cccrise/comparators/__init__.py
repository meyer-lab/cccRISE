"""Python re-implementations of comparator methods used to benchmark cccRISE.

Each submodule (``staccato``, ``scriabin``, ``niches``) is an independent
re-implementation, built to reproduce one method's published algorithm
closely enough for a fair comparison against cccRISE without needing that
method's original runtime (e.g. an R installation). They are NOT wrappers
around the original packages, and are not guaranteed to be numerically
identical to them -- see each submodule's docstring for the specific source
it was reimplemented from, any known differences, and how its output should
be compared against cccRISE.

Scriabin and NICHES are re-implemented against this project's own shared LR
table (``import_data.import_ligand_receptor_pairs``, the same one cccRISE
uses) rather than either method's bundled LR database, since the point of
these comparators is to evaluate the algorithmic approach, not database
choice.
"""
