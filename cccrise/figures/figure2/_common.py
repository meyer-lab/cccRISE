"""Choices shared by Figure 2's cell type granularity panels
(`cmp_eigenstate_loadings`, `cmp_weight_scatter`, `eigenstate_scatter`): the
CP component that splits a cell type into sender and receiver
subpopulations, and the RISE eigenstates its sender and receiver loadings
draw on."""

DATASET = "alad"
GRANULARITY_COMPONENT = 5
# Its largest sender (B) and receiver (C) loadings
SENDER_EIGENSTATE, RECEIVER_EIGENSTATE = 17, 15
HIGHLIGHT_CELLTYPE = "Epithelial cells"
