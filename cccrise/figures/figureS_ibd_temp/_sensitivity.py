"""Shared loading for the IBD annotation-sensitivity drafts: each cccRISE
component's carrier fine types (`run_lr_concordance`'s cell activity), the
granularity metrics and the type isolation (`run_granularity_sampling`)."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from ...runners.run_comparator_fits import CODES
from .. import common

DATASET = "ibd"
CONTAINED = 0.85
HINT = f"    uv run python -m cccrise.runners.run_granularity_sampling --dataset {DATASET}"
SHORT_TYPES = {
    "Enteroendocrine cells": "EE",
    "Enterochromaffin cells": "EC",
    "Tuft cells": "Tuft",
    "ILCs": "ILC",
    "Mast cells": "Mast",
    "Monocytes SOD2+": "Mono SOD2+",
    "Monocytes S100A8/9+": "Mono S100A8/9+",
    "Macrophages CCL3/4+ CXCL9/10+": "Mac CCL3/4+ CXCL9/10+",
    "Macrophages CD163-hi": "Mac CD163-hi",
    "Macrophages ITGA4+": "Mac ITGA4+",
    "Enterocytes BEST4+": "Ent. BEST4+",
    "Secretory": "Secretory",
    "Myeloid": "Myeloid",
    "T cell": "T cell",
}


def load(name: str) -> pd.DataFrame:
    """One `run_granularity_sampling` table (``granularity_<name>_ibd``)."""
    return common.load_runner_output(
        f"granularity_{name}_{DATASET}.parquet", VALIDATION_DIR, hint=HINT
    )


def activity() -> pd.DataFrame:
    """cccRISE's per-fine-type activity: the share of each type's cells
    active in each component, per side."""
    table = common.load_runner_output(
        f"lr_concordance_cell_activity_{DATASET}.parquet", VALIDATION_DIR
    )
    return table[table["annotation"] == "fine"]


def carriers() -> pd.DataFrame:
    """Per cccRISE component, its most active sender and receiver fine type
    (``sender``, ``receiver``) and their active fractions."""
    act = activity()
    rows = []
    for k, part in act.groupby("component"):
        row = {"component": k}
        for side in ("sender", "receiver"):
            best = part[part["side"] == side].nlargest(1, "active_frac").iloc[0]
            row[side] = best["celltype"]
            row[f"{side}_frac"] = best["active_frac"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("component")


def labels() -> dict[int, str]:
    """``C<k> <top pair>`` per cccRISE component."""
    profiles = common.load_runner_output(
        f"lr_concordance_profiles_{DATASET}.parquet", VALIDATION_DIR
    )
    top = profiles[
        (profiles["method"] == "cccRISE")
        & (profiles["annotation"] == "coarse")
        & (profiles["lr_rank"] == 1)
    ]
    return {
        int(k): f"C{k} {pair.replace('^', '–')}"
        for k, pair in zip(top["component"], top["lr_pair"], strict=True)
    }


def describe(labelling: str) -> str:
    """A readable name for a targeted labelling suffix, e.g. ``_lin+ee+ec``
    -> ``lineage +EE +EC``."""
    head, *edits = labelling.lstrip("_").replace("-", " -").replace("+", " +").split()
    if head == "lin":
        name = "lineage"
    elif head.startswith("fine"):
        name = f"fine, {SHORT_TYPES[CODES[edits.pop(0).lstrip('-')]]} pooled"
    else:
        name = head
    return " ".join([name] + [e[0] + SHORT_TYPES[CODES[e[1:]]] for e in edits])
