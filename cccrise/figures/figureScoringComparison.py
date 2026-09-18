"""
Figure comparing CCC-RISE results across alternate communication-scoring
methods (issue #143 / PR #157): are the manuscript's conclusions dependent on
the exact ligand x receptor combination rule used to build the 4D tensor?

Compares the default score (``expression_product``) against the three
alternates (``expression_mean``, CellPhoneDB-style; ``expression_gmean``;
``expression_hill``, CellChat-style) on ALAD, at the already-selected ranks
(RISE 19, CP 3), across four analyses:

    1. CPD factor matrices -- component match quality, aligned factor cosine
       similarity, top ligand-receptor pairs, top sender/receiver cell types.
    2. Permutation significance of ligand-receptor pairs.
    3. ANOVA association between CP components and ALADstatus (point
       estimates only -- no bootstrap; see the caveats below).
    4. Bootstrap CP component stability (placeholder scale -- see caveats).

This figure only loads results; it never generates them. All four methods
compared here -- including the default -- are read from
``DATA_DIR/alt_score_methods/<method>/``, a comparison-scope copy generated
by ``cccrise-altcommscore`` at identical ranks and settings for all four, so
the comparison is apples-to-apples. This is deliberately separate from the
default's comprehensive, canonical results directly in ``DATA_DIR`` (produced
by ``cccrise-run`` / ``cccrise-significance`` / ``cccrise-stability``, at
whatever scale and ranks those were last run at), which every other figure
and the manuscript's interpretation reads instead. Regenerate all four with:

    uv run cccrise-altcommscore --dataset alad --rise-rank 19 --cp-rank 3 \
        --communication-score expression_product --n-perms 3 --n-replicates 3
    uv run cccrise-altcommscore --dataset alad --rise-rank 19 --cp-rank 3 \
        --communication-score expression_mean --n-perms 3 --n-replicates 3
    uv run cccrise-altcommscore --dataset alad --rise-rank 19 --cp-rank 3 \
        --communication-score expression_gmean --n-perms 3 --n-replicates 3
    uv run cccrise-altcommscore --dataset alad --rise-rank 19 --cp-rank 3 \
        --communication-score expression_hill --n-perms 3 --n-replicates 3

Known caveats in this comparison:

* The ANOVA point estimates are exactly that -- a single fit per method, no
  patient-resampling bootstrap -- so they show whether components associate
  with ALADstatus at all under each score, not whether that association
  would survive a different patient roster.
* Every method's significance/stability results here were generated at
  smoke-test scale (3 permutations / 3 bootstrap replicates), to validate
  this figure's plumbing cheaply, not to be a statistically powered result.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..factors import load_factors_netcdf, load_factors_to_anndata
from ..runners.datasets import DATASETS, comparison_output_dir
from ..stability.anova_stability import align_to_reference_components, top_lr_pairs
from .common import getSetup, load_runner_csv, subplotLabel

DATASET_KEY = "alad"
SPEC = DATASETS[DATASET_KEY]

RISE_RANK = 19
CP_RANK = 3

DEFAULT_SCORE = "expression_product"
ALT_SCORES = ["expression_mean", "expression_gmean", "expression_hill"]
ALL_SCORES = [DEFAULT_SCORE, *ALT_SCORES]
SCORE_LABELS = {
    "expression_product": "product (default)",
    "expression_mean": "mean (CellPhoneDB-style)",
    "expression_gmean": "geometric mean",
    "expression_hill": "Hill (CellChat-style)",
}

# The three alternates' significance/stability results were generated at this
# scale; see module docstring.
N_PERMS_SMOKE = 3
N_REPLICATES_SMOKE = 3

TOP_N_LR = 10
TOP_N_CELLTYPE = 3

MATCH_MODE_WEIGHTS = {3: 1.0}  # match CP components on the LR mode only

ALT_COMM_SCORE_HINT = (
    "    uv run cccrise-altcommscore --dataset {dataset} --rise-rank {rise_rank} "
    "--cp-rank {cp_rank} --communication-score {method} "
    "--n-perms 3 --n-replicates 3"
).format(dataset=DATASET_KEY, rise_rank=RISE_RANK, cp_rank=CP_RANK, method="<method>")


# =============================================================================
# Result loading
# =============================================================================


def _cp_factors(method: str) -> list[np.ndarray]:
    ds = load_factors_netcdf(comparison_output_dir(method) / SPEC.factors_filename)
    return [ds[f"cp_{mode}"].values for mode in "ABCD"]


def _lr_symbols(method: str) -> np.ndarray:
    ds = load_factors_netcdf(comparison_output_dir(method) / SPEC.factors_filename)
    return ds.coords["lr_pair"].values


def _celltype_loadings(method: str, adata_raw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean sender (sc_B) and receiver (rc_C) loading per cell type, per component.

    RISE itself does not depend on `communication_score` (it is fit on raw
    expression before the interaction tensor exists), and every method here
    is fit with the same rise_rank/random_state, so the RISE eigen-states --
    and therefore the sender/receiver coordinate system CP's B/C factors sit
    in -- are identical across methods. Aligning CP components on the LR mode
    (`align_to_reference_components`) is enough to make B/C directly
    comparable; no separate RISE-basis alignment is needed.
    """
    ad = load_factors_to_anndata(
        comparison_output_dir(method) / SPEC.factors_filename,
        adata=adata_raw,
        condition_column=SPEC.condition_column,
    )
    celltypes = ad.obs[SPEC.celltype_column].to_numpy()
    sender = pd.DataFrame(ad.obsm["sc_B"], index=celltypes).groupby(level=0).mean()
    receiver = pd.DataFrame(ad.obsm["rc_C"], index=celltypes).groupby(level=0).mean()
    return sender, receiver


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _match_and_compare(reference_method: str, method: str, adata_raw) -> pd.DataFrame:
    """Per reference component: CP match score, aligned B/C/D cosine, top-LR
    Jaccard, top sender/receiver cell-type Jaccard against `reference_method`."""
    ref_factors = _cp_factors(reference_method)
    alt_factors = _cp_factors(method)

    matching = align_to_reference_components(
        ref_factors, alt_factors, np.arange(1, CP_RANK + 1), MATCH_MODE_WEIGHTS
    )

    lr_symbols = _lr_symbols(reference_method)
    ref_top_lr = top_lr_pairs(ref_factors[3], lr_symbols, top_n=TOP_N_LR)
    alt_top_lr = top_lr_pairs(alt_factors[3], lr_symbols, top_n=TOP_N_LR)

    ref_sender, ref_receiver = _celltype_loadings(reference_method, adata_raw)
    alt_sender, alt_receiver = _celltype_loadings(method, adata_raw)

    rows = []
    for _, row in matching.iterrows():
        ref_c, alt_c, sign = (
            int(row["Component"]),
            int(row["MatchedComponent"]),
            np.sign(row["SignedCosine"]) or 1.0,
        )
        b_cos = float(
            np.dot(ref_factors[1][:, ref_c - 1], sign * alt_factors[1][:, alt_c - 1])
            / (
                np.linalg.norm(ref_factors[1][:, ref_c - 1])
                * np.linalg.norm(alt_factors[1][:, alt_c - 1])
            )
        )
        c_cos = float(
            np.dot(ref_factors[2][:, ref_c - 1], sign * alt_factors[2][:, alt_c - 1])
            / (
                np.linalg.norm(ref_factors[2][:, ref_c - 1])
                * np.linalg.norm(alt_factors[2][:, alt_c - 1])
            )
        )

        ref_lr_set = set(ref_top_lr.loc[ref_top_lr["Component"] == ref_c, "LRPair"])
        alt_lr_set = set(alt_top_lr.loc[alt_top_lr["Component"] == alt_c, "LRPair"])

        def top_celltypes(df: pd.DataFrame, component: int) -> set:
            col = df.iloc[:, component - 1]
            return set(
                col.abs().sort_values(ascending=False).head(TOP_N_CELLTYPE).index
            )

        sender_jaccard = _jaccard(
            top_celltypes(ref_sender, ref_c), top_celltypes(alt_sender, alt_c)
        )
        receiver_jaccard = _jaccard(
            top_celltypes(ref_receiver, ref_c), top_celltypes(alt_receiver, alt_c)
        )

        rows.append(
            {
                "Component": ref_c,
                "MatchScore": float(row["MatchScore"]),
                "MeanBCCosine": float(np.mean([b_cos, c_cos])),
                "TopLRJaccard": _jaccard(ref_lr_set, alt_lr_set),
                "CelltypeJaccard": float(np.mean([sender_jaccard, receiver_jaccard])),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================


def _grouped_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    sig: pd.DataFrame | None = None,
) -> None:
    n_groups, n_methods = data.shape
    width = 0.8 / n_methods
    x = np.arange(n_groups)
    for i, method in enumerate(data.columns):
        ax.bar(x + i * width, data[method], width, label=SCORE_LABELS[method])
        if sig is not None:
            for xi, (val, is_sig) in enumerate(
                zip(data[method], sig[method], strict=True)
            ):
                if is_sig:
                    ax.text(
                        x[xi] + i * width,
                        val,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(data.index)
    ax.set_xlabel("CP component (reference numbering)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=5, loc="best")


def makeFigure():
    adata_raw = SPEC.load()

    # --- CPD factor comparison (a-d) -----------------------------------
    comparisons = {
        method: _match_and_compare(DEFAULT_SCORE, method, adata_raw)
        for method in ALT_SCORES
    }
    match_score = pd.DataFrame(
        {m: comparisons[m].set_index("Component")["MatchScore"] for m in ALT_SCORES}
    )
    bc_cosine = pd.DataFrame(
        {m: comparisons[m].set_index("Component")["MeanBCCosine"] for m in ALT_SCORES}
    )
    lr_jaccard = pd.DataFrame(
        {m: comparisons[m].set_index("Component")["TopLRJaccard"] for m in ALT_SCORES}
    )
    celltype_jaccard = pd.DataFrame(
        {
            m: comparisons[m].set_index("Component")["CelltypeJaccard"]
            for m in ALT_SCORES
        }
    )

    # --- ANOVA point estimates (e) --------------------------------------
    ref_anova_factors = _cp_factors(DEFAULT_SCORE)
    anova_eta = {}
    anova_sig = {}
    for method in ALL_SCORES:
        df = load_runner_csv(
            SPEC.anova_point_estimate_filename(),
            root=comparison_output_dir(method),
            hint=ALT_COMM_SCORE_HINT.replace("<method>", method),
        )
        if method == DEFAULT_SCORE:
            eta = df.set_index("Component")["EtaSquared"]
            sig = df.set_index("Component")["P"] < 0.05
        else:
            method_factors = _cp_factors(method)
            matched = align_to_reference_components(
                ref_anova_factors,
                method_factors,
                df["EtaSquared"].to_numpy(),
                MATCH_MODE_WEIGHTS,
            ).set_index("Component")
            eta = matched["Value"]
            sig_matched = align_to_reference_components(
                ref_anova_factors,
                method_factors,
                (df["P"] < 0.05).to_numpy().astype(float),
                MATCH_MODE_WEIGHTS,
            ).set_index("Component")
            sig = sig_matched["Value"] > 0.5
        anova_eta[method] = eta
        anova_sig[method] = sig
    anova_eta_df = pd.DataFrame(anova_eta)
    anova_eta_df.index.name = "Component"
    anova_sig_df = pd.DataFrame(anova_sig)

    # --- Permutation significance (f) -----------------------------------
    frac_sig = {}
    top_sig_lr = {}
    for method in ALL_SCORES:
        sig_df = load_runner_csv(
            SPEC.significance_lr_filename(),
            root=comparison_output_dir(method),
            hint=ALT_COMM_SCORE_HINT.replace("<method>", method),
        )
        frac_sig[method] = float(sig_df["FractionSignificant_p05"].mean())
        top_sig_lr[method] = set(sig_df.nlargest(TOP_N_LR, "MeanZScore")["LRPair"])
    sig_jaccard = {
        method: _jaccard(top_sig_lr[DEFAULT_SCORE], top_sig_lr[method])
        for method in ALL_SCORES
    }

    # --- Bootstrap component stability (g) -------------------------------
    stability_fms = {}
    for method in ALL_SCORES:
        df = load_runner_csv(
            SPEC.stability_filename(),
            root=comparison_output_dir(method),
            hint=ALT_COMM_SCORE_HINT.replace("<method>", method),
        )
        cpd = df[df["Stage"] == "CPD"]
        stability_fms[method] = cpd.groupby("Axis")["FMS"].median()
    stability_df = pd.DataFrame(stability_fms)

    # --- Draw ------------------------------------------------------------
    axs, fig = getSetup((16, 8), (2, 4))

    _grouped_bar(
        axs[0],
        match_score,
        "CP component match quality (LR-mode cosine)",
        "Match score",
    )
    _grouped_bar(
        axs[1],
        bc_cosine,
        "Aligned sender/receiver (B/C) cosine similarity",
        "Cosine similarity",
    )
    _grouped_bar(
        axs[2], lr_jaccard, f"Top-{TOP_N_LR} LR-pair Jaccard overlap", "Jaccard index"
    )
    _grouped_bar(
        axs[3],
        celltype_jaccard,
        f"Top-{TOP_N_CELLTYPE} sender/receiver cell-type Jaccard",
        "Jaccard index",
    )

    _grouped_bar(
        axs[4],
        anova_eta_df,
        "ANOVA eta-squared vs ALADstatus (point estimate, * p<0.05)",
        "Eta-squared",
        sig=anova_sig_df,
    )

    bars = axs[5].bar(
        [SCORE_LABELS[m] for m in ALL_SCORES],
        [frac_sig[m] for m in ALL_SCORES],
        color="#3a6b9c",
    )
    for bar, method in zip(bars, ALL_SCORES, strict=True):
        axs[5].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"J={sig_jaccard[method]:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    axs[5].set_ylabel("Mean fraction of entries\nsignificant per LR pair (p<0.05)")
    axs[5].set_title(
        f"Permutation significance (label: top-{TOP_N_LR} LR-pair\n"
        "Jaccard vs. default)",
        fontsize=8,
    )
    axs[5].tick_params(axis="x", labelrotation=25)

    x = np.arange(len(stability_df.index))
    width = 0.8 / len(ALL_SCORES)
    for i, method in enumerate(ALL_SCORES):
        axs[6].bar(
            x + i * width, stability_df[method], width, label=SCORE_LABELS[method]
        )
    axs[6].set_xticks(x + width * (len(ALL_SCORES) - 1) / 2)
    axs[6].set_xticklabels(stability_df.index)
    axs[6].set_ylabel("Median CP component FMS")
    axs[6].set_title(
        f"CP component stability (n={N_REPLICATES_SMOKE} placeholder)", fontsize=8
    )
    axs[6].legend(fontsize=5, loc="best")

    axs[7].axis("off")
    axs[7].text(
        0.0,
        1.0,
        (
            f"Dataset: {SPEC.label}\n"
            f"RISE rank {RISE_RANK}, CP rank {CP_RANK}\n"
            f"All 4 methods: n_perms={N_PERMS_SMOKE}, "
            f"n_replicates={N_REPLICATES_SMOKE} (smoke test)\n\n"
            "Caveats:\n"
            "- ANOVA is a point estimate, not a bootstrap.\n"
            "- All results here are smoke-test scale, not\n"
            "  statistically powered -- a separate, comprehensive\n"
            "  copy of the default score's results lives in the\n"
            "  root data dir for manuscript interpretation."
        ),
        transform=axs[7].transAxes,
        va="top",
        ha="left",
        fontsize=6,
        family="monospace",
    )

    subplotLabel(axs)
    return fig
