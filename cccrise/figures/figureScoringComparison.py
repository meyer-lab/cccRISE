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
       estimates only -- no bootstrap; see module docstring caveat below).
    4. Bootstrap CP component stability (placeholder scale -- see caveat).

Result-file handling
---------------------
The default score's results are the ones every other figure and the
manuscript's interpretation reads, so they are never regenerated here --
whatever is already committed in ``DATA_DIR`` is loaded as-is, even where its
settings (rise/cp rank, n_perms, n_replicates) predate this comparison. The
three alternate scores' results do not exist yet; this module generates them
under ``DATA_DIR/alt_score_methods/<method>/`` the first time it runs and
reuses them on every subsequent run (`_ensure_*` functions below check the
target path before calling the runner).

Known caveats in this comparison (see PR discussion, not fixed here since
fixing them means regenerating the default's committed results, which is the
user's call, not this figure's):

* The committed default significance run used RISE rank 25, not 19 -- the
  three alternates are run at 19. LR-pair-level significance is compared
  anyway since it is not rank-sensitive at the level reported here, but this
  is an apples-to-oranges rank comparison, not apples-to-apples.
* The committed default stability run used RISE rank 25 / CP rank 4, not
  19 / 3 -- same caveat, and it is already only 3 replicates (a placeholder,
  not a real result) before this figure was ever written.
* The ANOVA point estimates are exactly that -- a single fit per method, no
  patient-resampling bootstrap -- so they show whether components associate
  with ALADstatus at all under each score, not whether that association
  would survive a different patient roster.
* Every alternate-score run here uses 3 permutations / 3 bootstrap replicates,
  a smoke-test scale to validate this figure's plumbing, not a statistically
  powered result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..factors import load_factors_netcdf, load_factors_to_anndata
from ..import_data import import_ligand_receptor_pairs
from ..runners._cli import resolve_group_labels
from ..runners.datasets import DATASETS, scoring_output_dir
from ..runners.run_ccc_rise import run_pipeline as run_ccc_rise_pipeline
from ..runners.run_component_stability import (
    run_component_stability as run_stability_pipeline,
)
from ..runners.run_significance import run_significance as run_significance_pipeline
from ..stability.anova_stability import (
    align_to_reference_components,
    fit_anova_pipeline,
    top_lr_pairs,
)
from .common import getSetup, subplotLabel

DATASET_KEY = "alad"
SPEC = DATASETS[DATASET_KEY]

RISE_RANK = 19
CP_RANK = 3
RANDOM_STATE = 42

DEFAULT_SCORE = "expression_product"
ALT_SCORES = ["expression_mean", "expression_gmean", "expression_hill"]
ALL_SCORES = [DEFAULT_SCORE, *ALT_SCORES]
SCORE_LABELS = {
    "expression_product": "product (default)",
    "expression_mean": "mean (CellPhoneDB-style)",
    "expression_gmean": "geometric mean",
    "expression_hill": "Hill (CellChat-style)",
}

# Smoke-test scale for the three alternate scores; see module docstring.
N_PERMS_SMOKE = 3
N_REPLICATES_SMOKE = 3
STABILITY_AXES = ("cells", "lr_pairs", "permuted")

TOP_N_LR = 10
TOP_N_CELLTYPE = 3

MATCH_MODE_WEIGHTS = {3: 1.0}  # match CP components on the LR mode only


# =============================================================================
# Result-file generation / loading
# =============================================================================


def _ensure_factors(method: str) -> Path:
    path = scoring_output_dir(method) / SPEC.factors_filename
    if not path.exists():
        print(f"[ScoringComparison] Generating CPD factors for {method!r}...")
        run_ccc_rise_pipeline(
            SPEC,
            rise_rank=RISE_RANK,
            cp_rank=CP_RANK,
            random_state=RANDOM_STATE,
            communication_score=method,
            output_path=path,
        )
    return path


def _ensure_significance(method: str) -> Path:
    root = scoring_output_dir(method)
    path = root / SPEC.significance_lr_filename()
    if not path.exists():
        print(f"[ScoringComparison] Generating significance test for {method!r}...")
        run_significance_pipeline(
            SPEC,
            rise_rank=RISE_RANK,
            n_perms=N_PERMS_SMOKE,
            communication_score=method,
            random_state=RANDOM_STATE,
            output_root=root,
        )
    return path


def _anova_point_estimate_path(method: str) -> Path:
    return scoring_output_dir(method) / f"anova_point_estimate_{DATASET_KEY}.csv"


def _ensure_anova_point_estimate(method: str) -> pd.DataFrame:
    path = _anova_point_estimate_path(method)
    if path.exists():
        return pd.read_csv(path)

    print(f"[ScoringComparison] Fitting ANOVA point estimate for {method!r}...")
    group_column, control_label, other_label = resolve_group_labels(
        SPEC, None, None, None
    )
    adata = SPEC.load()
    lr_pairs = import_ligand_receptor_pairs()
    reference = fit_anova_pipeline(
        adata,
        condition_column=SPEC.condition_column,
        group_column=group_column,
        control_label=control_label,
        other_label=other_label,
        rise_rank=RISE_RANK,
        cp_rank=CP_RANK,
        lr_pairs=lr_pairs,
        communication_score=method,
        random_state=RANDOM_STATE,
    )
    df = reference["anova"].copy()
    labels = top_lr_pairs(reference["factors"][3], reference["lr_symbols"], top_n=1)
    df = df.merge(
        labels[["Component", "LRPair"]].rename(columns={"LRPair": "TopLRPair"}),
        on="Component",
        how="left",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _ensure_stability(method: str) -> Path:
    path = scoring_output_dir(method) / SPEC.stability_filename()
    if not path.exists():
        print(f"[ScoringComparison] Generating component stability for {method!r}...")
        run_stability_pipeline(
            SPEC,
            rise_rank=RISE_RANK,
            cp_rank=CP_RANK,
            axes=STABILITY_AXES,
            n_replicates=N_REPLICATES_SMOKE,
            communication_score=method,
            random_state=0,
            output_path=path,
        )
    return path


# =============================================================================
# CPD factor comparison (analysis 1)
# =============================================================================


def _cp_factors(method: str) -> list[np.ndarray]:
    ds = load_factors_netcdf(_ensure_factors(method))
    return [ds[f"cp_{mode}"].values for mode in "ABCD"]


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
        _ensure_factors(method), adata=adata_raw, condition_column=SPEC.condition_column
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

    lr_pairs_ds = load_factors_netcdf(_ensure_factors(reference_method))
    lr_symbols = lr_pairs_ds.coords["lr_pair"].values
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


def _heatmap(
    ax: plt.Axes, data: pd.DataFrame, title: str, vmin: float = 0.0, vmax: float = 1.0
) -> None:
    im = ax.imshow(data.to_numpy(), vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=0)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=6)
    ax.set_xlabel("CP component (reference numbering)")
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


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
    ).T
    bc_cosine = pd.DataFrame(
        {m: comparisons[m].set_index("Component")["MeanBCCosine"] for m in ALT_SCORES}
    ).T
    lr_jaccard = pd.DataFrame(
        {m: comparisons[m].set_index("Component")["TopLRJaccard"] for m in ALT_SCORES}
    ).T
    celltype_jaccard = pd.DataFrame(
        {
            m: comparisons[m].set_index("Component")["CelltypeJaccard"]
            for m in ALT_SCORES
        }
    ).T

    # --- ANOVA point estimates (e) --------------------------------------
    ref_anova_factors = _cp_factors(DEFAULT_SCORE)
    anova_eta = {}
    anova_sig = {}
    for method in ALL_SCORES:
        df = _ensure_anova_point_estimate(method)
        if method == DEFAULT_SCORE:
            eta = df.set_index("Component")["EtaSquared"]
            sig = df.set_index("Component")["P"] < 0.05
        else:
            method_factors = [
                load_factors_netcdf(_ensure_factors(method))[f"cp_{mode}"].values
                for mode in "ABCD"
            ]
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
        sig_df = pd.read_csv(_ensure_significance(method))
        frac_sig[method] = float(sig_df["FractionSignificant_p05"].mean())
        top_sig_lr[method] = set(sig_df.nlargest(TOP_N_LR, "MeanZScore")["LRPair"])
    sig_jaccard = {
        method: _jaccard(top_sig_lr[DEFAULT_SCORE], top_sig_lr[method])
        for method in ALL_SCORES
    }

    # --- Bootstrap component stability (g) -------------------------------
    stability_fms = {}
    for method in ALL_SCORES:
        df = pd.read_csv(_ensure_stability(method))
        cpd = df[df["Stage"] == "CPD"]
        stability_fms[method] = cpd.groupby("Axis")["FMS"].median()
    stability_df = pd.DataFrame(stability_fms)

    # --- Draw ------------------------------------------------------------
    axs, fig = getSetup((16, 8), (2, 4))

    _heatmap(axs[0], match_score, "CP component match quality (LR-mode cosine)")
    _heatmap(axs[1], bc_cosine, "Aligned sender/receiver (B/C) cosine similarity")
    _heatmap(axs[2], lr_jaccard, f"Top-{TOP_N_LR} LR-pair Jaccard overlap")
    _heatmap(
        axs[3],
        celltype_jaccard,
        f"Top-{TOP_N_CELLTYPE} sender/receiver cell-type Jaccard",
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
            f"Alternate scores: n_perms={N_PERMS_SMOKE}, "
            f"n_replicates={N_REPLICATES_SMOKE} (smoke test)\n\n"
            "Caveats:\n"
            "- Default significance/stability results predate this\n"
            "  comparison and were run at different RISE/CP ranks.\n"
            "- ANOVA is a point estimate, not a bootstrap.\n"
            "- Alt-score results are smoke-test scale, not\n"
            "  statistically powered."
        ),
        transform=axs[7].transAxes,
        va="top",
        ha="left",
        fontsize=6,
        family="monospace",
    )

    subplotLabel(axs)
    return fig
