"""ANOVA association of CP components with the dataset's outcome under each
score (point estimate), in the default score's component numbering."""

import pandas as pd

from ....anova import get_sample_to_group, prepare_group_labels
from ....utils import correct_conditions
from ....validation.stability import align_to_reference_components
from ... import common, style
from . import _common as S


def aligned_anova(method: str) -> tuple[pd.Series, pd.Series]:
    """`method`'s eta-squared and p < alpha flags per default component."""
    df = S.load_method_output(S.SPEC.anova_point_estimate_filename(), method)
    significant = df["P"] < common.SIG_ALPHA
    if method == S.DEFAULT_SCORE:
        return df.set_index("Component")["EtaSquared"], significant.set_axis(
            df["Component"]
        )
    ref, _ = S.cp_factors(S.DEFAULT_SCORE)
    alt, _ = S.cp_factors(method)

    def align(values):
        return align_to_reference_components(ref, alt, values, S.MATCH_MODES).set_index(
            "Component"
        )["Value"]

    return align(df["EtaSquared"].to_numpy()), align(
        significant.to_numpy().astype(float)
    ) > 0.5


def plot(fig):
    aligned = {m: aligned_anova(m) for m in S.ALL_SCORES}
    eta = pd.DataFrame({m: e for m, (e, _) in aligned.items()})
    sig = pd.DataFrame({m: s for m, (_, s) in aligned.items()})
    outcome = S.default_provenance(
        S.SPEC.anova_point_estimate_filename(), "group_column"
    )
    outcome = common.OUTCOME_LABELS.get(outcome, outcome)

    ax = fig.subplots()
    S.grouped_bar(ax, eta, rf"$\eta^2$ ({outcome})", sig=sig)
    save_source_data(eta, sig)


def save_source_data(eta: pd.DataFrame, sig: pd.DataFrame) -> None:
    """The bars, each method's own ANOVA table, and the per-sample loadings
    each tested."""
    labels = {m: S.SCORE_LABELS[m] for m in S.ALL_SCORES}
    shown = (
        pd.concat({"EtaSquared": eta, "Significant": sig}, axis=1)
        .rename_axis("Component")
        .stack(future_stack=True)
        .rename_axis(["Component", "Method"])
        .reset_index()
        .replace({"Method": labels})
    )
    common.source_data(
        "anova_outcome_by_score",
        shown,
        "One row per (default-score CPD component, score): the eta-squared of "
        "the score's paired component (the bars); Significant marks raw "
        f"P < {common.SIG_ALPHA} (asterisks).",
    )
    filename = S.SPEC.anova_point_estimate_filename()
    tables = pd.concat(
        [S.load_method_output(filename, m).assign(Method=labels[m]) for m in labels]
    )
    columns = ["Method", "Component", "NGroups", "GroupSizes", "F", "EtaSquared", "P"]
    common.source_data(
        "anova_outcome_by_score_components",
        tables[columns],
        "One row per (score, CPD component in that score's own numbering): its "
        "one-way ANOVA against the outcome. P is raw (uncorrected).",
    )
    common.source_data(
        "loading_condition_by_score",
        pd.concat([sample_loadings(m).assign(Method=labels[m]) for m in labels]),
        "One row per (score, BAL sample, CPD component in that score's own "
        "numbering): the read-depth-corrected condition loading the ANOVA "
        "tested, and the sample's outcome group.",
    )


def sample_loadings(method: str) -> pd.DataFrame:
    """`method`'s read-depth-corrected condition loadings per sample (long),
    with the outcome group its ANOVA tested them across."""
    filename = S.SPEC.anova_point_estimate_filename()
    adata = S.method_adata(method)
    groups = prepare_group_labels(
        get_sample_to_group(
            adata,
            S.SPEC.condition_column,
            S.default_provenance(filename, "group_column"),
        ),
        control_label=S.default_provenance(filename, "control_label"),
        other_label=S.default_provenance(filename, "other_label"),
    )
    loadings = correct_conditions(adata)
    return (
        pd.DataFrame(
            loadings, index=groups.index, columns=range(1, loadings.shape[1] + 1)
        )
        .assign(Group=groups.to_numpy())
        .rename_axis("Sample")
        .reset_index()
        .melt(id_vars=["Sample", "Group"], var_name="Component", value_name="Loading")
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"], legend=S.method_legend)
