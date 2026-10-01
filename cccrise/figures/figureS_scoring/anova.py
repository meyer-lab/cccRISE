"""ANOVA association of CP components with the dataset's outcome under each
score (point estimate), in the default score's component numbering."""

import pandas as pd

from ...validation.anova_stability import align_to_reference_components
from .. import common, style
from . import _shared as S


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
    outcome = S.default_provenance("anova_point_estimate", "group_column")

    ax = fig.subplots()
    S.grouped_bar(ax, eta, r"$\eta^2$", sig=sig)
    ax.set_title(f"Association with {outcome}\n(* p < {common.SIG_ALPHA})")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"], legend=S.method_legend)
