"""
Unit tests for `run_alt_comm_score._consolidate_provenance`, which merges one
scoring method's four individual provenance sidecars into a single
provenance.yaml per `alt_score_methods/<method>/` directory.
"""

import yaml

from cccrise.runners.provenance import write_metadata
from cccrise.runners.run_alt_comm_score import _consolidate_provenance


def test_consolidate_provenance_hoists_shared_and_keeps_unique(tmp_path):
    factors_path = tmp_path / "factors.nc"
    significance_path = tmp_path / "significance_lr.csv"
    stability_path = tmp_path / "stability.csv"
    anova_path = tmp_path / "anova.csv"

    write_metadata(
        factors_path,
        {
            "dataset": "alad",
            "communication_score": "expression_mean",
            "hill_k": 0.5,
            "r2x_pf2": 0.0355,
            "only_in_factors": "x",
        },
    )
    write_metadata(
        significance_path,
        {
            "dataset": "alad",
            "communication_score": "expression_mean",
            "hill_k": 0.5,
            "n_perms": 3,
            "r2x_rise": 0.0355,
        },
    )
    write_metadata(
        anova_path,
        {
            "dataset": "alad",
            "reference_r2x_cp": 0.39,
        },
    )
    write_metadata(
        stability_path,
        {
            "dataset": "alad",
            "communication_score": "expression_mean",
            "hill_k": 0.9,  # deliberately conflicting -- must NOT be hoisted
            "n_replicates": 3,
        },
    )

    out_path = _consolidate_provenance(
        tmp_path,
        {
            "factors": factors_path,
            "significance": significance_path,
            "anova_point_estimate": anova_path,
            "stability": stability_path,
        },
    )
    assert out_path == tmp_path / "provenance.yaml"

    with open(out_path) as handle:
        merged = yaml.safe_load(handle)

    # Present, identical, in >1 section -> hoisted once at the top level.
    assert merged["dataset"] == "alad"
    assert merged["communication_score"] == "expression_mean"

    # Present in >1 section but with conflicting values -> NOT hoisted,
    # each section keeps its own value.
    assert "hill_k" not in merged
    assert merged["factors"]["hill_k"] == 0.5
    assert merged["significance"]["hill_k"] == 0.5
    assert merged["stability"]["hill_k"] == 0.9

    # Present in only one section -> stays nested, not hoisted.
    assert "only_in_factors" not in merged
    assert merged["factors"]["only_in_factors"] == "x"
    assert merged["anova_point_estimate"]["reference_r2x_cp"] == 0.39
    assert merged["significance"]["n_perms"] == 3
    assert merged["stability"]["n_replicates"] == 3

    # Each individual sidecar's own trailer is dropped in favor of one
    # consolidated trailer.
    assert "git_sha" not in merged["factors"]
    assert "generated" not in merged["stability"]
    assert "git_sha" in merged
    assert "generated" in merged
