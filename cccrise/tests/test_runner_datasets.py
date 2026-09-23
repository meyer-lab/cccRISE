from cccrise.runners._cli import resolve_anova_labels
from cccrise.runners.datasets import DATA_DIR, DATASETS, scoring_output_dir
from cccrise.runners.run_alt_comm_score import DEFAULT_STABILITY_AXES, build_parser
from cccrise.validation.component_stability import DEFAULT_AXES


def test_scoring_output_dir_default_score_uses_data_dir():
    assert scoring_output_dir("expression_product") == DATA_DIR


def test_scoring_output_dir_alt_score_uses_subdirectory():
    assert (
        scoring_output_dir("expression_hill")
        == DATA_DIR / "alt_score_methods" / "expression_hill"
    )


def test_alt_comm_score_default_axes_are_accepted_by_component_stability():
    assert set(DEFAULT_STABILITY_AXES) <= set(DEFAULT_AXES)


def test_alt_comm_score_parser_defaults_to_compressed_cp():
    args = build_parser().parse_args(["--dataset", "alad", "--reference-source", "x"])
    assert args.compress and args.backend is None


def test_resolve_anova_labels_prefers_outcome_column_only_by_default():
    spec = DATASETS["alad"]
    assert resolve_anova_labels(spec, None, None, None) == (
        spec.outcome_column,
        None,
        None,
    )
    assert resolve_anova_labels(spec, "ALADstatus", None, None)[0] == "ALADstatus"
