from cccrise.runners.datasets import DATA_DIR, scoring_output_dir


def test_scoring_output_dir_default_score_uses_data_dir():
    assert scoring_output_dir("expression_product") == DATA_DIR


def test_scoring_output_dir_alt_score_uses_subdirectory():
    assert (
        scoring_output_dir("expression_hill")
        == DATA_DIR / "alt_score_methods" / "expression_hill"
    )
