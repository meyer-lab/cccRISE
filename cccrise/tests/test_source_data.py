"""
Tests for EMBO source data: panels saving what they plot while a figure is
built (`figures.common.source_data`) and its packaging into panel folders
(`figures.source_data`).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cccrise.figures import common
from cccrise.figures import source_data as packager


def _panel(name: str, plot, **attrs) -> SimpleNamespace:
    """A stand-in panel module named `name`, drawing with `plot`."""
    doc = f"{name} docstring.\n\nMore detail."
    return SimpleNamespace(__name__=f"fake.{name}", __doc__=doc, plot=plot, **attrs)


def _two_axes(axs):
    """Data across both its letters, and data for its second letter only."""
    frame = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    common.source_data("both", frame, "Both axes' points.")
    common.source_data("second", frame.set_index("x")["y"], "Second only.", part=1)


def _all_nan(axs):
    common.source_data("bad", pd.DataFrame({"x": [1], "y": [np.nan]}), "Broken.")


@pytest.fixture
def source_dir(tmp_path, monkeypatch):
    """Save source data under `tmp_path` as `cccrise-fbuild` would."""
    common.style.apply_style()
    monkeypatch.setitem(common._SOURCE, "dir", tmp_path / "source")
    monkeypatch.setitem(common._SOURCE, "cleared", set())
    source = tmp_path / "source"
    source.mkdir()
    (source / common.SOURCE_INPUTS).write_text("[]")  # as a full build records
    return source


def test_source_data_is_a_no_op_outside_a_build(monkeypatch):
    monkeypatch.setitem(common._SOURCE, "dir", None)
    common.source_data("x", pd.DataFrame({"a": [1]}), "Nothing written.")


def test_panels_are_packaged_by_letter(source_dir, tmp_path):
    panels = {
        "two": _panel("two", _two_axes, AXES=("square", "square")),
        "schematic": _panel(
            "schematic",
            lambda axs: common.placeholder(axs[0], "S"),
            AXES=("square",),
            SCHEMATIC=True,
        ),
        "nan": _panel("nan", _all_nan, AXES=("square",)),
    }
    fig = common.compose_fixed([list(panels.values())])
    letters = common.panel_letters(fig)
    assert letters == {"two": ["a", "b"], "schematic": ["c"], "nan": ["d"]}
    saved = pd.read_csv(source_dir / "two" / "second.csv")
    assert list(saved.columns) == ["x", "y"]  # the index kept as a column

    out = tmp_path / "Figure 9"
    rows, errors = packager.stage_figure(
        "Figure 9", "9", letters, panels, source_dir, out
    )
    assert {r["Panel"]: r["Status"] for r in rows} == {
        "9A-B": "data",
        "9B": "data",
        "9C": "schematic",
        "9D": "data",
    }
    assert errors == ["Figure 9 9D: bad.csv has all-NaN columns ['y']"]
    assert sorted(p.name for p in (out / "9A-B").iterdir()) == [
        "README.txt",
        "both.csv",
    ]
    readme = (out / "9B" / "README.txt").read_text()
    assert "two docstring." in readme and "Second only." in readme
    assert "Columns: x, y" in readme
    assert "9C" in (out / "README.txt").read_text()  # listed as schematic


def test_subfigure_panels_and_missing_data(source_dir, tmp_path):
    def plot(fig):
        fig.subplots()
        common.source_data("values", pd.DataFrame({"a": [1]}), "Values.")

    panels = {
        "with_data": _panel("with_data", plot),
        "without": _panel("without", lambda fig: fig.subplots()),
    }
    fig = common.compose((4, 2), [["with_data", "without"]], panels)

    rows, errors = packager.stage_figure(
        "Figure 9", "9", common.panel_letters(fig), panels, source_dir, tmp_path / "f"
    )
    assert [r["Panel"] for r in rows] == ["9A", "9B"]
    assert errors == ["Figure 9 without: no source data"]

    (source_dir / common.SOURCE_INPUTS).unlink()  # only a panel was built
    _, errors = packager.stage_figure(
        "Figure 9", "9", common.panel_letters(fig), panels, source_dir, tmp_path / "f"
    )
    assert "Figure 9: no record of its inputs; build it in full" in errors


def test_span():
    for letters, span in ((["b"], "B"), (["b", "c", "d"], "B-D"), (["b", "d"], "B,D")):
        assert packager.span(letters) == span
        assert packager.letters_of(span) == [x.upper() for x in letters]


def test_checklist_has_one_row_per_main_figure_letter():
    manifest = pd.DataFrame(
        {
            "Figure": ["Figure 1", "Figure 2", "Figure 2", "Figure 2", "Figure EV1"],
            "Panel": ["1A-B", "2A-B", "2B", "2C", "EV1A"],
            "Status": ["schematic", "data", "data", "missing", "data"],
        }
    )
    rows = packager.checklist(manifest)
    assert list(rows["Panel"]) == ["A", "B", "A", "B", "C"]  # no EV figures
    assert list(rows["Provided?"]) == [0, 0, 1, 1, 0]
    assert rows["Source Data Category"].iloc[0] == "N/A (schematics)"
    assert rows["Source Data Category"].iloc[-1] == ""


def test_supplements_are_labeled_by_class():
    labels = {f: (label, prefix) for f, label, prefix in packager.figures()}
    assert labels["2"] == ("Figure 2", "2")
    assert labels["S_alad_cohort"] == ("Figure EV1", "EV1")
    assert labels["S_rise_factor_matrices"] == ("Appendix Figure S1", "S1")
    assert "S_imputation_ridge" not in labels  # skipped
