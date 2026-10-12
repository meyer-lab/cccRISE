"""Unit tests for the labellings of `run_comparator_fits`."""

import numpy as np
import pytest

from cccrise.runners.run_comparator_fits import labelling

FINE = [
    "Enteroendocrine cells",
    "Enterochromaffin cells",
    "Goblet cells",
    "Tuft cells",
    "ILCs",
    "T cells",
]
COARSE = ["Secretory"] * 4 + ["T cell"] * 2
CACHE = {"fine_types": np.array(FINE), "fine_to_coarse": np.array(COARSE)}


def test_lineage_with_types_broken_out():
    suffix, mapping, chosen = labelling(CACHE, "lin+ee+ec")
    assert suffix == "_lin+ee+ec"
    assert chosen == ["Enterochromaffin cells", "Enteroendocrine cells"]
    assert mapping["Enteroendocrine cells"] == "Enteroendocrine cells"
    assert mapping["Tuft cells"] == mapping["Goblet cells"] == "Secretory"
    assert mapping["ILCs"] == "T cell"


def test_fine_with_one_lineage_pooled():
    _, mapping, chosen = labelling(CACHE, "fine-sec+tuft")
    assert set(chosen) == {"Tuft cells", "ILCs", "T cells"}
    assert mapping["Enteroendocrine cells"] == mapping["Goblet cells"] == "Secretory"
    assert mapping["ILCs"] == "ILCs"
    assert len(set(labelling(CACHE, "fine-sec")[1].values())) == 3


def test_sampled_labelling_with_a_type_reverted():
    codes = {"Enteroendocrine cells": "ee", "Enterochromaffin cells": "ec",
             "Tuft cells": "tuft", "ILCs": "ilc"}  # fmt: skip
    _, _, sampled = labelling(CACHE, "s4", repeat=1)
    # 4 of 6 types: at least one of the 4 coded ones is drawn
    target = next(t for t in sampled if t in codes)
    code = codes[target]
    suffix, mapping, chosen = labelling(CACHE, f"s4r1-{code}")
    assert suffix == f"_s4r1-{code}"
    assert chosen == [t for t in sampled if t != target]
    assert mapping[target] == COARSE[FINE.index(target)]


def test_unknown_type_code_raises():
    with pytest.raises(KeyError):
        labelling(CACHE, "lin+nope")
