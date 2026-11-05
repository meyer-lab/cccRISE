"""R-vs-Python equivalence of the vendored Scriabin comparator.

Runs `rcheck/compare.py`'s stages against the upstream R outputs in
`rcheck/out/check_<dataset>` (written by `rcheck/run_upstream.R`), and
`rcheck/wgcna_compare.py` against `rcheck/out/wgcna` (`rcheck/wgcna_check.R`,
upstream and non-default module-discovery settings). These need the isolated
R library in `rcheck/lib` (see `rcheck/README.md`), so each check is skipped
when its R outputs are absent, e.g. in CI.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rcheck" / "out"
STAGES = {
    "programs": "r_merged.json",
    "significance": "r_significance.csv",
    "scores": "r_controls.json",
}


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"rcheck_{name}", ROOT / "rcheck" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset", ["alad", "ibd"])
@pytest.mark.parametrize("stage", list(STAGES))
def test_matches_upstream_r(dataset, stage):
    check_dir = OUT / f"check_{dataset}"
    if not (check_dir / STAGES[stage]).exists():
        pytest.skip(f"no R outputs for stage {stage!r} in {check_dir}")
    compare = _module("compare")
    compare.RESULTS.clear()
    getattr(compare, stage)(check_dir)
    failed = [r for r in compare.RESULTS if not r["ok"]]
    assert compare.RESULTS and not failed, failed


def test_wgcna_matches_upstream_r_across_settings():
    d = OUT / "wgcna"
    if not list(d.glob("r_mods_*.csv")):
        pytest.skip(f"no R outputs in {d}")
    wgcna = _module("wgcna_compare")
    wgcna.RESULTS.clear()
    wgcna.compare(d)
    failed = [r for r in wgcna.RESULTS if not r["ok"]]
    assert wgcna.RESULTS and not failed, failed
