"""
Bundle the analysis results for public deposit (e.g. Zenodo), alongside the
EMBO source data: the canonical and alternate-score factor fits and every
runner output, each with its provenance sidecar, including the files too large
for git or the source data zips (`significance_entries_<dataset>.nc`), plus a
README listing every file with its size and sha256.

    uv run cccrise-deposit   # -> output/deposit/cccrise_results.zip
"""

import zipfile

from .datasets import DATA_DIR, FACTORS_DIR, RANK_SELECTION_DIR, VALIDATION_DIR
from .provenance import REPO_ROOT, git_sha, sha256_file

RESULT_DIRS = (FACTORS_DIR, RANK_SELECTION_DIR, VALIDATION_DIR)
OUTPUT_PATH = REPO_ROOT / "output" / "deposit" / "cccrise_results.zip"
CODE_URL = "https://github.com/meyer-lab/cccRISE"


def main() -> None:
    files = sorted(p for d in RESULT_DIRS for p in d.rglob("*") if p.is_file())
    lines = [
        "CCC-RISE analysis results: factor fits and runner outputs, each with a",
        "`.provenance.yaml` sidecar recording the code version, parameters, and",
        "inputs that produced it.",
        "",
        f"Bundled by `uv run cccrise-deposit` from {CODE_URL} at commit {git_sha()}.",
        "",
        "file\tbytes\tsha256",
    ]
    lines += [
        f"{p.relative_to(DATA_DIR)}\t{p.stat().st_size}\t{sha256_file(p)}"
        for p in files
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", "\n".join(lines) + "\n")
        for path in files:
            archive.write(path, path.relative_to(DATA_DIR))
    print(
        f"{OUTPUT_PATH}: {len(files)} files, {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB"
    )
