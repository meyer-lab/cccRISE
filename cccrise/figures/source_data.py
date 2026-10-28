"""
EMBO Press source data for the manuscript's figures, packaged from what each
panel saved with `common.source_data` during its figure's last build:

    uv run cccrise-sourcedata                    # every figure in FIGURES
    uv run cccrise-sourcedata 2 S_cmp_stability  # some of them

Each figure becomes `output/source_data/<label>.zip` (e.g. `Figure 2.zip`),
one folder per panel (`2A/`, or `2B-C/` for files spanning a panel module's
letters) holding its CSVs and a README, plus a README of the figure's
provenance. Letters come from `_panels.yml`, as the figure's last full build
recorded them. Schematic panels (`SCHEMATIC = True`) need no data, and a
figure of only those gets no zip. Every panel is listed in the manifest, the
basis of the EMBO Source Data Checklist. Exits non-zero, writing no zips, if
any panel's data is missing or malformed.
"""

import importlib
import inspect
import itertools
import shutil
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from ..runners.datasets import DATASETS
from ..runners.provenance import REPO_ROOT, git_sha, relpath, sidecar_path
from . import common

MAIN_FIGURES = ("1", "2", "3")
QUARTO_PATH = REPO_ROOT / "manuscript" / "_quarto.yml"
SKIPPED_SUPPLEMENTS = {"imputation_ridge"}  # no source data provided
# The manuscript's label for supplements. MSB has no "Supplementary" figures:
# up to five are "Figure EV<n>", the rest "Appendix Figure S<n>". Which is
# which isn't settled yet, so the manuscript's current label stays.
SUPPLEMENT_LABEL = "Supplementary Figure S"
OUTPUT_DIR = common.OUTPUT_DIR / "source_data"
MANIFEST_PATH = REPO_ROOT / "docs" / "source_data_manifest.md"
# Every figure here plots this dataset's canonical fit
DATASET = "alad"
CODE_URL = "https://github.com/meyer-lab/cccRISE"


def figures() -> list[tuple[str, str, str]]:
    """(cccrise-fbuild name, label, panel prefix) of each figure packaged:
    the main figures, then the manuscript's supplements, numbered in order."""
    supplements = yaml.safe_load(QUARTO_PATH.read_text())["supplements"]
    return [(f, f"Figure {f}", f) for f in MAIN_FIGURES] + [
        (f"S_{name}", f"{SUPPLEMENT_LABEL}{i}", f"S{i}")
        for i, name in enumerate(supplements, start=1)
        if name not in SKIPPED_SUPPLEMENTS
    ]


def main() -> None:
    wanted = sys.argv[1:]
    chosen = [f for f in figures() if not wanted or f[0] in wanted]
    rows, errors = stage_figures(chosen)
    manifest = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)
    if not wanted:
        MANIFEST_PATH.write_text(manifest_markdown(manifest))
    print(manifest[["Figure", "Panel", "Module", "Status", "Size (KB)"]].to_string())
    if errors:
        sys.exit("Source data not packaged:\n  " + "\n  ".join(errors))
    for _, label, _ in chosen:
        zip_figure(OUTPUT_DIR / label)


def zip_figure(folder: Path) -> None:
    """Zip a staged figure's panel folders, unless it had only schematics."""
    if folder.is_dir():
        archive = shutil.make_archive(str(folder), "zip", root_dir=folder)
        print(f"{archive}: {Path(archive).stat().st_size / 1e6:.1f} MB")


def stage_figures(chosen: list[tuple[str, str, str]]) -> tuple[list[dict], list[str]]:
    """`stage_figure` for each of `figures()` chosen: the manifest rows and
    errors of all of them."""
    letters = yaml.safe_load(common.PANELS_PATH.read_text())
    rows, errors = [], []
    for figure, label, prefix in chosen:
        figure_letters = letters.get(f"fig{figure}") or {}
        if not figure_letters:
            errors.append(f"{label}: no panel letters; build it in full first")
        package = common.figure_package(figure)
        figure_rows, figure_errors = stage_figure(
            label,
            prefix,
            figure_letters,
            {n: importlib.import_module(f"{package}.{n}") for n in figure_letters},
            common.OUTPUT_DIR / f"figure{figure}" / common.SOURCE_DATA_SUBDIR,
            OUTPUT_DIR / label,
        )
        rows += figure_rows
        errors += figure_errors
    return rows, errors


def stage_figure(
    label: str,
    prefix: str,
    letters: dict[str, list[str]],
    panels: dict,
    source_dir: Path,
    out_dir: Path,
) -> tuple[list[dict], list[str]]:
    """Copy each panel module's files from `source_dir` into `out_dir`'s panel
    folders with their READMEs, given each module's `letters` and module in
    `panels`. Returns the figure's manifest rows and any errors."""
    shutil.rmtree(out_dir, ignore_errors=True)
    Path(f"{out_dir}.zip").unlink(missing_ok=True)
    rows, errors = [], []
    for name, module in panels.items():
        files = sorted((source_dir / name).glob("*.csv"))
        if files:
            folders = panel_folders(files, prefix, letters[name])
            for folder, entries in sorted(folders.items()):
                errors += [f"{label} {folder}: {e}" for e in check(entries)]
                stage_panel(out_dir / folder, label, module, entries)
                rows.append(row(label, folder, name, module, "data", entries))
            continue
        status = "schematic" if getattr(module, "SCHEMATIC", False) else "missing"
        if status == "missing":
            errors.append(f"{label} {name}: no source data")
        rows.append(row(label, prefix + span(letters[name]), name, module, status))
    if any(r["Status"] == "data" for r in rows):
        (out_dir / "README.txt").write_text(figure_readme(label, source_dir, rows))
    return rows, errors


def panel_folders(
    files: list[Path], prefix: str, letters: list[str]
) -> dict[str, list[tuple[Path, dict]]]:
    """A panel module's files, with their `common.source_data` metadata, by
    the folder their `part` puts them in."""
    folders: dict[str, list[tuple[Path, dict]]] = {}
    for path in files:
        meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
        part = meta["part"]
        folder = prefix + span(letters if part is None else [letters[part]])
        folders.setdefault(folder, []).append((path, meta))
    return folders


def stage_panel(folder: Path, label: str, module, entries) -> None:
    folder.mkdir(parents=True)
    for path, _ in entries:
        shutil.copy(path, folder)
    (folder / "README.txt").write_text(
        panel_readme(label, folder.name, module, entries)
    )


def span(letters: list[str]) -> str:
    """'B', 'B-D' for consecutive letters, else 'B,D'."""
    up = [letter.upper() for letter in letters]
    consecutive = all(ord(b) - ord(a) == 1 for a, b in itertools.pairwise(up))
    if len(up) > 1 and consecutive:
        return f"{up[0]}-{up[-1]}"
    return ",".join(up)


def check(entries: list[tuple[Path, dict]]) -> list[str]:
    """Problems with a panel's files: empty, or with a column of only NaN."""
    problems = []
    for path, _ in entries:
        df = pd.read_csv(path)
        if df.empty:
            problems.append(f"{path.name} is empty")
        if empty := [str(c) for c in df.columns if df[c].isna().all()]:
            problems.append(f"{path.name} has all-NaN columns {empty}")
    return problems


INDENT = "    "


def panel_readme(label: str, folder: str, module, entries) -> str:
    """The panel's docstring, then each file's description and columns."""
    lines = [f"{label}, panel {folder}", "", inspect.getdoc(module) or "", ""]
    for path, meta in entries:
        df = pd.read_csv(path)
        columns = "Columns: " + ", ".join(map(str, df.columns))
        lines.append(f"{path.name} ({len(df)} rows)")
        for text in (meta["description"], columns):
            lines.append(
                textwrap.fill(text, 79, initial_indent=INDENT, subsequent_indent=INDENT)
            )
        lines.append("")
    return "\n".join(lines)


def figure_readme(label: str, source_dir: Path, rows: list[dict]) -> str:
    """What the zip holds, its schematic panels, and the provenance of the
    canonical fit and every runner output its figure build read."""
    schematic = [r["Panel"] for r in rows if r["Status"] == "schematic"]
    inputs_path = source_dir / common.SOURCE_INPUTS
    inputs = yaml.safe_load(inputs_path.read_text()) if inputs_path.exists() else []
    lines = [
        f"Source data for {label}: one folder per panel, each with CSV files",
        "and a README describing them.",
        "",
        "Generated by `uv run cccrise-sourcedata` from",
        f"{CODE_URL} at commit {git_sha()}.",
    ]
    if schematic:
        lines += ["", f"Schematic panels (no numerical data): {', '.join(schematic)}"]
    lines += ["", "Provenance of the analyses plotted", "=" * 34]
    for path in [DATASETS[DATASET].factors_path(), *map(Path, inputs)]:
        if (sidecar := sidecar_path(path)).exists():
            lines += ["", f"--- {relpath(path)} ---", sidecar.read_text()]
    return "\n".join(lines)


def row(label, folder, name, module, status, entries=()) -> dict:
    """A manifest row: one panel folder, or one panel module without data."""
    sizes = [p.stat().st_size for p, _ in entries]
    return {
        "Figure": label,
        "Panel": folder,
        "Module": name,
        "Status": status,
        "Files": "; ".join(p.name for p, _ in entries),
        "Size (KB)": round(sum(sizes) / 1e3, 1),
        # The docstring's first sentence; the panel README has all of it
        "Description": (inspect.getdoc(module) or "")
        .replace("\n", " ")
        .split(". ")[0]
        .rstrip(".")
        + ".",
    }


def manifest_markdown(manifest: pd.DataFrame) -> str:
    """The manifest as a Markdown table per figure."""
    lines = [
        "# EMBO source data manifest",
        "",
        "Generated by `uv run cccrise-sourcedata` from the last figure builds",
        "-- do not hand-edit. One row per panel folder in each figure's zip;",
        "status is `data`, `schematic` (no numerical data), or `missing`.",
    ]
    columns = ["Panel", "Module", "Status", "Files", "Size (KB)", "Description"]
    for label, rows in manifest.groupby("Figure", sort=False):
        lines += ["", f"## {label}", "", "| " + " | ".join(columns) + " |"]
        lines.append("|" + "---|" * len(columns))
        for row in rows[columns].itertuples(index=False):
            cells = [str(c).replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
