"""
EMBO Press source data for the manuscript's figures, packaged from what each
panel saved with `common.source_data` during its figure's last build:

    uv run cccrise-sourcedata                    # every figure in `figures()`
    uv run cccrise-sourcedata 2 S_alad_cohort  # some of them

Each figure becomes `output/source_data/<label>.zip` (e.g. `Figure 2.zip`,
`Figure EV1.zip`, `Appendix Figure S2.zip`), one folder per panel (`2A/`,
`EV1A/`, `S2A/`, or `2B-C/` for files spanning a panel module's letters)
holding its CSVs and a README (a figure without panel letters is one
folder, `EV4/`), plus a README of the provenance of every
factors file and runner output its last full build read. Main figures are
those `_panels.yml` has letters for, supplements those `_quarto.yml` lists,
and letters come from `_panels.yml`, as each figure's last full build
recorded them. Schematic panels (`SCHEMATIC = True`) need no data, and a
figure of only those gets no zip. Every panel is listed in the manifest
(`manuscript/source_data_manifest.md`). Exits non-zero,
writing no zips, if any panel's data is missing or malformed, or a figure
hasn't been built in full.
"""

import importlib
import inspect
import itertools
import re
import shutil
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from ..runners.provenance import REPO_ROOT, git_sha, provenance_text, relpath
from . import common

QUARTO_PATH = REPO_ROOT / "manuscript" / "_quarto.yml"
# MSB's classes of supplementary figure, as `_quarto.yml` lists them: the
# list, the label before each figure's number, and its panel folder prefix
SUPPLEMENT_CLASSES = (
    ("ev_figures", "Figure EV", "EV"),
    ("appendix_figures", "Appendix Figure S", "S"),
)
OUTPUT_DIR = common.OUTPUT_DIR / "source_data"
MANIFEST_PATH = REPO_ROOT / "manuscript" / "source_data_manifest.md"
CODE_URL = "https://github.com/meyer-lab/cccRISE"


def figures() -> list[tuple[str, str, str]]:
    """(cccrise-fbuild name, label, panel prefix) of each figure packaged:
    the main figures, then the Expanded View and Appendix figures, each
    numbered in its `_quarto.yml` list's order."""
    quarto = yaml.safe_load(QUARTO_PATH.read_text())
    return [(f, f"Figure {f}", f) for f in main_figures()] + [
        (f"S_{name}", f"{label}{i}", f"{prefix}{i}")
        for key, label, prefix in SUPPLEMENT_CLASSES
        for i, name in enumerate(quarto[key], start=1)
    ]


def main_figures() -> list[str]:
    """The main figures `_panels.yml` has panel letters for, in order."""
    panels = yaml.safe_load(common.PANELS_PATH.read_text())
    numbers = [
        m[1] for k, v in panels.items() if v and (m := re.fullmatch(r"fig(\d+)", k))
    ]
    return sorted(numbers, key=int)


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
        if f"fig{figure}" not in letters:
            errors.append(f"{label}: no panel letters; build it in full first")
        # A figure built without lettered panels is one panel, its own module
        figure_letters = letters.get(f"fig{figure}") or {f"figure{figure}": []}
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
    return rows, errors + add_provenance(label, rows, source_dir, out_dir)


def add_provenance(label: str, rows: list[dict], source_dir: Path, out_dir: Path):
    """Write the figure's README of the inputs its last full build read, and
    list them in its manifest `rows`. Returns an error if it has data but no
    record of its inputs."""
    inputs = figure_inputs(source_dir)
    for r in rows:
        r["Figure inputs"] = "; ".join(relpath(p) for p in inputs or [])
    if not any(r["Status"] == "data" for r in rows):
        return []
    (out_dir / "README.txt").write_text(figure_readme(label, inputs or [], rows))
    return (
        []
        if inputs is not None
        else [f"{label}: no record of its inputs; build it in full"]
    )


def figure_inputs(source_dir: Path) -> list[Path] | None:
    """The factors files and runner outputs the figure's last full build
    read, or None if it hasn't been built in full."""
    path = source_dir / common.SOURCE_INPUTS
    return (
        [Path(p) for p in yaml.safe_load(path.read_text())] if path.exists() else None
    )


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


def figure_readme(label: str, inputs: list[Path], rows: list[dict]) -> str:
    """What the zip holds, its schematic panels, and the provenance of every
    factors file and runner output its figure build read."""
    schematic = [r["Panel"] for r in rows if r["Status"] == "schematic"]
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
    for path in inputs:
        if (text := provenance_text(path)) is not None:
            lines += ["", f"--- {relpath(path)} ---", text]
    return "\n".join(lines)


def row(label, folder, name, module, status, entries=()) -> dict:
    """A manifest row: one panel folder, or one panel module without data."""
    sizes = [p.stat().st_size for p, _ in entries]
    return {
        "Figure": label,
        "Panel": folder,
        "Module": name,
        "Script": relpath(module.__file__) if hasattr(module, "__file__") else "",
        "Status": status,
        "Files": "; ".join(p.name for p, _ in entries),
        "Size (KB)": round(sum(sizes) / 1e3, 1),
        # The docstring's first paragraph; the panel README has all of it
        "Description": (inspect.getdoc(module) or "").split("\n\n")[0].replace(
            "\n", " "
        ),
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
    columns = ["Panel", "Script", "Status", "Files", "Size (KB)", "Description"]
    for label, rows in manifest.groupby("Figure", sort=False):
        inputs = rows["Figure inputs"].iloc[0].split("; ")
        lines += [
            "",
            f"## {label}",
            "",
            "Analyses plotted (with provenance in the zip's README):",
        ]
        lines += [f"- `{p}`" for p in inputs if p] + [
            "",
            "| " + " | ".join(columns) + " |",
        ]
        lines.append("|" + "---|" * len(columns))
        for row in rows[columns].itertuples(index=False):
            cells = [str(c).replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
