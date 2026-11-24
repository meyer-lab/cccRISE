"""Manuscript tables, each written from the data as an editable spreadsheet
to `OUTPUT_DIR/tables/<name>.xlsx` (`manuscript/figures/tables/`), as MSB
asks for tables, and numbered by its `appendix_tables` position in
`_quarto.yml` (Appendix Table S1, ...). Build with
`uv run python -m cccrise.figures.tables`."""

import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from ..runners.datasets import DATASETS, VALIDATION_DIR
from .common import OUTPUT_DIR, load_runner_output
from .figure3._common import condition_obs
from .source_data import QUARTO_PATH

# {label: (obs column, {group: {subgroup: obs levels}})}; a group counts the
# samples of its subgroups
ALAD_OUTCOME_GROUPS = {
    "ALAD status": (
        "ALADstatus",
        {
            "Control": {"Control": ("control",)},
            "ALAD": {"Declined": ("declined",), "Recovered": ("recovered",)},
        },
    ),
    "Six-month outcome": (
        "6monthcondition",
        {
            "Stable/recovered": {
                "Stable": ("No CLAD/Stable",),
                "Recovered": ("No CLAD/Recovered",),
            },
            "CLAD/died": {"CLAD": ("CLAD",), "Died": ("Died",)},
        },
    ),
}
ALAD_OUTCOME_HEADER = ("Label", "Group", "Samples", "Sub-group", "Samples")
#: The cohorts whose cell-type labels `cohort_cell_types` lists, by name
COHORTS = {"alad": "ALAD", "ibd": "IBD"}
CELL_TYPE_HEADER = ("Cohort", "Coarse label", "Cells", "Fine label", "Cells")
COLUMN_WIDTH = 10.83
BOLD = Font(bold=True)
CELL_ALIGNMENT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def main():
    path = OUTPUT_DIR / "tables"
    path.mkdir(parents=True, exist_ok=True)
    numbers = yaml.safe_load(QUARTO_PATH.read_text())["appendix_tables"]
    for name, table in (
        ("alad_outcome_groups", alad_outcome_groups),
        ("cohort_cell_types", cohort_cell_types),
    ):
        sheet = f"Appendix Table S{numbers.index(name) + 1}"
        table(sheet).save(path / f"{name}.xlsx")


def workbook(sheet: str, header: tuple[str, ...]) -> Workbook:
    """A workbook whose one sheet, `sheet`, has `header` in bold from cell A1
    and every column `COLUMN_WIDTH` wide."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for column, heading in enumerate(header, start=1):
        put(ws, 1, column, heading).font = BOLD
        ws.column_dimensions[get_column_letter(column)].width = COLUMN_WIDTH
    return wb


def alad_outcome_groups(sheet: str) -> Workbook:
    """BAL samples per ALAD status and six-month outcome group and subgroup
    from cell A1, each label and group cell merged down the rows of its
    subgroups; the title and notes are the legend's."""
    obs = condition_obs()
    wb = workbook(sheet, ALAD_OUTCOME_HEADER)
    ws = wb.active
    row = 2
    for label, (column, groups) in ALAD_OUTCOME_GROUPS.items():
        label_row = row
        for group, subgroups in groups.items():
            group_row, total = row, 0
            for subgroup, levels in subgroups.items():
                n = int(obs[column].isin(levels).sum())
                put(ws, row, 4, subgroup)
                put(ws, row, 5, n)
                total, row = total + n, row + 1
            put_merged(ws, group_row, row - 1, 2, group)
            put_merged(ws, group_row, row - 1, 3, total)
        put_merged(ws, label_row, row - 1, 1, label)
    return wb


def label_counts(dataset: str) -> pd.Series:
    """Cells per coarse and fine cell-type label of `dataset` (the labellings
    the comparators pool cells by), from `cccrise-cohort`'s counts."""
    counts = load_runner_output(
        f"cohort_celltype_counts_{DATASETS[dataset].output_key}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run cccrise-cohort --dataset {dataset}",
    )
    return counts.groupby(["lineage", "celltype"], observed=True)["n_cells"].sum()


def cohort_cell_types(sheet: str) -> Workbook:
    """Cells per coarse and fine cell-type label of each cohort (`COHORTS`)
    from cell A1: coarse labels largest first, each over its fine labels,
    largest first; each cohort and coarse cell merged down the rows of its
    fine labels. The title and notes are the legend's."""
    wb = workbook(sheet, CELL_TYPE_HEADER)
    ws = wb.active
    row = 2
    for dataset, cohort in COHORTS.items():
        counts = label_counts(dataset)
        cohort_row = row
        totals = counts.groupby(level=0).sum().sort_values(ascending=False)
        for coarse, total in totals.items():
            coarse_row = row
            for fine, n in counts[coarse].sort_values(ascending=False).items():
                put(ws, row, 4, fine)
                put(ws, row, 5, int(n))
                row += 1
            put_merged(ws, coarse_row, row - 1, 2, coarse)
            put_merged(ws, coarse_row, row - 1, 3, int(total))
        put_merged(ws, cohort_row, row - 1, 1, cohort)
    return wb


def put_merged(ws, first_row: int, last_row: int, column: int, value):
    """`value` in a cell merged down from `first_row` to `last_row`."""
    put(ws, first_row, column, value)
    if last_row > first_row:
        ws.merge_cells(
            start_row=first_row,
            end_row=last_row,
            start_column=column,
            end_column=column,
        )


def put(ws, row: int, column: int, value):
    """Write `value` left and vertically centered."""
    cell = ws.cell(row, column, value)
    cell.alignment = CELL_ALIGNMENT
    return cell


if __name__ == "__main__":
    main()
