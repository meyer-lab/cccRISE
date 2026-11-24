"""Manuscript tables, each written from the data as an editable spreadsheet
to `OUTPUT_DIR/tables/<name>.xlsx` (`manuscript/figures/tables/`), as MSB
asks for tables, and numbered by its `appendix_tables` position in
`_quarto.yml` (Appendix Table S1, ...). Build with
`uv run python -m cccrise.figures.tables`."""

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .common import OUTPUT_DIR
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
COLUMN_WIDTH = 10.83
BOLD = Font(bold=True)
CELL_ALIGNMENT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def main():
    path = OUTPUT_DIR / "tables"
    path.mkdir(parents=True, exist_ok=True)
    numbers = yaml.safe_load(QUARTO_PATH.read_text())["appendix_tables"]
    name = "alad_outcome_groups"
    sheet = f"Appendix Table S{numbers.index(name) + 1}"
    alad_outcome_groups(sheet).save(path / f"{name}.xlsx")


def alad_outcome_groups(sheet: str) -> Workbook:
    """BAL samples per ALAD status and six-month outcome group and subgroup
    from cell A1, each label and group cell merged down the rows of its
    subgroups; the title and notes are the legend's."""
    obs = condition_obs()
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for column, heading in enumerate(ALAD_OUTCOME_HEADER, start=1):
        put(ws, 1, column, heading).font = BOLD
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
    for column in range(1, len(ALAD_OUTCOME_HEADER) + 1):
        ws.column_dimensions[get_column_letter(column)].width = COLUMN_WIDTH
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
