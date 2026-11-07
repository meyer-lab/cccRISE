"""Manuscript tables, each written from the data as a Markdown pipe table to
`OUTPUT_DIR/tables/<name>.md` (`manuscript/figures/tables/`), which the
Appendix includes beside its legend (MSB's Appendix Table S1, ...). Build
with `uv run python -m cccrise.figures.tables`."""

from .common import OUTPUT_DIR
from .figure3._common import condition_obs

EM_SPACE = "\u2003"  # a subgroup's indent per level; Markdown drops plain spaces

# {heading: (obs column, [(level, label, levels pooled)])}; "All" has none
ALAD_OUTCOME_GROUPS = {
    None: ("ALADstatus", [(0, "All", ("control", "declined", "recovered"))]),
    "ALAD status": (
        "ALADstatus",
        [
            (1, "Control", ("control",)),
            (1, "ALAD", ("declined", "recovered")),
            (2, "Declined", ("declined",)),
            (2, "Recovered", ("recovered",)),
        ],
    ),
    "Six-month outcome": (
        "6monthcondition",
        [
            (1, "CLAD/died", ("CLAD", "Died")),
            (2, "CLAD", ("CLAD",)),
            (2, "Died", ("Died",)),
            (1, "Stable/recovered", ("No CLAD/Stable", "No CLAD/Recovered")),
            (2, "Stable", ("No CLAD/Stable",)),
            (2, "Recovered", ("No CLAD/Recovered",)),
        ],
    ),
}


def main():
    path = OUTPUT_DIR / "tables"
    path.mkdir(parents=True, exist_ok=True)
    (path / "alad_outcome_groups.md").write_text(alad_outcome_groups())


def alad_outcome_groups() -> str:
    """BAL samples per ALAD status and six-month outcome group, headings and
    the whole cohort in bold, subgroups indented under the group pooling
    them."""
    obs = condition_obs()
    rows = []
    for heading, (column, groups) in ALAD_OUTCOME_GROUPS.items():
        if heading:
            rows.append((f"**{heading}**", ""))
        for level, label, levels in groups:
            n = int(obs[column].isin(levels).sum())
            rows.append(
                (f"**{label}**", f"**{n}**")
                if level == 0
                else (EM_SPACE * 2 * (level - 1) + label, str(n))
            )
    return "".join(
        f"| {group} | {n} |\n"
        for group, n in [("Group", "Samples"), (":--", "--:"), *rows]
    )


if __name__ == "__main__":
    main()
