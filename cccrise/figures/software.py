"""
EMBO Press's Reagents and Tools table, its Software section: CCC-RISE's
repository at the commit generating the table, Python and the packages the
method itself rests on (`CORE_PACKAGES`, not those only plotting, serving the
app or reading/writing files), each with its PyPI page and the version
`uv.lock` pins, and the comparator methods
(`COMPARATORS`) with their papers and repositories. Written to
`output/source_data/software_versions.csv`, and as an HTML table (with *et
al.* italicized, which CSV can't hold) to copy into the manuscript's table:

    uv run python -m cccrise.figures.software
"""

import html
import platform
import tomllib

import pandas as pd

from ..runners.provenance import REPO_ROOT, git_sha
from . import common
from .source_data import CODE_URL

OUTPUT_PATH = common.OUTPUT_DIR / "source_data" / "software_versions.csv"
COLUMNS = ["Reagent/Resource", "Reference or Source", "Identifier or catalogue number"]
PYPI_URL = "https://pypi.org/project/{}/"
PYTHON_URL = "https://www.python.org"
ET_AL = "et al."
#: The packages the method rests on: fitting, rank selection, normalization,
#: statistics, embedding, and the optional GPU backend (`gpu` extra).
CORE_PACKAGES = [
    "anndata",
    "cupy-cuda12x",
    "kneed",
    "numpy",
    "nvmath-python",
    "pacmap",
    "pandas",
    "parafac2",
    "scanpy",
    "scikit-learn",
    "scipy",
    "scrise",
    "tensorly",
    "tensorly-viz",
    "vsparse",
]
#: The comparator methods, none a Python dependency, each with its paper as
#: the reference list cites it (first author, year first published). NICHES
#: runs as its R package at release v1.2.5 (`run_niches.NICHES_COMMIT`; its
#: DESCRIPTION still says 1.2.4). Scriabin runs as its R package, and STACCato
#: is re-implemented from its R package (`comparators.staccato`), at these
#: commits: neither has a release, and every commit's version is 0.0.0.9000,
#: so the commit alone identifies them. Tensor-cell2cell's pipeline is
#: re-implemented from the current release (`comparators.methods`).
COMPARATORS = [
    (
        "NICHES",
        f"Raredon {ET_AL}, 2022",
        "https://github.com/msraredon/NICHES",
        "1.2.5",
    ),
    (
        "Scriabin",
        f"Wilk {ET_AL}, 2023",
        "https://github.com/BlishLab/scriabin",
        "commit bce695b",
    ),
    (
        "STACCato",
        f"Dai {ET_AL}, 2026",
        "https://github.com/daiqile96/STACCato",
        "commit cb76f21",
    ),
    (
        "Tensor-cell2cell",
        f"Armingol {ET_AL}, 2022",
        "https://github.com/earmingol/cell2cell",
        "0.9.0",
    ),
]


def main() -> None:
    commit = git_sha() or ""
    dirty = "-dirty" if commit.endswith("-dirty") else ""
    rows = [
        ("CCC-RISE", f"This study; {CODE_URL}", f"commit {commit[:7]}{dirty}"),
        ("Python", PYTHON_URL, platform.python_version()),
    ]
    rows += sorted(
        [
            *(
                (name, PYPI_URL.format(name), version)
                for name, version in locked_versions().items()
            ),
            *(
                (name, f"{ref}; {url}", version)
                for name, ref, url, version in COMPARATORS
            ),
        ],
        key=lambda row: row[0].lower(),
    )
    table = pd.DataFrame([("Software", "", ""), *rows], columns=COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    OUTPUT_PATH.with_suffix(".html").write_text(html_table(table))
    print(f"{OUTPUT_PATH} (and .html): {len(rows)} packages")


def html_table(table: pd.DataFrame) -> str:
    """`table` as an HTML table, `ET_AL` italicized."""

    def cell(tag: str, value: str) -> str:
        text = html.escape(value).replace(ET_AL, f"<i>{ET_AL}</i>")
        return f"<{tag}>{text}</{tag}>"

    head = "".join(cell("th", c) for c in table.columns)
    body = "".join(
        "<tr>" + "".join(cell("td", v) for v in row) + "</tr>\n"
        for row in table.itertuples(index=False)
    )
    return f"<table>\n<tr>{head}</tr>\n{body}</table>\n"


def locked_versions() -> dict[str, str]:
    """The version `uv.lock` pins of each of `CORE_PACKAGES`."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    versions = {p["name"]: p["version"] for p in lock["package"]}
    return {name: versions[name] for name in CORE_PACKAGES}


if __name__ == "__main__":
    main()
