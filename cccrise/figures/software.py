"""
EMBO Press's Reagents and Tools table, its Software section: Python and every
package the project depends on (and those its optional extras add on this
platform), each with its PyPI page and the version `uv.lock` pins, written to
`output/source_data/software_versions.csv`:

    uv run python -m cccrise.figures.software
"""

import platform
import tomllib

import pandas as pd
from packaging.markers import Marker

from ..runners.provenance import REPO_ROOT
from . import common

OUTPUT_PATH = common.OUTPUT_DIR / "source_data" / "software_versions.csv"
COLUMNS = ["Reagent/Resource", "Reference or Source", "Identifier or catalogue number"]
PYPI_URL = "https://pypi.org/project/{}/"
PYTHON_URL = "https://www.python.org"


def main() -> None:
    rows = [("Python", PYTHON_URL, platform.python_version())]
    rows += [
        (name, PYPI_URL.format(name), version)
        for name, version in sorted(locked_versions().items())
    ]
    table = pd.DataFrame([("Software", "", ""), *rows], columns=COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH}: {len(rows)} packages")


def locked_versions() -> dict[str, str]:
    """The version `uv.lock` pins of each of the project's dependencies, and
    of those its optional extras pull in through their own extras (e.g.
    parafac2's `gpu`), skipping any whose marker excludes this platform."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    project = packages["cccrise"]
    deps = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        for dep in extra:
            deps.append(dep)
            for name in dep.get("extra", []):
                deps += packages[dep["name"]]["optional-dependencies"][name]
    return {
        d["name"]: packages[d["name"]]["version"]
        for d in deps
        if "marker" not in d or Marker(d["marker"]).evaluate()
    }


if __name__ == "__main__":
    main()
