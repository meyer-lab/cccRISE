"""
YAML provenance sidecars for runner outputs.

Issue #164: the committed ``.nc`` factor files had no generator and no record
of how they were made, and the ALAD artifact turned out to be irreproducible
(445 vs 513 LR pairs, 15,899 vs 16,675 genes against the same loader today) --
a drift that was invisible because nothing recorded the source file, the
package versions, or the ranks' justification. Every runner therefore writes
its result alongside a ``<stem>.provenance.yaml`` following the schema
documented in that issue:

``dataset``, ``source_file``, ``source_sha256``, ``loader``, ``loader_args``,
``condition_key``, the ranks and fitting parameters, ``rank_selection_evidence``,
``resulting_dims``, ``r2x_pf2`` / ``r2x_cp``, ``git_sha``, ``package_versions``
and ``generated``.

Two additions beyond that schema:

``rank_selection``
    Per-stage record of *why* each rank was chosen -- method, the sweep files,
    the sweep's own argmax rank, and a free-text ``notes`` rationale. The
    ``notes`` fields are the only ones meant to be hand-edited, and a re-run
    overwrites them, so keep the reasoning in the PR too.
``outputs``
    Path, size and sha256 of each file written, so a stale artifact is
    detectable rather than silently re-plotted.

The rank-selection runners write the same-shaped sidecar for their sweep CSVs,
with a ``selection`` block in place of ``rank_selection``.
"""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent

#: Suffix appended to an output file's stem, per the schema in issue #164.
PROVENANCE_SUFFIX = ".provenance.yaml"

#: Packages whose version can change a numerical result. Issue #164 names
#: tensorly, parafac2, vcsc and numpy specifically; the rest are here because
#: they too sit on the path from raw counts to factors. Missing ones are omitted.
TRACKED_PACKAGES = (
    "tensorly",
    "parafac2",
    "vcsc",
    "numpy",
    "scipy",
    "pandas",
    "anndata",
    "xarray",
    "scikit-learn",
)


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning None if git/the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_sha() -> str | None:
    """Commit of the code that produced a result.

    A trailing ``-dirty`` marks uncommitted edits to tracked files: the commit
    alone would otherwise imply a reproducibility those edits do not have.
    Untracked files are excluded -- they are not part of the tree the commit
    names, so a stray scratch file should not flag every run as unreproducible.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if status else commit


def package_versions() -> dict[str, str]:
    """Versions of the packages that can change a numerical result."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """Streaming sha256 of a file, so multi-GB ``.h5ad`` inputs don't load into RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: str | Path) -> str:
    """Render a path relative to the repo root when it lives inside it.

    Keeps sidecars portable between checkouts, and keeps cross-referenced paths
    (a sweep CSV cited by the pipeline's provenance) in the same form whichever
    runner wrote them. Paths outside the repo -- ``/opt/IVCSC/...`` source data,
    say -- stay absolute, which is what makes them useful.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def source_info(
    source_file: str | Path | None, checksum: bool = True
) -> dict[str, Any]:
    """``source_file`` / ``source_sha256`` for the raw input the run consumed.

    Checksumming a multi-GB ``.h5ad`` takes a minute or two -- negligible beside
    the fit it accompanies, and it is the field that would have caught the ALAD
    drift in issue #164. `checksum=False` skips it and says so in the sidecar.
    """
    if source_file is None:
        return {"source_file": None, "source_sha256": None}

    path = Path(source_file)
    if not path.exists():
        return {"source_file": relpath(path), "source_sha256": "file not found"}
    if not checksum:
        return {"source_file": relpath(path), "source_sha256": "skipped"}
    return {"source_file": relpath(path), "source_sha256": sha256_file(path)}


def file_info(path: Path) -> dict[str, Any]:
    """Path (relative to the repo when possible), byte size, and sha256 of a file."""
    path = Path(path)
    return {
        "path": relpath(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _to_plain(obj: Any) -> Any:
    """Coerce numpy scalars/arrays and Paths into YAML-representable builtins."""
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        return _to_plain(obj.item())
    if hasattr(obj, "tolist"):
        return _to_plain(obj.tolist())
    return obj


def build_metadata(runner: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a runner's own fields with the common provenance trailer.

    `fields` is emitted verbatim, in the order given -- each runner owns the
    shape of its own sidecar -- followed by ``git_sha``, ``package_versions``
    and ``generated``, which every sidecar carries.
    """
    return _to_plain(dict(fields)) | {
        "git_sha": git_sha(),
        "package_versions": package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "runner": runner,
    }


def sidecar_path(output_path: str | Path) -> Path:
    """The ``<stem>.provenance.yaml`` sidecar that belongs to an output file."""
    path = Path(output_path)
    return path.with_name(path.stem + PROVENANCE_SUFFIX)


def write_metadata(
    output_path: str | Path,
    metadata: Mapping[str, Any],
    header: str | None = None,
) -> Path:
    """Write a sidecar next to ``output_path`` and return the sidecar's path."""
    path = sidecar_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if header:
            handle.writelines(
                f"# {line}\n".rstrip() + "\n" for line in header.strip().splitlines()
            )
            handle.write("\n")
        yaml.safe_dump(
            _to_plain(metadata),
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return path


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read a sidecar written by `write_metadata`."""
    with open(path) as handle:
        return yaml.safe_load(handle) or {}
