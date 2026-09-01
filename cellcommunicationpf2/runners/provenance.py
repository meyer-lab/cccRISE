"""
YAML provenance sidecars for runner outputs.

Each runner writes its result alongside a ``<stem>.provenance.yaml`` recording
what produced it: the source data and its checksum, the loader and its
arguments, every parameter, the resulting dimensions, summary metrics, the code
version, and the versions of the packages that affect the numbers. A result can
then be traced to an exact invocation, and a stale one is detectable.
"""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent

#: Suffix appended to an output file's stem to name its sidecar.
PROVENANCE_SUFFIX = ".provenance.yaml"

#: Packages whose version can change a numerical result. Missing ones are omitted.
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
    """Run a git command in the repo, returning None if it fails."""
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


@cache
def git_sha() -> str | None:
    """Commit of the code that produced a result.

    A trailing ``-dirty`` marks uncommitted edits to tracked files; untracked
    files are ignored. Memoized, and each runner calls it before writing
    anything -- a runner overwrites tracked artifacts, so reading the tree
    afterwards would see the run's own output and report ``-dirty`` every time.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    return f"{commit}-dirty" if _git("status", "--porcelain", "-uno") else commit


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
    """Streaming sha256, so multi-GB source files don't load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: str | Path) -> str:
    """Render a path relative to the repo root when it lives inside it.

    Keeps sidecars portable between checkouts. Source data outside the repo
    stays absolute, which is what makes it identifiable.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def file_info(path: Path) -> dict[str, Any]:
    """Path, byte size, and sha256 of a file written by a runner."""
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
    """Wrap a runner's own fields with the trailer every sidecar carries.

    `fields` is emitted verbatim in the order given -- each runner owns the
    shape of its own sidecar.
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
    """The sidecar that belongs to an output file."""
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
