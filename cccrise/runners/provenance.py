"""
YAML provenance for runner outputs.

Each runner records what produced its result: the source data and its
checksum, the loader arguments, every parameter, the resulting dimensions, the
headline metrics, and the code version. Most write it alongside the result as
a ``<stem>.provenance.yaml`` sidecar; the comparator runners record every file
of a folder in that folder's one ``provenance.yaml`` (`record_output`), with
each file's checksum and the command that writes it.
"""

import hashlib
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent

#: Suffix appended to an output file's stem to name its sidecar.
PROVENANCE_SUFFIX = ".provenance.yaml"
#: A folder's provenance file (`record_output`).
FOLDER_PROVENANCE = "provenance.yaml"
FOLDER_GUIDE = """\
One record per data file in this folder: `shared` holds what every file's
record has in common, `files` the rest of each, with the file's sha256 and
the command that writes it with this repository's code.
"""
#: A record holding this key was merged into its folder's provenance file
#: from a per-file sidecar, without rerunning (`FOLDER_POST_HOC`).
POST_HOC_KEY = "consolidated_post_hoc"
FOLDER_POST_HOC = """\
Records with `consolidated_post_hoc` were written by their runs as per-file
`<stem>.provenance.yaml` sidecars and merged here, on the date it gives,
without rerunning: their settings and git_sha are as the runs recorded them,
and their `command` was reconstructed from those settings. Running a command
replaces its files' records.
"""


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
    files are ignored.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    return f"{commit}-dirty" if _git("status", "--porcelain", "-uno") else commit


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """Streaming sha256, so multi-GB source files don't load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: str | Path) -> str:
    """Render a path relative to the repo root when it lives inside it.
    Keeps sidecars portable between checkouts.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


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


def build_metadata(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a runner's own fields with the trailer every sidecar carries.

    `fields` is emitted verbatim in the order given -- each runner owns the
    shape of its own sidecar.
    """
    return _to_plain(dict(fields)) | {
        "git_sha": git_sha(),
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }


class _Dumper(yaml.SafeDumper):
    """SafeDumper that keeps integer sequences on one wrapped line.

    A rank sweep records every rank it visited, which block style renders as
    one line per rank -- fifty lines of scalar, burying the parameters that
    actually differ between runs. Flow style still round-trips to the same
    list, so this is presentation only.
    """


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.Node:
    inline = bool(data) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in data
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_Dumper.add_representer(list, _represent_list)


def sidecar_path(output_path: str | Path) -> Path:
    """The sidecar that belongs to an output file."""
    path = Path(output_path)
    return path.with_name(path.stem + PROVENANCE_SUFFIX)


def read_metadata(output_path: str | Path) -> dict[str, Any]:
    """Read the sidecar belonging to `output_path`."""
    return read_metadata_at(sidecar_path(output_path))


def read_metadata_at(path: str | Path) -> dict[str, Any]:
    """Read a YAML provenance file at the exact given `path`; the reading
    counterpart of `write_metadata_at`."""
    with open(path) as handle:
        return yaml.safe_load(handle)


def write_metadata_at(
    path: str | Path,
    metadata: Mapping[str, Any],
    header: str | None = None,
) -> Path:
    """Write a YAML provenance file at the exact given `path`, for a file
    not tied to any single output's stem (`write_metadata` is the usual
    entry point for a sidecar).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if header:
            handle.writelines(
                f"# {line}\n".rstrip() + "\n" for line in header.strip().splitlines()
            )
            handle.write("\n")
        yaml.dump(
            _to_plain(metadata),
            handle,
            Dumper=_Dumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=88,
        )
    return path


def write_metadata(
    output_path: str | Path,
    metadata: Mapping[str, Any],
    header: str | None = None,
) -> Path:
    """Write a sidecar next to ``output_path`` and return the sidecar's path."""
    return write_metadata_at(sidecar_path(output_path), metadata, header)


def folder_records(folder: str | Path) -> dict[str, dict[str, Any]]:
    """Each file's full record in `folder`'s provenance file (`record_output`),
    its shared settings merged in; empty if it has none."""
    path = Path(folder) / FOLDER_PROVENANCE
    if not path.exists():
        return {}
    data = read_metadata_at(path)
    return {name: data["shared"] | entry for name, entry in data["files"].items()}


def record_output(
    output_path: str | Path,
    metadata: Mapping[str, Any],
    command: str,
    header: str | None = None,
) -> Path:
    """Record ``output_path``'s provenance in its folder's provenance file,
    with its sha256 and the `command` that wrote it, replacing any earlier
    record of it; returns the provenance file's path."""
    path = Path(output_path)
    records = folder_records(path.parent)
    records[path.name] = {
        "sha256": sha256_file(path),
        "command": command,
    } | dict(metadata)
    return write_folder_records(path.parent, records, header)


def write_table(
    path: str | Path,
    table: Any,
    metadata: Mapping[str, Any],
    command: str,
    header: str | None = None,
) -> None:
    """Write a runner's `table` (a DataFrame) as Parquet at `path` and
    `record_output` it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    record_output(path, metadata, command, header=header)
    print(f"   wrote {path.name} ({len(table)} rows)", flush=True)


def write_folder_records(
    folder: str | Path, records: Mapping[str, Mapping[str, Any]], header: str | None
) -> Path:
    """Write `folder`'s provenance file from each file's full record (of the
    files still there): what every record shares once, then each file's own
    fields."""
    folder = Path(folder)
    kept = {
        name: _to_plain(record)
        for name, record in sorted(records.items())
        if (folder / name).exists()
    }
    shared = _shared_fields(list(kept.values()))
    files = {
        name: {k: v for k, v in record.items() if k not in shared}
        for name, record in kept.items()
    }
    path = write_metadata_at(
        folder / FOLDER_PROVENANCE,
        {"shared": shared, "files": files},
        "\n".join(filter(None, [header, FOLDER_GUIDE])),
    )
    if any(POST_HOC_KEY in r for r in kept.values()):
        with open(path, "a") as handle:
            handle.write("\n")
            handle.writelines(f"# {line}\n" for line in FOLDER_POST_HOC.splitlines())
    return path


def _shared_fields(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The fields, but a file's sha256, every one of `records` has alike."""
    first, *rest = records
    return {
        key: value
        for key, value in first.items()
        if key != "sha256" and all(key in r and r[key] == value for r in rest)
    }


def provenance_text(output_path: str | Path) -> str | None:
    """An output file's provenance as YAML text: its sidecar, else its record
    in its folder's provenance file; None if it has neither."""
    sidecar = sidecar_path(output_path)
    if sidecar.exists():
        return sidecar.read_text()
    record = folder_records(Path(output_path).parent).get(Path(output_path).name)
    if record is None:
        return None
    first = {k: record[k] for k in ("sha256", "command") if k in record}
    return yaml.dump(
        first | record, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=88
    )
