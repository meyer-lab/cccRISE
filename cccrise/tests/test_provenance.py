import yaml

from cccrise.runners import provenance


def test_record_output_shares_common_fields_and_replaces_its_own_record(tmp_path):
    for name in ("a.parquet", "b.parquet"):
        (tmp_path / name).write_bytes(name.encode())
    provenance.record_output(
        tmp_path / "a.parquet", {"dataset": "alad", "rank": 1}, "run a"
    )
    provenance.record_output(
        tmp_path / "b.parquet", {"dataset": "alad", "rank": 2}, "run b"
    )
    written = yaml.safe_load((tmp_path / provenance.FOLDER_PROVENANCE).read_text())
    assert written["shared"] == {"dataset": "alad"}
    assert written["files"]["b.parquet"]["rank"] == 2
    assert written["files"]["a.parquet"]["command"] == "run a"

    provenance.record_output(
        tmp_path / "a.parquet", {"dataset": "alad", "rank": 3}, "run a again"
    )
    records = provenance.folder_records(tmp_path)
    assert records["a.parquet"]["rank"] == 3
    assert records["b.parquet"] == {
        "dataset": "alad",
        "sha256": provenance.sha256_file(tmp_path / "b.parquet"),
        "command": "run b",
        "rank": 2,
    }


def test_records_of_removed_files_are_dropped(tmp_path):
    for name in ("a.parquet", "b.parquet"):
        (tmp_path / name).write_bytes(name.encode())
        provenance.record_output(tmp_path / name, {"dataset": "alad"}, "run")
    (tmp_path / "b.parquet").unlink()
    provenance.record_output(tmp_path / "a.parquet", {"dataset": "alad"}, "run")
    assert list(provenance.folder_records(tmp_path)) == ["a.parquet"]


def test_provenance_text_reads_a_folder_record(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"a")
    provenance.record_output(tmp_path / "a.parquet", {"rank": 4}, "run a")
    text = provenance.provenance_text(tmp_path / "a.parquet")
    assert text is not None and "rank: 4" in text
    assert provenance.provenance_text(tmp_path / "missing.parquet") is None
