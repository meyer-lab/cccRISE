"""Tests for the results bundle for public deposit (`runners.deposit`)."""

import zipfile

from cccrise.runners import deposit


def test_deposit_bundles_every_result_with_checksums(tmp_path, monkeypatch):
    data = tmp_path / "data"
    results = data / "validation_results"
    (results / "alt").mkdir(parents=True)
    (results / "a.parquet").write_bytes(b"a")
    (results / "alt" / "b.nc").write_bytes(b"bb")
    monkeypatch.setattr(deposit, "DATA_DIR", data)
    monkeypatch.setattr(deposit, "RESULT_DIRS", (results,))
    monkeypatch.setattr(deposit, "OUTPUT_PATH", tmp_path / "out" / "results.zip")

    deposit.main()

    with zipfile.ZipFile(tmp_path / "out" / "results.zip") as archive:
        assert sorted(archive.namelist()) == [
            "README.txt",
            "validation_results/a.parquet",
            "validation_results/alt/b.nc",
        ]
        readme = archive.read("README.txt").decode()
    assert "validation_results/alt/b.nc\t2\t" in readme
