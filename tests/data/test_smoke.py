from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import ValidationError

runner = CliRunner()


@pytest.fixture
def images(tmp_path: Path) -> fb.data.Dataset:
    src = tmp_path / "images"
    for split, count in (("train", 30), ("val", 3)):
        for label in ("cat", "dog"):
            (src / split / label).mkdir(parents=True)
            for i in range(count):
                (src / split / label / f"{i}.txt").write_text(f"{split}/{label}/{i}")
    (src / "README.txt").write_text("readme")
    ds = fb.dataset("images")
    with ds.edit() as edit:
        edit.add_dir(src)
    return ds


@pytest.fixture
def smoke(monkeypatch: pytest.MonkeyPatch):
    def start(n: int) -> None:
        monkeypatch.setenv("PLATFORM_SMOKE", str(n))

    return start


def test_reads_sample_each_top_level_folder(images, smoke) -> None:
    full = images.files()
    smoke(5)
    paths = images.files()["path"].to_pylist()
    assert sum(p.startswith("train/") for p in paths) == 5
    assert sum(p.startswith("val/") for p in paths) == 5  # a small split is never left empty
    assert "README.txt" in paths  # files directly under the prefix count as one folder
    assert paths == images.files()["path"].to_pylist()  # the same items every time
    train = images.files(prefix="train/")["path"].to_pylist()  # folders under the prefix
    assert sum(p.startswith("train/cat/") for p in train) == 5 == sum("dog/" in p for p in train)
    assert images.stats().to_pylist()[0]["files"] == len(paths)
    unsampled = next(p for p in full["path"].to_pylist() if p not in paths)
    with images.open(unsampled) as f:  # hard-coded paths still work
        assert f.read()


def test_nothing_is_saved(images, smoke, tmp_path: Path, capsys) -> None:
    images.tag("first")
    smoke(5)
    with images.edit() as edit:
        edit.remove("README.txt")
    assert edit.version == 1 and images.version == 1
    assert "smoke run: new version of test/images (1 items changed) (not saved)" in (
        capsys.readouterr().err
    )
    images.tag("second")
    images.untag("first")
    assert images.tags() == {"first": 1}
    with fb.dataset("other").edit() as edit:
        edit.copy_from(images)
    assert not fb.dataset("other").exists
    assert fb.data.prune(keep_latest=1, retention=timedelta(0), grace=timedelta(0)).dry_run


def test_checks_still_run(images, smoke) -> None:
    with images.edit() as edit:
        edit.set_schema("columns:\n  label:\n    dtype: str\n")  # saved before the smoke run
        edit.update("README.txt", label="x")
        for path in images.files()["path"].to_pylist():
            if path != "README.txt":
                edit.update(path, label="y")
    smoke(5)
    with pytest.raises(ValidationError):
        with images.edit() as edit:
            edit.update("README.txt", label=None)


def test_query_samples_rows(tmp_path: Path, smoke) -> None:
    pq.write_table(pa.table({"id": list(range(5000))}), tmp_path / "big.parquet")
    with fb.dataset("sales").edit() as edit:
        edit.add_records(tmp_path / "big.parquet")
    smoke(10)
    assert fb.dataset("sales").to_table().num_rows == 1000  # a hundred rows per item
    result = runner.invoke(
        app, ["data", "query", "-d", "sales", "-q", "SELECT count(*) AS n FROM sales"]
    )
    assert result.exit_code == 0 and result.output.splitlines()[1].strip() == "1000"


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch):
    """A smoke run with a scratch area, as `fieldbook run --smoke` starts it."""
    monkeypatch.setenv("PLATFORM_SMOKE", "5")
    monkeypatch.setenv("PLATFORM_SCRATCH", "run-1")
    return "run-1"


def test_scratch_area(images, scratch, monkeypatch: pytest.MonkeyPatch) -> None:
    real_version = images.version
    with images.edit() as edit:
        edit.remove("README.txt")
    with fb.dataset("derived").edit() as edit:
        edit.copy_from(images)  # reads the scratch version of images
    first = images.version  # the scratch copy's own numbering
    assert images.in_scratch and first != real_version
    assert "README.txt" not in images.files()["path"].to_pylist()
    pinned = fb.dataset("images", version=real_version).files()["path"].to_pylist()
    assert "README.txt" in pinned  # a pinned version reads the real dataset (sampled)
    assert fb.dataset("derived").log()[0].inputs == (f"scratch/run-1/test/images@{first}",)
    with images.edit() as edit:
        edit.remove("train/cat/0.txt")
    second = images.version
    assert second > first
    monkeypatch.delenv("PLATFORM_SMOKE")
    monkeypatch.delenv("PLATFORM_SCRATCH")
    assert images.version == real_version and not fb.dataset("derived").exists  # untouched
    monkeypatch.setenv("FIELDBOOK_SCRATCH", "run-1")  # a read-only view of the scratch area
    assert fb.dataset("images").version == second
    for write in (
        lambda: fb.dataset("images").edit(),
        lambda: fb.dataset("images").tag("x"),
        lambda: fb.dataset("images").revert(1),
        lambda: fb.dataset("images").delete_versions([1]),
    ):
        with pytest.raises(fb.data.DatasetError, match="read-only"):
            write()


def test_cleanup_removes_stale_scratch_areas(images, scratch, monkeypatch) -> None:
    with images.edit() as edit:
        edit.remove("README.txt")
    monkeypatch.delenv("PLATFORM_SMOKE")
    monkeypatch.delenv("PLATFORM_SCRATCH")
    assert fb.data.prune().scratch == []  # within the grace period
    report = fb.data.prune(grace=timedelta(0))
    assert len(report.scratch) == 1 and not fb.storage.local_path("scratch/run-1").exists()
