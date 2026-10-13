import os
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.data import ConflictError, DatasetError, verify


def _three_versions(files: Path) -> fb.data.Dataset:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
        edit.set_info(license="MIT")
    with ds.edit() as edit:
        edit.update("a/1.txt", files / "a" / "2.txt")
        edit.update("3.jpg", label="tabby")
        edit.remove("a/2.txt")
        edit.add("new.txt", files / "a" / "1.txt")
    with ds.edit() as edit:
        edit.set_info(license=None, description="cats")
    return ds


# --- parallel add ------------------------------------------------------------


def test_add_many_in_parallel_with_progress(tmp_path: Path) -> None:
    src = tmp_path / "many"
    src.mkdir()
    for i in range(200):
        (src / f"{i:03}.bin").write_bytes(os.urandom(32) if i % 2 else b"same")
    ticks: list[int] = []
    ds = fb.dataset("many")
    with ds.edit() as edit:
        assert edit.add_dir(src, workers=8, progress=ticks.append) == 200
    assert sum(ticks) == 200
    assert ds.stats().to_pylist()[0]["files"] == 200
    assert ds.verify(deep=True).ok


def test_add_many_checks_paths_before_copying(files: Path) -> None:
    ds = fb.dataset("cats")
    with pytest.raises(DatasetError, match="already added"):
        with ds.edit() as edit:
            edit.add_many([("a.txt", files / "a" / "1.txt"), ("a.txt", files / "a" / "2.txt")])
    assert list(fb.storage.fs().find("objects")) == [], "nothing copied"


# --- info --------------------------------------------------------------------


def test_info_is_versioned_and_kept_across_edits(files: Path) -> None:
    ds = _three_versions(files)
    first, second, third = (info.version for info in reversed(ds.log()))
    assert fb.dataset("cats", version=first).info() == {"license": "MIT"}
    assert fb.dataset("cats", version=second).info() == {"license": "MIT"}
    assert ds.info() == {"description": "cats"} and ds.version == third


def test_concurrent_info_changes_conflict(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    first, second = ds.edit(), ds.edit()
    first.set_info(license="MIT")
    second.set_info(license="GPL")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()
    third, fourth = ds.edit(), ds.edit()
    third.set_info(source="web")
    fourth.set_info(description="cats")
    third.commit()
    fourth.commit()
    assert ds.info() == {"license": "MIT", "source": "web", "description": "cats"}


# --- revert ------------------------------------------------------------------


def test_revert_creates_identical_version(files: Path) -> None:
    ds = _three_versions(files)
    ds.tag("first", version=1)
    reverted = ds.revert("first")
    assert reverted == ds.version and len(ds.log()) == 4
    assert ds.files().to_pylist() == fb.dataset("cats", version=1).files().to_pylist()
    assert ds.info() == {"license": "MIT"}
    assert ds.open("a/1.txt").read() == b"one"
    assert ds.log()[0].message == "revert to v1"
    assert ds.revert(reverted) == reverted, "reverting to the latest version is a no-op"
    assert len(ds.log()) == 4
    with pytest.raises(DatasetError):
        fb.dataset("cats", version=ds.log()[2].version).revert(1)
    with pytest.raises(DatasetError):
        ds.revert(99)


# --- diff --------------------------------------------------------------------


def test_diff(files: Path) -> None:
    ds = _three_versions(files)
    changes = {r["path"]: (r["change"], r["changed"]) for r in ds.diff(1).to_pylist()}
    assert changes == {
        "3.jpg": ("modified", "label"),
        "a/1.txt": ("modified", "content"),
        "a/2.txt": ("removed", ""),
        "new.txt": ("added", ""),
    }
    first, second, _ = (info.version for info in reversed(ds.log()))
    assert fb.dataset("cats", version=first).diff(second, prefix="a/").num_rows == 2
    assert ds.diff(second).num_rows == 0, "info-only changes do not change items"


# --- verify ------------------------------------------------------------------


def test_verify_finds_missing_and_corrupt_files(tmp_path: Path) -> None:
    ds = fb.dataset("cats")
    files = []
    for i in range(3):  # large enough that Lance keeps them outside the main data file
        path = tmp_path / f"{i}.bin"
        path.write_bytes(os.urandom(200_000))
        files.append(path)
    with ds.edit() as edit:
        for path in files:
            edit.add(path.name, path)
    with ds.edit() as edit:
        edit.remove("0.bin")
    report = ds.verify()
    assert report.ok and report.objects == 3  # shared by both versions: checked once
    assert verify(deep=True).ok
    stored = fb.storage.local_path("datasets/cats.lance")
    (blobs,) = [p for p in stored.rglob("*.blob")]
    content = bytearray(blobs.read_bytes())
    content[100] ^= 0xFF  # one byte of one file
    blobs.write_bytes(bytes(content))
    assert ds.verify().ok  # the sizes are still right
    report = verify(deep=True)
    assert len(report.corrupt) == 1 and report.corrupt[0].startswith("test/cats@v")
    assert report.missing == [] and not report.ok
    blobs.unlink()
    report = verify(deep=True)
    assert len(report.missing) == 3 and not report.ok
