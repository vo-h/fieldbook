"""Deleting versions and datasets, and freeing storage (fieldbook/data/cleanup.py)."""

import os
from datetime import timedelta
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.data import DatasetError, DatasetNotFoundError, cleanup, list_datasets, prune

NOW = timedelta(0)
SIZE = 100_000  # bytes a file, so that freed space is told apart from bookkeeping


def _versions(name: str) -> list[int]:
    return sorted(info.version for info in fb.dataset(name).log())


def _stored(name: str) -> int:
    return cleanup._size(fb.dataset(name))


def _file(folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(os.urandom(SIZE))
    return path


def _make_history(files: Path) -> list[bytes]:
    """Five versions; each replaces a.txt with different content. Returns the contents."""
    ds = fb.dataset("cats")
    contents = []
    for i in range(5):
        src = _file(files, f"v{i}.bin")
        contents.append(src.read_bytes())
        with ds.edit() as edit:
            if i:
                edit.remove("a.txt")
            edit.add("a.txt", src)
    return contents


# --- deleting versions -------------------------------------------------------


def test_delete_versions_frees_files_only_they_used(files: Path) -> None:
    contents = _make_history(files)
    before = _stored("cats")
    report = fb.dataset("cats").delete_versions([2, 3], grace=NOW)
    assert report.versions == {"test/cats": [2, 3]}
    assert 2 * SIZE <= report.freed < 3 * SIZE and _stored("cats") == before - report.freed
    assert _versions("cats") == [1, 4, 5]
    assert fb.dataset("cats", version=1).open("a.txt").read() == contents[0]
    assert fb.dataset("cats").open("a.txt").read() == contents[4]
    with pytest.raises(DatasetError):
        fb.dataset("cats", version=2).files()


def test_delete_versions_keeps_files_later_versions_use(files: Path) -> None:
    ds = fb.dataset("cats")
    kept = _file(files, "kept.bin")
    with ds.edit() as edit:
        edit.add("a.bin", kept)
    with ds.edit() as edit:
        edit.add("b.bin", _file(files, "more.bin"))
    report = ds.delete_versions([1], grace=NOW)
    assert report.freed < SIZE / 10  # a.bin is still in version 2: only bookkeeping went
    assert ds.open("a.bin").read() == kept.read_bytes()


def test_delete_versions_refusals(files: Path) -> None:
    _make_history(files)
    ds = fb.dataset("cats")
    ds.tag("keep", version=2)
    with pytest.raises(DatasetError, match="latest"):
        ds.delete_versions([5])
    with pytest.raises(DatasetError, match="tagged"):
        ds.delete_versions([1, 2])
    with pytest.raises(DatasetError, match="not found"):
        ds.delete_versions([99])
    with pytest.raises(DatasetError, match="used by runs"):
        ds.delete_versions([3], referenced={"test/cats": {3}})
    with pytest.raises(DatasetError):
        fb.dataset("cats", version=1).delete_versions([1])
    assert _versions("cats") == [1, 2, 3, 4, 5]


# --- deleting datasets -------------------------------------------------------


def test_delete_dataset_frees_its_files_and_no_others(files: Path) -> None:
    shared = _file(files, "shared.bin")
    with fb.dataset("cats").edit() as edit:
        edit.add("shared.bin", shared)
        edit.add("own.bin", _file(files, "own.bin"))
    with fb.dataset("dogs").edit() as edit:
        edit.add("shared.bin", shared)  # each dataset has its own copy
    report = fb.dataset("cats").delete(dry_run=True)
    assert fb.dataset("cats").exists and report.freed == 0 and report.dry_run
    report = fb.dataset("cats").delete(grace=NOW)
    assert report.datasets == ["test/cats"] and report.freed >= 2 * SIZE
    assert list_datasets() == ["dogs"]
    assert fb.dataset("dogs").open("shared.bin").read() == shared.read_bytes()
    with pytest.raises(DatasetNotFoundError):
        fb.dataset("cats").delete()


# --- pruning -----------------------------------------------------------------


def test_prune_keeps_latest_tagged_and_referenced(files: Path) -> None:
    _make_history(files)
    fb.dataset("cats").tag("keep", version=1)
    report = fb.dataset("cats").prune(
        keep_latest=2, retention=NOW, grace=NOW, referenced={"test/cats": {2}}
    )
    assert report.versions == {"test/cats": [3]}
    assert _versions("cats") == [1, 2, 4, 5]
    assert SIZE <= report.freed < 2 * SIZE


def test_prune_retention_window_and_dry_run(files: Path) -> None:
    contents = _make_history(files)
    assert prune(keep_latest=1, grace=NOW).versions == {}  # all within the last week
    before = _stored("cats")
    report = prune(keep_latest=1, retention=NOW, grace=NOW, dry_run=True)
    assert report.versions == {"test/cats": [1, 2, 3, 4]} and report.freed == 0
    assert _versions("cats") == [1, 2, 3, 4, 5] and _stored("cats") == before
    report = prune(keep_latest=1, retention=NOW, grace=NOW)
    assert _versions("cats") == [5] and report.freed >= 4 * SIZE
    assert _stored("cats") < 2 * SIZE
    assert fb.dataset("cats").open("a.txt").read() == contents[4]


def test_steps_of_edits_are_removed_with_no_version_lost(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    with ds.edit() as edit:  # a new column and a new file: the edit takes a step first
        edit.add("new.txt", files / "a" / "1.txt", label="cat")
    assert len(ds.index().versions()) == 3 and _versions("cats") == [1, 3]
    prune(grace=NOW)
    assert [v["version"] for v in ds.index().versions()] == [1, 3]
    assert fb.dataset("cats", version=1).files().num_rows == 3


def test_a_table_with_many_removed_rows_is_compacted(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        for i in range(10):
            edit.add(f"{i}.bin", _file(files, f"{i}.bin"))
    kept = ds.open("9.bin").read()
    with ds.edit() as edit:
        for i in range(8):
            edit.remove(f"{i}.bin")
    last = ds.version
    assert _stored("cats") > 10 * SIZE  # removed rows are only marked
    report = prune(keep_latest=1, retention=NOW, grace=NOW)
    assert report.compacted == ["test/cats"] and report.freed > 7 * SIZE
    assert _stored("cats") < 3 * SIZE
    # the compacted table is a version of its own (the same items), and the one before it
    # went by the same rules as any other
    assert ds.version > last and _versions("cats") == [ds.version]
    assert ds.log()[0].message.startswith("compacted")
    assert ds.open("9.bin").read() == kept and ds.files().num_rows == 2
    with ds.edit() as edit:  # and edits go on from there
        edit.add("new.bin", _file(files, "new.bin"))
    assert ds.files().num_rows == 3 and len(_versions("cats")) == 2


def test_an_interrupted_edit_leaves_nothing_behind(files: Path) -> None:
    with pytest.raises(RuntimeError):
        with fb.dataset("cats").edit() as edit:
            edit.add("a.txt", files / "a" / "1.txt")
            raise RuntimeError("interrupted")
    assert not fb.dataset("cats").exists and list_datasets() == []
    assert prune(grace=NOW).versions == {}


def test_compaction_keeps_every_item_as_it_was(files: Path) -> None:
    ds = fb.dataset("cats")
    contents = {f"f{i}.bin": os.urandom(SIZE) for i in range(12)}
    for name, content in contents.items():
        (files / name).write_bytes(content)
    with ds.edit("add") as edit:
        for name in contents:
            edit.add(name, files / name, label=name[1])
    with ds.edit("thin") as edit:
        for name in list(contents)[:6]:
            edit.remove(name)
        edit.update("f7.bin", label="changed")
        edit.move("f8.bin", "moved/f8.bin")
    before = ds.files().to_pylist()
    report = prune(ds, keep_latest=1, retention=NOW, grace=NOW, referenced={})
    assert report.compacted == ["test/cats"] and report.freed > 5 * SIZE
    assert all(f.num_deletions == 0 for f in ds.index().get_fragments())
    assert ds.files().to_pylist() == before  # paths, hashes, metadata, added_at
    kept = [row["path"] for row in before]
    assert ds.read_many(kept) == [contents[path.rsplit("/", 1)[-1]] for path in kept]
    assert fb.data.verify(deep=True).ok
    with ds.edit("after") as edit:  # and it is edited like any other table
        edit.update("f9.bin", label="again")
    assert ds.files(where="label = 'again'").num_rows == 1


def test_a_data_file_with_a_large_file_is_left_to_lance(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cleanup, "STREAM_ABOVE", SIZE // 2)  # every file here is "large"
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        for i in range(5):
            edit.add(f"f{i}.bin", _file(files, f"f{i}.bin"))
    with ds.edit() as edit:
        edit.remove("f0.bin")
        edit.remove("f1.bin")
    report = prune(ds, keep_latest=1, retention=NOW, grace=NOW, referenced={})
    assert report.compacted == ["test/cats"] and report.freed > SIZE
    assert ds.files().num_rows == 3 and fb.data.verify(deep=True).ok


def test_a_save_during_compaction_is_not_lost(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        for i in range(5):
            edit.add(f"f{i}.bin", _file(files, f"f{i}.bin"))
    with ds.edit() as edit:
        edit.remove("f0.bin")
        edit.remove("f1.bin")
    kept = cleanup._kept

    def kept_then_a_save(*args):  # another process saves while the rows are rewritten
        yield from kept(*args)
        with fb.dataset("cats").edit("meanwhile") as edit:
            edit.remove("f2.bin")

    monkeypatch.setattr(cleanup, "_kept", kept_then_a_save)
    report = prune(ds, keep_latest=10, retention=NOW, grace=NOW, referenced={})
    assert report.compacted == []  # given up; the next cleanup does it
    assert sorted(ds.files()["path"].to_pylist()) == ["f3.bin", "f4.bin"]
    monkeypatch.setattr(cleanup, "_kept", kept)
    assert prune(ds, keep_latest=1, retention=NOW, grace=NOW, referenced={}).compacted
    assert sorted(ds.files()["path"].to_pylist()) == ["f3.bin", "f4.bin"]
    assert fb.data.verify(deep=True).ok
