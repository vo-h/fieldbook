"""Saves write only their changes, and never a file's bytes again (data.md §6), and
concurrent saves still conflict at path level."""

import importlib
import os
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.data import ConflictError


def _index_files(storage_root: Path, name: str) -> dict[str, int]:
    folder = storage_root / "test" / "datasets" / f"{name}.lance" / "data"
    return {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob("*") if p.is_file()}


def _big(tmp_path: Path, n: int = 2000) -> fb.data.Dataset:
    src = tmp_path / "big"
    src.mkdir()
    (src / "x.txt").write_bytes(os.urandom(2000))  # 2 KB a file: 4 MB of files in all
    ds = fb.dataset("big")
    with ds.edit(message="first") as edit:
        edit.add_many([(f"items/{i:05d}.txt", src / "x.txt") for i in range(n)], label="raw")
    return ds


def test_a_save_writes_only_its_changes(tmp_path: Path, storage_root: Path, files: Path) -> None:
    ds = _big(tmp_path)
    before = _index_files(storage_root, "big")
    with ds.edit(message="small", inputs=["hf://datasets/x"]) as edit:
        edit.add("new/1.txt", files / "a" / "1.txt", label="new")
        edit.update("items/00001.txt", label="fixed")
        edit.remove("items/00002.txt")
        edit.move("items/00003.txt", "moved/3.txt")
    after = _index_files(storage_root, "big")
    assert set(before) <= set(after)  # nothing rewritten
    written = sum(size for name, size in after.items() if name not in before)
    # the new file, two columns of values, and a few marks: none of the 4 MB of files
    assert written < sum(before.values()) / 10, (written, before)
    assert len(ds.log()) == 2  # one edit, one version
    rows = {r["path"]: r for r in ds.files().to_pylist()}
    assert len(rows) == 2000 and "items/00002.txt" not in rows and "items/00003.txt" not in rows
    assert rows["items/00001.txt"]["label"] == "fixed" and rows["new/1.txt"]["label"] == "new"
    assert rows["moved/3.txt"]["label"] == "raw"
    entry = ds.log()[0]
    assert (entry.message, entry.author, entry.inputs) == ("small", "tester", ("hf://datasets/x",))
    with ds.open("new/1.txt") as f:
        assert f.read() == b"one"


def test_info_and_schema_are_kept_by_small_saves(tmp_path: Path, files: Path) -> None:
    ds = fb.dataset("described")
    with ds.edit() as edit:
        edit.add_dir(files, label="cat")
        edit.set_info(description="pets")
        edit.set_schema("columns:\n  label:\n    dtype: str\n    nullable: false\n")
    with ds.edit() as edit:
        edit.update("3.jpg", label="dog")
    assert ds.info() == {"description": "pets"} and ds.schema() is not None
    with pytest.raises(fb.data.ValidationError), ds.edit() as edit:
        edit.add("4.txt", files / "a" / "1.txt")  # no label: checked on the whole version


def test_concurrent_saves_of_the_same_new_path_conflict(files: Path) -> None:
    """Lance alone would store both rows; the save refuses the second instead."""
    ds = fb.dataset("labels")
    with ds.edit() as edit:
        edit.add_dir(files)
    first, second = ds.edit(), ds.edit()
    first.add("new.txt", files / "a" / "1.txt")
    second.add("new.txt", files / "a" / "2.txt")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()
    assert [r["path"] for r in ds.files().to_pylist()].count("new.txt") == 1


def test_concurrent_saves_of_different_paths_both_land(files: Path) -> None:
    ds = fb.dataset("labels")
    with ds.edit() as edit:
        edit.add_dir(files)
    edits = [ds.edit() for _ in range(3)]
    for i, edit in enumerate(edits):
        edit.add(f"new/{i}.txt", files / "a" / "1.txt")
    edits[0].update("3.jpg", label="dog")
    versions = [edit.commit() for edit in edits]
    assert versions == sorted(versions) and len(ds.log()) == 4  # one version an edit
    paths = [r["path"] for r in ds.files().to_pylist()]
    assert sorted(paths) == sorted({*paths}) and len(paths) == 6


def test_emptying_and_refilling_a_dataset(files: Path) -> None:
    ds = fb.dataset("cycle")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    with ds.edit() as edit:
        edit.remove("a.txt")
    with ds.edit() as edit:
        edit.add("b.txt", files / "a" / "2.txt")
    assert [r["path"] for r in ds.files().to_pylist()] == ["b.txt"]


def test_removing_many_items_marks_them_and_keeps_earlier_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above a number of paths they are found by reading the path column, not by a filter
    naming each; either way the rows are only marked, never rewritten."""
    dataset_module = importlib.import_module("fieldbook.data.dataset")
    monkeypatch.setattr(dataset_module, "REWRITE_ABOVE_REMOVALS", 3)
    ds = _big(tmp_path, 20)

    def data_files() -> set[str]:
        return {f.path for fragment in ds._lance().get_fragments() for f in fragment.data_files()}

    original = data_files()
    with ds.edit() as edit:
        for i in range(3):
            edit.remove(f"items/{i:05d}.txt")
    with ds.edit() as edit:
        for i in range(3, 7):
            edit.remove(f"items/{i:05d}.txt")
    assert original <= data_files()  # nothing rewritten
    left = [r["path"] for r in ds.files().to_pylist()]
    assert left == [f"items/{i:05d}.txt" for i in range(7, 20)]
    assert fb.dataset("big", version=1).files().num_rows == 20  # earlier versions are untouched
    ds.revert(1)
    assert ds.files().num_rows == 20 and len(ds.log()) == 4
    assert original == data_files()  # a revert copies nothing


def test_a_clash_reads_only_the_touched_items(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: redoing an edit on another writer's version does not load the whole index."""
    import lance

    ds = fb.dataset("labels")
    with ds.edit() as edit:
        edit.add_dir(files)
    first, second = ds.edit(), ds.edit()
    first.add("one.txt", files / "a" / "1.txt")
    second.add("two.txt", files / "a" / "2.txt")
    first.commit()
    reads = []
    whole = lance.LanceDataset.to_table

    def recording(self, *args, **kwargs):
        reads.append((kwargs.get("filter"), kwargs.get("limit")))
        return whole(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "to_table", recording)
    assert second.commit() > 2
    assert reads and all(filter or limit == 1 for filter, limit in reads), reads
    assert "path IN ('two.txt')" in [filter for filter, _ in reads]
    monkeypatch.undo()
    assert sorted(r["path"] for r in ds.files().to_pylist())[-2:] == ["one.txt", "two.txt"]


def test_an_edit_loads_paths_and_only_the_rows_it_touches(
    tmp_path: Path, files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1: opening an edit reads the path column; full rows are fetched by position."""
    import lance

    ds = _big(tmp_path, 50)
    with ds.edit() as edit:  # a small save: rows marked deleted, positions shift
        edit.remove("items/00003.txt")
        edit.remove("items/00010.txt")
    reads = []
    whole = lance.LanceDataset.to_table

    def recording(self, *args, **kwargs):
        reads.append((kwargs.get("columns"), kwargs.get("filter"), kwargs.get("limit")))
        return whole(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "to_table", recording)
    with ds.edit() as edit:
        edit.update("items/00011.txt", label="fixed")
        edit.move("items/00020.txt", "moved/20.txt")
        edit.add("new.txt", files / "a" / "1.txt", label="new")
    # never the whole table: the path column, or the few rows the edit names
    assert all(columns == ["path"] or filter for columns, filter, _ in reads), reads
    assert not any("data" in (columns or ["data"]) for columns, _, _ in reads), reads
    monkeypatch.undo()
    rows = {r["path"]: r["label"] for r in ds.files().to_pylist()}
    assert len(rows) == 49 and rows["items/00011.txt"] == "fixed" and rows["moved/20.txt"] == "raw"
    assert rows["items/00012.txt"] == "raw" and "items/00020.txt" not in rows
