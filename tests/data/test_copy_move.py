from pathlib import Path

import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import ConflictError, DatasetError

runner = CliRunner()


def _raw(files: Path) -> fb.data.Dataset:
    ds = fb.dataset("raw")
    with ds.edit() as edit:
        edit.add_dir(files, label="cat")
        edit.update("3.jpg", label="dog")
    return ds


def _rows(ds: fb.data.Dataset) -> dict[str, dict]:
    return {r["path"]: r for r in ds.files().to_pylist()}


def test_copy_from_reuses_objects_and_metadata(files: Path, monkeypatch) -> None:
    raw = _raw(files)
    clean = fb.dataset("clean")
    with clean.edit() as edit:
        assert edit.copy_from(raw, "label = 'cat'", prefix="train/", split="train") == 2
    rows, source = _rows(clean), _rows(raw)
    assert sorted(rows) == ["train/a/1.txt", "train/a/2.txt"]
    assert rows["train/a/1.txt"]["hash"] == source["a/1.txt"]["hash"]
    assert rows["train/a/1.txt"]["label"] == "cat" and rows["train/a/1.txt"]["split"] == "train"
    assert clean.log()[0].inputs == ("test/raw@1",)
    with clean.open("train/a/1.txt") as f:
        assert f.read() == b"one"


def test_projects_do_not_read_each_others_datasets(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sharing.md §4: no copies or inputs across projects."""
    monkeypatch.setenv("FIELDBOOK_PROJECT", "audio")
    audio = fb.dataset("clips")
    with audio.edit() as edit:
        edit.add_dir(files)
    monkeypatch.setenv("FIELDBOOK_PROJECT", "test")
    assert fb.data.list_datasets() == []
    with pytest.raises(DatasetError, match="one project"), fb.dataset("mine").edit() as edit:
        edit.copy_from(audio)
    with pytest.raises(DatasetError, match="another project"):
        fb.dataset("mine").edit(inputs=["audio/clips@1"])


def test_copied_files_survive_deleting_the_source(files: Path) -> None:
    raw = _raw(files)
    with fb.dataset("clean").edit() as edit:
        edit.copy_from(raw)
    raw._remove_table()  # the copy has its own bytes
    clean = fb.dataset("clean")
    assert clean.read_many(["a/1.txt", "3.jpg"]) == [b"one", b"\xff\xd8three"]
    assert _rows(clean)["a/1.txt"]["size"] == 3 and len(_rows(clean)["a/1.txt"]["hash"]) == 64


def test_copy_from_errors(files: Path) -> None:
    raw = _raw(files)
    clean = fb.dataset("clean")
    with pytest.raises(DatasetError, match="no items"):
        clean.edit().copy_from(raw, "label = 'bird'")
    with clean.edit() as edit:
        edit.copy_from(raw)
    with pytest.raises(DatasetError, match="already exists"):
        clean.edit().copy_from(raw)
    with pytest.raises(DatasetError, match="not found"):
        clean.edit().copy_from(fb.dataset("nope"))


def test_move_items_and_directories(files: Path) -> None:
    raw = _raw(files)
    before = _rows(raw)
    with raw.edit() as edit:
        assert edit.move("a/", "b/") == 2
        assert edit.move("3.jpg", "img/3.jpg") == 1
    after = _rows(raw)
    assert sorted(after) == ["b/1.txt", "b/2.txt", "img/3.jpg"]
    assert after["img/3.jpg"]["hash"] == before["3.jpg"]["hash"]
    assert after["img/3.jpg"]["added_at"] == before["3.jpg"]["added_at"]
    assert after["img/3.jpg"]["label"] == "dog"


def test_move_errors_and_conflicts(files: Path) -> None:
    raw = _raw(files)
    with pytest.raises(DatasetError, match="no items"):
        raw.edit().move("zzz/", "b/")
    with pytest.raises(ValueError, match="both"):
        raw.edit().move("a/", "b")
    with pytest.raises(DatasetError, match="already exists"):
        raw.edit().move("a/1.txt", "a/2.txt")
    with raw.edit() as edit:  # a move carries changes made earlier in the edit
        edit.update("3.jpg", label="x")
        edit.move("3.jpg", "4.jpg")
    assert _rows(raw)["4.jpg"]["label"] == "x" and "3.jpg" not in _rows(raw)
    first, second = raw.edit(), raw.edit()
    first.move("4.jpg", "x.jpg")
    second.update("4.jpg", label="bird")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()


def run_cli(*args: str, ok: bool = True) -> str:
    result = runner.invoke(app, list(args))
    assert (result.exit_code == 0) == ok, result.output
    return result.output


def test_cli_copy_and_move(files: Path) -> None:
    _raw(files)
    out = run_cli("data", "add", "clean", "--copy", "raw@1", "--where", "label = 'dog'")
    assert "added 1 items, v1" in out
    out = run_cli("data", "add", "clean", str(files / "a"), "--copy", "raw", "--prefix", "x/")
    assert "added 5 items, v2" in out
    assert "moved 2 items" in run_cli("data", "update", "clean", "--move", "x/a/", "y/")
    assert "y/1.txt" in run_cli("data", "ls", "clean")
    assert "--where applies to --copy" in run_cli(
        "data", "add", "clean", str(files), "--where", "x", ok=False
    )
    assert "or --copy" in run_cli("data", "add", "clean", ok=False)


def test_operations_compose_in_one_edit(files: Path, tmp_path: Path) -> None:
    raw = _raw(files)
    other = tmp_path / "other.txt"
    other.write_text("other")
    with raw.edit() as edit:
        edit.remove("a/1.txt")
        with pytest.raises(DatasetError, match="not in raw"):
            edit.update("a/1.txt", label="x")  # bug #5: removed items stay removed
        edit.move("a/2.txt", "b.txt")
        with pytest.raises(DatasetError, match="not in raw"):
            edit.update("a/2.txt", label="x")  # the old path of a moved item is gone
        edit.update("b.txt", label="moved")
        edit.add("new.txt", other)
        edit.move("new.txt", "c.txt")  # moving an item added in this edit
        edit.move("c.txt", "d.txt")
        edit.remove("3.jpg")
        edit.add("3.jpg", other)  # replace by remove + add
        with pytest.raises(DatasetError, match="not in raw"):
            edit.remove("zzz.txt")
    rows = _rows(raw)
    assert sorted(rows) == ["3.jpg", "b.txt", "d.txt"]
    assert rows["b.txt"]["label"] == "moved"
    with raw.open("3.jpg") as f, raw.open("d.txt") as g:
        assert f.read() == g.read() == b"other"


def test_add_then_remove_leaves_no_trace(files: Path) -> None:
    raw = _raw(files)
    with raw.edit() as edit:
        edit.add("tmp.txt", files / "a" / "1.txt")
        edit.remove("tmp.txt")
    assert edit.version == 1 and raw.version == 1  # nothing to commit


def test_remove_conflicts_with_concurrent_update(files: Path) -> None:
    raw = _raw(files)
    first, second = raw.edit(), raw.edit()
    first.remove("3.jpg")
    second.update("3.jpg", label="bird")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()
