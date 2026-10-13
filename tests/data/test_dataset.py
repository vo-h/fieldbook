from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.data import ConflictError, DatasetError, DatasetNotFoundError, list_datasets


def test_add_dir_creates_version(files: Path) -> None:
    ds = fb.dataset("cats")
    assert not ds.exists
    with ds.edit(message="initial") as edit:
        assert edit.add_dir(files, prefix="train/") == 3
    assert edit.version == 1
    rows = ds.index().to_table().to_pylist()
    assert [r["path"] for r in rows] == ["train/3.jpg", "train/a/1.txt", "train/a/2.txt"]
    assert rows[0]["mime_type"] == "image/jpeg"
    assert rows[1]["size"] == 3
    assert list_datasets() == ["cats"]


def test_read_items(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    with ds.open("a/1.txt") as f:
        assert f.read() == b"one"
    assert ds.read_many(["3.jpg", "a/2.txt", "a/1.txt"]) == [b"\xff\xd8three", b"two", b"one"]
    with pytest.raises(KeyError):
        ds.open("missing.txt")
    with pytest.raises(KeyError):
        ds.read_many(["a/1.txt", "missing.txt"])
    with pytest.raises(ValueError):
        ds.open("a/1.txt", "wb")


def test_remove_update_and_user_columns(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("x.txt", files / "a" / "1.txt", label="tabby")
        edit.add("y.txt", files / "a" / "2.txt")
    with ds.edit() as edit:
        edit.update("y.txt", label="siamese", score=0.9)
        edit.remove("x.txt")
    rows = ds.files().to_pylist()
    assert len(rows) == 1
    assert rows[0]["label"] == "siamese"
    assert rows[0]["score"] == 0.9
    old = fb.dataset("cats", version=1).index().to_table().to_pylist()
    assert {r["path"]: r["label"] for r in old} == {"x.txt": "tabby", "y.txt": None}


def test_errors(files: Path) -> None:
    ds = fb.dataset("cats")
    with pytest.raises(DatasetNotFoundError):
        ds.index()
    with pytest.raises(ValueError):
        fb.dataset("bad/name")
    with ds.edit() as edit:
        for bad in ["../x", "/abs", "a//b", "", "a\\b"]:
            with pytest.raises(ValueError):
                edit.add(bad, files / "3.jpg")
        with pytest.raises(ValueError):
            edit.add("ok.jpg", files / "3.jpg", hash="x")
        edit.add("ok.jpg", files / "3.jpg")
    with pytest.raises(DatasetError):
        with ds.edit() as edit:
            edit.remove("missing.jpg")
    with pytest.raises(DatasetError):
        fb.dataset("cats", version=1).edit()


def test_failed_edit_does_not_commit(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    with pytest.raises(RuntimeError):
        with ds.edit() as edit:
            edit.add("b.txt", files / "a" / "2.txt")
            raise RuntimeError("boom")
    assert ds.version == 1


def test_empty_edit_does_not_commit(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    with ds.edit() as edit:
        pass
    assert edit.version == 1
    assert ds.version == 1


def test_tags_and_log(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = fb.dataset("cats")
    with ds.edit(message="first") as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    monkeypatch.setenv("PLATFORM_RUN_ID", "run-42")
    with ds.edit(message="second") as edit:
        edit.add("b.txt", files / "a" / "2.txt")
    ds.tag("raw", version=1)
    ds.tag("latest-clean")
    with pytest.raises(DatasetError):
        ds.tag("raw")
    assert ds.tags() == {"raw": 1, "latest-clean": 2}

    log = ds.log()
    assert [v.version for v in log] == [2, 1]
    assert (log[0].message, log[0].run_id, log[0].rows, log[0].tags) == (
        "second",
        "run-42",
        2,
        ("latest-clean",),
    )
    assert (log[1].message, log[1].author, log[1].run_id, log[1].tags) == (
        "first",
        "tester",
        "",
        ("raw",),
    )
    assert fb.dataset("cats", version="raw").index().count_rows() == 1


def test_non_conflicting_concurrent_edits_are_rebased(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("base.txt", files / "a" / "1.txt")
    first = ds.edit()
    second = ds.edit()
    first.add("one.txt", files / "a" / "1.txt")
    second.add("two.txt", files / "a" / "2.txt")
    assert first.commit() == 2
    assert second.commit() == 3
    paths = ds.files()["path"].to_pylist()
    assert paths == ["base.txt", "one.txt", "two.txt"]  # sorted by path


def test_conflicting_concurrent_edits_fail(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("x.txt", files / "a" / "1.txt")
    first = ds.edit()
    second = ds.edit()
    first.update("x.txt", label="a")
    second.update("x.txt", label="b")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()
    assert ds.index().to_table().to_pylist()[0]["label"] == "a"


def test_concurrent_creation_conflicts_only_on_same_paths(files: Path) -> None:
    first = fb.dataset("new").edit()
    second = fb.dataset("new").edit()
    first.add("a.txt", files / "a" / "1.txt")
    second.add("b.txt", files / "a" / "2.txt")
    first.commit()
    second.commit()
    assert fb.dataset("new").index().count_rows() == 2


def test_move_and_remove_tags(files: Path) -> None:
    ds = fb.dataset("cats")
    for name in ["1.txt", "2.txt"]:
        with ds.edit() as edit:
            edit.add(name, files / "a" / name)
    ds.tag("best", version=1)
    with pytest.raises(DatasetError):
        ds.tag("best", version=2)
    ds.tag("best", version=2, move=True)
    assert ds.tags() == {"best": 2}
    with pytest.raises(DatasetError):
        ds.tag("other", version=99)
    ds.untag("best")
    assert ds.tags() == {}
    with pytest.raises(DatasetError):
        ds.untag("best")


def _split_dataset(tmp_path: Path) -> None:
    """train: 3 cats + 1 dog, val: 1 cat, plus a top-level README."""
    src = tmp_path / "split"
    layout = {
        "train/cat/1.jpg": 10,
        "train/cat/2.jpg": 20,
        "train/cat/3.jpg": 30,
        "train/dog/1.jpg": 40,
        "val/cat/1.jpg": 50,
        "README.md": 5,
    }
    for path, size in layout.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_bytes(b"x" * size)
    with fb.dataset("pets").edit() as edit:
        edit.add_dir(src)
        edit.update("train/cat/1.jpg", label="tabby")
        edit.update("train/cat/2.jpg", label="tabby")


def test_files_prefix_and_where(tmp_path: Path) -> None:
    _split_dataset(tmp_path)
    ds = fb.dataset("pets")
    assert ds.files(prefix="val/")["path"].to_pylist() == ["val/cat/1.jpg"]
    assert ds.files(where="size >= 30")["path"].to_pylist() == [
        "train/cat/3.jpg",
        "train/dog/1.jpg",
        "val/cat/1.jpg",
    ]
    assert ds.files(prefix="train/", where="label = 'tabby'").num_rows == 2


def test_stats(tmp_path: Path) -> None:
    _split_dataset(tmp_path)
    ds = fb.dataset("pets")
    assert ds.stats().to_pylist() == [{"files": 6, "bytes": 155}]
    assert ds.stats(depth=1).to_pylist() == [
        {"dir": ".", "files": 1, "bytes": 5},
        {"dir": "train", "files": 4, "bytes": 100},
        {"dir": "val", "files": 1, "bytes": 50},
    ]
    by_class = ds.stats(prefix="train/", depth=1).to_pylist()
    assert by_class == [
        {"dir": "cat", "files": 3, "bytes": 60},
        {"dir": "dog", "files": 1, "bytes": 40},
    ]
    by_label = ds.stats(prefix="train/cat/", by=["label"]).to_pylist()
    assert {r["label"]: r["files"] for r in by_label} == {"tabby": 2, None: 1}


def test_rows_with_different_fields_in_one_edit(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
        edit.add("b.txt", files / "a" / "2.txt", label="tabby")
        edit.add("c.txt", files / "3.jpg", score=1)
    rows = {r["path"]: r for r in ds.index().to_table().to_pylist()}
    assert (rows["a.txt"]["label"], rows["b.txt"]["label"], rows["c.txt"]["score"]) == (
        None,
        "tabby",
        1,
    )


def test_add_refuses_existing_paths(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
        with pytest.raises(DatasetError, match="already added"):
            edit.add("a.txt", files / "a" / "2.txt")
    with pytest.raises(DatasetError, match="already exists"):
        with ds.edit() as edit:
            edit.add("a.txt", files / "a" / "2.txt")
    with ds.edit() as edit:  # explicit remove + add replaces content
        edit.remove("a.txt")
        edit.add("a.txt", files / "a" / "2.txt")
    with ds.open("a.txt") as f:
        assert f.read() == b"two"
    with ds.edit() as edit:  # adding then removing a new path is a no-op for that path
        edit.add("b.txt", files / "a" / "1.txt")
        edit.remove("b.txt")
    assert ds.files()["path"].to_pylist() == ["a.txt"]


def test_concurrent_adds_of_same_path_conflict(files: Path) -> None:
    first = fb.dataset("cats").edit()
    second = fb.dataset("cats").edit()
    first.add("a.txt", files / "a" / "1.txt")
    second.add("a.txt", files / "a" / "2.txt")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()


def test_update_replaces_content_and_keeps_metadata(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt", label="tabby")
    with ds.edit() as edit:
        edit.update("a.txt", files / "a" / "2.txt")
    with ds.edit() as edit:
        edit.update("a.txt", files / "3.jpg", label="siamese")
    rows = [fb.dataset("cats", version=v).files().to_pylist()[0] for v in (1, 2, 3)]
    assert [(r["size"], r["label"]) for r in rows] == [(3, "tabby"), (3, "tabby"), (7, "siamese")]
    assert rows[0]["hash"] != rows[1]["hash"]
    assert rows[0]["added_at"] == rows[2]["added_at"]
    for version, content in ((1, b"one"), (2, b"two"), (3, b"\xff\xd8three")):
        with fb.dataset("cats", version=version).open("a.txt") as f:
            assert f.read() == content
    with pytest.raises(DatasetError, match="not in"):
        with ds.edit() as edit:
            edit.update("missing.txt", files / "3.jpg")


def test_add_dir_skips_hidden_files(files: Path) -> None:
    (files / ".DS_Store").write_bytes(b"x")
    (files / ".git").mkdir()
    (files / ".git" / "HEAD").write_text("ref")
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        assert edit.add_dir(files) == 3
    with fb.dataset("all").edit() as edit:
        assert edit.add_dir(files, include_hidden=True) == 5


def test_values_are_cast_to_existing_column_types(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt", label="tabby", score=0.5, count=1)
    with ds.edit() as edit:
        edit.update("a.txt", label=5, score=1)
        edit.add("b.txt", files / "a" / "2.txt", label=7, count=2)
    rows = {r["path"]: r for r in ds.files().to_pylist()}
    assert (rows["a.txt"]["label"], rows["a.txt"]["score"]) == ("5", 1.0)
    assert (rows["b.txt"]["label"], rows["b.txt"]["count"]) == ("7", 2)
    for bad in ({"count": "many"}, {"count": 1.5}):
        with pytest.raises(DatasetError, match="has type int64"):
            with ds.edit() as edit:
                edit.update("a.txt", **bad)
    assert ds.version == 2


def test_invalid_filters_and_columns(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    with pytest.raises(DatasetError, match="invalid filter"):
        ds.files(where="nope = 1")
    with pytest.raises(DatasetError, match="unknown columns"):
        ds.stats(by=["nope"])


def test_edit_is_single_use(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    for action in (edit.commit, lambda: edit.add("b.txt", files / "a" / "2.txt")):
        with pytest.raises(DatasetError, match="already committed"):
            action()
    assert ds.version == 1


def test_reads_see_new_versions_and_pinned_views_stay(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add("a.txt", files / "a" / "1.txt")
    pinned = fb.dataset("cats", version=1)
    assert pinned.open("a.txt").read() == b"one"
    with ds.edit() as edit:
        edit.update("a.txt", files / "a" / "2.txt")
    assert ds.open("a.txt").read() == b"two"
    assert pinned.open("a.txt").read() == b"one"


def test_files_are_in_the_table_and_not_listed_with_it(files: Path, storage_root: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    assert "data" not in ds.files().column_names  # the bytes are read with open, read_many
    assert ds.index().schema.names[-1] == "data"
    assert not (storage_root / "test" / "objects").exists()  # no stored object per file
    with pytest.raises(ValueError, match="reserved"):
        ds.edit().update("3.jpg", data=b"x")
    listed = ds.files()
    assert listed["size"].to_pylist() == [7, 3, 3] and len(listed["hash"][0].as_py()) == 64


@pytest.mark.parametrize("project", ["../outside", "a/b", ".hidden"])
def test_invalid_project_names_are_rejected(project: str, monkeypatch) -> None:
    monkeypatch.setenv("FIELDBOOK_PROJECT", project)
    with pytest.raises(ValueError, match="invalid project name"):
        fb.dataset("cats")


def test_prefix_filter_matches_python_filtering(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files, label="cat")
        edit.add("it's/x.txt", files / "a" / "1.txt", label="dog")  # quote in the prefix
        edit.add("a%/y.txt", files / "a" / "1.txt", label="dog")  # LIKE wildcard in the prefix
    for prefix in ("a/", "it's/", "a%/", "a", "zzz", ""):
        expected = sorted(p for p in ds.files()["path"].to_pylist() if p.startswith(prefix))
        assert sorted(ds.files(prefix=prefix)["path"].to_pylist()) == expected, prefix
    assert ds.files(prefix="a", where="label = 'dog'")["path"].to_pylist() == ["a%/y.txt"]
    with pytest.raises(DatasetError, match="invalid filter"):
        ds.files(prefix="a/", where="nope = 1")


def test_history_opens_the_dataset_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C6: on local disk the history opens the dataset once (bucket: test_cloud_storage)."""
    import importlib

    dataset_module = importlib.import_module("fieldbook.data.dataset")

    ds = fb.dataset("many")
    for i in range(12):
        (tmp_path / f"{i}.txt").write_text(str(i))
        with ds.edit(f"v{i + 1}") as edit:
            edit.add(f"{i}.txt", tmp_path / f"{i}.txt")
    ds.tag("first", 1)
    ds.tag("latest", 12)
    opens = []
    real_open = dataset_module._open
    monkeypatch.setattr(
        dataset_module, "_open", lambda *a, **k: opens.append(a) or real_open(*a, **k)
    )
    history = fb.dataset("many").log()
    assert len(opens) == 1
    assert [v.version for v in history] == list(range(12, 0, -1))
    assert [v.message for v in history][:2] == ["v12", "v11"] and history[0].tags == ("latest",)
    assert history[-1].tags == ("first",) and history[-1].rows == 1
    opens.clear()
    assert fb.dataset("many")._version_info(3).message == "v3" and len(opens) == 1


def test_a_tag_created_at_the_same_instant_is_refused_plainly(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B12: Lance refuses the second creation; the message is fieldbook's own."""
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    ds.tag("release")
    tags = type(ds._lance().tags)
    monkeypatch.setattr(tags, "list", lambda self: {})  # as before the other person's tag
    with pytest.raises(fb.data.DatasetError, match="tag 'release' already exists on cats"):
        ds.tag("release")


def test_a_large_file_is_copied_in_by_lance(tmp_path: Path, monkeypatch) -> None:
    """Above `STREAM_ABOVE` a file is not read into memory: Lance is given its address."""
    import hashlib
    import os
    import sys

    dataset = sys.modules["fieldbook.data.dataset"]
    monkeypatch.setattr(dataset, "STREAM_ABOVE", 1000)
    big, small = os.urandom(300_000), b"small"
    (tmp_path / "a big file.bin").write_bytes(big)  # a space in its name
    (tmp_path / "small.txt").write_bytes(small)
    blobs = []
    real = dataset.lance.blob_array
    monkeypatch.setattr(dataset.lance, "blob_array", lambda v: blobs.extend(v) or real(v))
    ds = fb.dataset("big")
    with ds.edit() as edit:
        edit.add("big.bin", tmp_path / "a big file.bin", label="x")
        edit.add("small.txt", tmp_path / "small.txt")
    assert sorted(type(b).__name__ for b in blobs) == ["bytes", "str"]  # address, not bytes
    row = ds.files(where="path = 'big.bin'").to_pylist()[0]
    assert (row["size"], row["hash"]) == (300_000, hashlib.sha256(big).hexdigest())
    assert ds.read_many(["big.bin", "small.txt"]) == [big, small]
    (tmp_path / "a big file.bin").unlink()  # the dataset has its own copy
    (tmp_path / "second.bin").write_bytes(big[::-1])
    with ds.edit() as edit:  # and on a save to an existing table
        edit.add("second.bin", tmp_path / "second.bin")
    assert ds.read_many(["second.bin", "big.bin"]) == [big[::-1], big]
    assert fb.data.verify(deep=True).ok
