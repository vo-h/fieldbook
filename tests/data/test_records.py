"""Records datasets: tabular data as the rows of the dataset's table (data.md §4)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import DatasetError, ValidationError

runner = CliRunner()


def run_cli(*args: str, ok: bool = True) -> str:
    result = runner.invoke(app, list(args))
    assert (result.exit_code == 0) == ok, result.output
    return result.output


def _rows(output: str) -> list[list[str]]:
    return [line.split() for line in output.splitlines()]


@pytest.fixture
def sales(tmp_path: Path) -> fb.data.Dataset:
    pq.write_table(pa.table({"id": [1, 2], "label": ["cat", "dog"]}), tmp_path / "p1.parquet")
    pq.write_table(pa.table({"id": [3], "label": ["cat"], "extra": [1.5]}), tmp_path / "p2.parquet")
    ds = fb.dataset("sales")
    with ds.edit() as edit:
        edit.add_records(tmp_path / "p1.parquet")
    with ds.edit() as edit:
        edit.add_records(tmp_path / "p2.parquet")
    return ds


def _sorted(ds: fb.data.Dataset, **options) -> list[dict]:
    return sorted(ds.to_table(**options).to_pylist(), key=lambda row: row["id"])


def test_rows_are_imported_and_columns_reconciled(sales: fb.data.Dataset) -> None:
    assert sales.kind == "records"
    assert _sorted(sales) == [
        {"id": 1, "label": "cat", "extra": None},
        {"id": 2, "label": "dog", "extra": None},
        {"id": 3, "label": "cat", "extra": 1.5},
    ]
    # the column the second file brought is a step of that save, not a version: a
    # version number is skipped
    assert [v.version for v in sales.log()] == [3, 1]
    assert fb.dataset("sales", 1).to_table().column_names == ["id", "label"]
    assert sales.to_table("label = 'cat'", columns=["id"], limit=1).to_pylist() == [{"id": 1}]


def test_delimited_text_of_any_delimiter(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("id,name\n1,x\n2,y\n")
    (tmp_path / "b.tsv").write_text("id\tname\n3\tz\n")
    (tmp_path / "c.csv").write_text("id;name\n4;w\n")  # found from the content
    (tmp_path / "d.jsonl").write_text('{"id": 5, "name": "v"}\n')
    (tmp_path / "e.txt").write_text("id|name\n6|u\n")
    ds = fb.dataset("mixed")
    with ds.edit() as edit:
        for name in ("a.csv", "b.tsv", "c.csv", "d.jsonl"):
            edit.add_records(tmp_path / name)
        edit.add_records(tmp_path / "e.txt", delimiter="|")
    assert edit.rows_added == 6
    assert "".join(r["name"] for r in _sorted(ds)) == "xyzwvu"
    with pytest.raises(DatasetError, match="not a table"), ds.edit() as edit:
        edit.add_records(tmp_path / "e.txt")


def test_rows_from_memory(sales: fb.data.Dataset) -> None:
    import pandas as pd

    with sales.edit() as edit:
        edit.add_records([{"id": 4, "label": "bird"}])
        edit.add_records(pa.table({"id": [5]}))
        edit.add_records(pd.DataFrame({"id": [6], "extra": [2.0]}))
    assert [r["id"] for r in _sorted(sales)] == [1, 2, 3, 4, 5, 6]
    with pytest.raises(DatasetError, match="'id' in sales has type int64"), sales.edit() as edit:
        edit.add_records([{"id": "seven"}])


def test_remove_and_update_rows_in_one_version(sales: fb.data.Dataset) -> None:
    with sales.edit("clean") as edit:
        edit.remove_records("id = 2")
        edit.update_records("label = 'cat'", label="feline", checked=True)
        edit.add_records([{"id": 9, "label": "bird"}])
    assert _sorted(sales) == [
        {"id": 1, "label": "feline", "extra": None, "checked": True},
        {"id": 3, "label": "feline", "extra": 1.5, "checked": True},
        {"id": 9, "label": "bird", "extra": None, "checked": None},
    ]
    assert [(v.version, v.message) for v in sales.log()][0] == (sales.version, "clean")
    assert len(sales.log()) == 3  # one version, whatever steps it took
    before = sales.log()[1].version
    assert len(_sorted(fb.dataset("sales", before))) == 3  # the earlier version is whole
    sales.revert(before)
    assert [r["label"] for r in _sorted(sales)] == ["cat", "dog", "cat"]
    with pytest.raises(DatasetError, match="invalid filter"), sales.edit() as edit:
        edit.remove_records("no_such_column = 1")


def test_a_dataset_keeps_its_kind(sales: fb.data.Dataset, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    with pytest.raises(DatasetError, match="holds records"), sales.edit() as edit:
        edit.add("a.txt", tmp_path / "a.txt")
    for read in (sales.files, lambda: sales.open("a"), lambda: sales.read_many(["a"])):
        with pytest.raises(DatasetError, match="holds records"):
            read()
    files = fb.dataset("files")
    with files.edit() as edit:
        edit.add("a.txt", tmp_path / "a.txt")
    assert files.kind == "files"
    with pytest.raises(DatasetError, match="holds files"), files.edit() as edit:
        edit.add_records([{"id": 1}])
    with pytest.raises(DatasetError, match="holds files"):
        files.to_table()
    with pytest.raises(DatasetError, match="holds records"), fb.dataset("new").edit() as edit:
        edit.add_records([{"id": 1}])
        edit.add("a.txt", tmp_path / "a.txt")


def test_schema_info_and_dropped_columns(sales: fb.data.Dataset) -> None:
    with sales.edit() as edit:
        edit.set_schema("columns:\n  id:\n    dtype: int64\n    greater_than: 0\n")
        edit.set_info(license="MIT")
        edit.drop_columns("extra")
    assert sales.info() == {"license": "MIT"} and "extra" not in sales.to_table().column_names
    with pytest.raises(ValidationError, match="row 1: id = -1"), sales.edit() as edit:
        edit.add_records([{"id": -1, "label": "x"}])
    with pytest.raises(ValidationError), sales.edit() as edit:
        edit.update_records("id = 1", id=0)
    with pytest.raises(ValidationError), sales.edit() as edit:  # the rows already saved
        edit.set_schema("columns:\n  id:\n    dtype: int64\n    greater_than: 1\n")
    with sales.edit() as edit:  # unless the edit removes them
        edit.remove_records("id = 1")
        edit.set_schema("columns:\n  id:\n    dtype: int64\n    greater_than: 1\n")
    assert [r["id"] for r in _sorted(sales)] == [2, 3]
    assert fb.data.verify(deep=True).ok


def test_cli_decides_the_kind_by_what_is_first_added(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.csv").write_text("id,label\n1,cat\n2,dog\n")
    (tmp_path / "t" / "b.tsv").write_text("id\tlabel\n3\tcat\n")
    assert "added 3 rows, v1" in run_cli("data", "add", "sales", str(tmp_path / "t"))
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "labels.csv").write_text("path,label\na.jpg,cat\n")
    (tmp_path / "m" / "a.jpg").write_bytes(b"jpg")
    assert "added 2 items" in run_cli("data", "add", "pets", str(tmp_path / "m"))
    listed = _rows(run_cli("data", "ls"))
    assert listed[1][:4] == ["pets", "files", "v1", "2"]
    assert listed[2] == ["sales", "records", "v1", "3", "-"]
    assert _rows(run_cli("data", "ls", "sales", "--where", "id > 1", "--limit", "1")) == [
        ["id", "label"], ["2", "dog"],
    ]  # fmt: skip
    assert run_cli("data", "ls", "sales", "--stats").splitlines()[0] == "3 rows"
    # a table added to a files dataset is a file; a file is no row of a records dataset
    assert "added 1 items" in run_cli("data", "add", "pets", str(tmp_path / "t" / "a.csv"))
    assert "not a table" in run_cli("data", "add", "sales", str(tmp_path / "m" / "a.jpg"), ok=False)
    (tmp_path / "x.dat").write_text("id|label\n4|bird\n")
    run_cli("data", "add", "sales", str(tmp_path / "x.dat"), "--delimiter", "|")
    run_cli("data", "update", "sales", "--where", "label = 'cat'", "--set", "label=feline")
    run_cli("data", "rm", "sales", "--where", "id = 2")
    out = run_cli("data", "query", "-d", "sales", "-q", "SELECT * FROM sales ORDER BY id")
    assert _rows(out) == [["id", "label"], ["1", "feline"], ["3", "feline"], ["4", "bird"]]
    assert "holds records" in run_cli("data", "cat", "sales", "a", ok=False)
    assert run_cli("data", "ls", "sales", "--diff", "1").splitlines() == [
        "--- sales@1",
        "+++ sales",
        "   id  label",
        "-  1   cat",
        "+  1   feline",
        "-  2   dog",
        "-  3   cat",
        "+  3   feline",
        "+  4   bird",
        "3 removed, 3 added",
    ]
    out = run_cli("data", "ls", "sales", "--diff", "1", "--limit", "1")
    assert "-  1   cat" in out and "3 removed, 3 added" in out and "+  4" not in out


def test_query_across_versions_and_kinds(sales: fb.data.Dataset, tmp_path: Path) -> None:
    out = run_cli("data", "query", "-d", "sales@1", "-q", "SELECT count(*) AS n FROM sales")
    assert _rows(out)[1] == ["2"]
    out = run_cli(
        "data", "query", "-d", "old=sales@1", "-d", "new=sales",
        "-q", "SELECT id FROM new EXCEPT SELECT id FROM old",
    )  # fmt: skip
    assert _rows(out)[1:] == [["3"]]
    (tmp_path / "a.txt").write_text("a")
    with fb.dataset("pets").edit() as edit:
        edit.add("a.txt", tmp_path / "a.txt", label="cat")
    out = run_cli("data", "query", "-d", "pets", "-q", "SELECT path, label, size FROM pets")
    assert _rows(out)[1] == ["a.txt", "cat", "1"]  # a files dataset: its index
    out = run_cli("data", "query", "-d", "sales", "-q", "SELECT nope FROM sales", ok=False)
    assert "nope" in out
    run_cli("data", "query", "-d", "sales", "-d", "sales@1", "-q", "SELECT 1", ok=False)
    out = tmp_path / "out.parquet"
    run_cli("data", "query", "-d", "sales", "-q", f"COPY (SELECT * FROM sales) TO '{out}'")
    assert pq.read_table(out).num_rows == 3


def test_reading_and_saving_record_lineage(sales: fb.data.Dataset) -> None:
    derived = fb.dataset("derived")
    with derived.edit() as edit:
        edit.add_records(sales.to_table("label = 'cat'"))
    assert derived.log()[0].inputs == (f"test/sales@{sales.version}",)


def test_a_save_made_meanwhile_is_kept(sales: fb.data.Dataset) -> None:
    first = fb.dataset("sales").edit()
    first.add_records([{"id": 10, "label": "a"}])
    first.update_records("id IN (1, 10)", label="first")  # redone in order on the latest
    with fb.dataset("sales").edit() as second:
        second.remove_records("id = 2")
        second.add_records([{"id": 11, "label": "b", "more": 1}])
    first.commit()
    rows = {r["id"]: r["label"] for r in _sorted(sales)}
    assert rows == {1: "first", 3: "cat", 10: "first", 11: "b"}


def test_operations_apply_in_the_order_written(sales: fb.data.Dataset) -> None:
    versions = len(sales.log())
    with sales.edit("in order") as edit:
        edit.add_records([{"id": 9, "label": "cat"}, {"id": 10, "label": None}])
        edit.update_records("label = 'cat'", label="feline")  # saved rows and the new one
        edit.remove_records("label IS NULL")  # a row this edit added
        edit.update_records("id = 2", label="feline")
        edit.remove_records("label = 'cat'")  # none are left: they were renamed above
        edit.update_records("label = 'feline' AND id > 2", label="big", size=1)  # changed twice
        edit.remove_records("id = 1")  # changed, then removed
        edit.add_records([{"id": 11, "label": "cat"}])  # after the filters: untouched
    assert edit.rows_added == 2
    assert [(r["id"], r["label"], r["size"]) for r in _sorted(sales)] == [
        (2, "feline", None), (3, "big", 1), (9, "big", 1), (11, "cat", None),
    ]  # fmt: skip
    assert len(sales.log()) == versions + 1  # one version
    before = sales.log()[1].version
    assert [r["label"] for r in _sorted(fb.dataset("sales", before))] == ["cat", "dog", "cat"]


def test_a_new_dataset_takes_filters_after_its_first_rows() -> None:
    ds = fb.dataset("fresh")
    with ds.edit() as edit:
        edit.add_records([{"id": 1, "ok": True}, {"id": 2, "ok": False}])
        edit.remove_records("NOT ok")
        edit.update_records("id = 1", note="kept")
    assert ds.to_table().to_pylist() == [{"id": 1, "ok": True, "note": "kept"}]
    with ds.edit() as edit:
        edit.set_schema("columns:\n  id:\n    dtype: int64\n    greater_than: 0\n")
    with pytest.raises(ValidationError), ds.edit() as edit:
        edit.add_records([{"id": 5, "ok": True}])
        edit.update_records("id = 5", id=-5)
    assert ds.to_table().num_rows == 1


def test_diff_counts_repeated_rows_and_new_columns(sales: fb.data.Dataset) -> None:
    first = sales.log()[1].version  # id, label
    with sales.edit() as edit:
        edit.add_records([{"id": 1, "label": "cat"}, {"id": 1, "label": "cat"}])
    diff = sales.diff(first).to_pylist()
    assert diff == [  # the row already there once is not among them; the new column is null
        {"_change": "added", "id": 1, "label": "cat", "extra": None},
        {"_change": "added", "id": 1, "label": "cat", "extra": None},
        {"_change": "added", "id": 3, "label": "cat", "extra": 1.5},
    ]
    assert fb.dataset("sales", first).diff(sales.version).num_rows == 3  # the other way: removed
    assert sales.diff(sales.version).num_rows == 0


def test_a_table_without_the_kind_mark_is_told_by_its_columns(sales: fb.data.Dataset) -> None:
    """One table that fieldbook did not mark must not break the listing of all datasets."""
    index = sales.index()
    kept = {k.decode(): v.decode() for k, v in (index.schema.metadata or {}).items()
            if k != b"fieldbook.kind"}  # fmt: skip
    index.update_schema_metadata(kept, replace=True)
    assert fb.dataset("sales").kind == "records"
    assert "records" in run_cli("data", "ls").splitlines()[1]
