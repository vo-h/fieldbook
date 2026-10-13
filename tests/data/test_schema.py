import sys
from pathlib import Path

import pandera.pandas as pa
import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import ConflictError, DatasetError, ValidationError

runner = CliRunner()

SCHEMA = """
columns:
  label:
    dtype: str
    isin: [cat, dog]
  score:
    dtype: float
    nullable: true
    required: false
    in_range: {min_value: 0, max_value: 1}
"""


def _dataset(files: Path) -> fb.data.Dataset:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files, label="cat")
    return ds


def test_schema_checks_every_commit(files: Path) -> None:
    ds = _dataset(files)
    with ds.edit() as edit:
        edit.set_schema(SCHEMA)
    with_schema = ds.version
    assert with_schema > 1 and "isin" in ds.schema()
    with pytest.raises(ValidationError) as e:
        with ds.edit() as edit:
            edit.update("3.jpg", label="dgo", score=1.5)
            edit.add("new.txt", files / "a" / "1.txt")  # no label
    assert ds.version == with_schema
    failures = {(f["path"], f["column"], f["check"]) for f in e.value.failures}
    assert ("3.jpg", "label", "isin(['cat', 'dog'])") in failures
    assert ("3.jpg", "score", "in_range(0, 1)") in failures
    assert ("new.txt", "label", "not_nullable") in failures
    assert "nothing was committed" in str(e.value)
    with ds.edit() as edit:
        edit.add("new.txt", files / "a" / "1.txt", label="dog", score=0.5)
    assert ds.version > with_schema and len(ds.log()) == 3  # three versions, whatever their numbers


def test_set_schema_checks_existing_rows_and_backfill(files: Path) -> None:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files)
    with pytest.raises(ValidationError, match="column_in_dataframe"):
        with ds.edit() as edit:
            edit.set_schema(SCHEMA)
    with ds.edit() as edit:
        edit.set_schema(SCHEMA)
        for path in ("a/1.txt", "a/2.txt", "3.jpg"):
            edit.update(path, label="cat")
    with pytest.raises(ValidationError):
        with ds.edit() as edit:
            edit.drop_columns("label")


def test_schema_is_versioned(files: Path) -> None:
    ds = _dataset(files)
    with ds.edit() as edit:
        edit.set_schema(pa.DataFrameSchema({"label": pa.Column(str)}))
    with_schema = ds.version
    assert fb.dataset("cats", version=1).schema() is None
    ds.revert(1)
    assert ds.schema() is None
    ds.revert(with_schema)
    assert ds.schema() is not None
    with ds.edit() as edit:
        edit.set_schema(None)
    assert ds.schema() is None and len(ds.log()) == 5


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            pa.DataFrameSchema({"label": pa.Column(str, pa.Check(lambda s: s.str.len() < 5))}),
            "custom Python checks",
        ),
        ("columns:\n  label:\n    dtype: str\n    isn: [cat]\n", "unknown keys"),
        ("columns:\n  label:\n    dtype: nonsense\n", "invalid schema"),
        ("schema_type: polars_dataframe\ncolumns:\n  a:\n    dtype: String\n", "only pandas"),
        ("[1, 2]", "'columns' mapping"),
        ("a: [", "not valid YAML"),
        (42, "must be a pandera"),
    ],
)
def test_set_schema_rejects(files: Path, schema: object, error: str) -> None:
    ds = _dataset(files)
    with pytest.raises(DatasetError, match=error):
        ds.edit().set_schema(schema)


def test_concurrent_edits_follow_the_latest_schema(files: Path) -> None:
    ds = _dataset(files)
    late = ds.edit()
    late.update("3.jpg", label="bird")
    with ds.edit() as edit:
        edit.set_schema(SCHEMA)
    with pytest.raises(ValidationError):
        late.commit()
    first, second = ds.edit(), ds.edit()
    first.set_schema(None)
    second.set_schema(SCHEMA.replace("dog]", "dog, bird]"))
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()


def test_missing_pandera_is_reported(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(files)
    monkeypatch.setitem(sys.modules, "pandera.pandas", None)
    with pytest.raises(DatasetError, match=r"fieldbook\[validate\]"):
        ds.edit().set_schema(SCHEMA)


def run_cli(*args: str, ok: bool = True) -> str:
    result = runner.invoke(app, list(args))
    assert (result.exit_code == 0) == ok, result.output
    return result.output


def test_cli_schema(files: Path, tmp_path: Path) -> None:
    _dataset(files)
    schema = tmp_path / "schema.yaml"
    schema.write_text(SCHEMA)
    assert "no schema" in run_cli("data", "ls", "cats", "--schema")
    assert "updated schema, v" in run_cli("data", "update", "cats", "--schema", str(schema))
    assert "isin" in run_cli("data", "ls", "cats", "--schema")
    out = run_cli("data", "update", "cats", "3.jpg", "--set", "label=dgo", ok=False)
    assert "3.jpg: label = 'dgo' fails isin" in out
    assert "removed schema, v" in run_cli("data", "update", "cats", "--schema", "")
    assert "no schema" in run_cli("data", "ls", f"cats@{fb.dataset('cats').version}", "--schema")
