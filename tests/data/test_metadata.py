from pathlib import Path

import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import ConflictError, DatasetError

runner = CliRunner()


def _labeled(files: Path) -> fb.data.Dataset:
    ds = fb.dataset("cats")
    with ds.edit() as edit:
        edit.add_dir(files, label="cat", score=0.5)
    return ds


# --- dropping columns --------------------------------------------------------


def test_drop_columns_keeps_history(files: Path) -> None:
    ds = _labeled(files)
    with ds.edit() as edit:
        edit.drop_columns("label")
    assert "label" not in ds.files().column_names
    assert "score" in ds.files().column_names
    assert "label" in fb.dataset("cats", version=1).files().column_names


def test_drop_columns_errors(files: Path) -> None:
    ds = _labeled(files)
    with pytest.raises(ValueError, match="reserved"):
        ds.edit().drop_columns("hash")
    with pytest.raises(DatasetError, match="not in cats"):
        with ds.edit() as edit:
            edit.drop_columns("missing")
    with pytest.raises(DatasetError, match="both set and dropped"):
        with ds.edit() as edit:
            edit.update("3.jpg", label="dog")
            edit.drop_columns("label")
    assert ds.version == 1


def test_drop_conflicts_with_concurrent_edit(files: Path) -> None:
    ds = _labeled(files)
    first, second = ds.edit(), ds.edit()
    first.update("3.jpg", label="dog")
    second.drop_columns("label")
    first.commit()
    with pytest.raises(ConflictError):
        second.commit()


# --- CLI ---------------------------------------------------------------------


def run_cli(*args: str, ok: bool = True) -> str:
    result = runner.invoke(app, list(args))
    assert (result.exit_code == 0) == ok, result.output
    return result.output


def test_update_from_csv_is_one_version(files: Path, tmp_path: Path) -> None:
    _labeled(files)
    csv = tmp_path / "labels.csv"
    csv.write_text("path,label,score\na/1.txt,tabby,0.9\n3.jpg,,null\n")
    assert "updated 2 items, v2" in run_cli("data", "update", "cats", "--from", str(csv))
    rows = {r["path"]: r for r in fb.dataset("cats").files().to_pylist()}
    assert rows["a/1.txt"]["label"] == "tabby" and rows["a/1.txt"]["score"] == 0.9
    assert rows["3.jpg"]["label"] == "cat"  # empty cell: unchanged
    assert rows["3.jpg"]["score"] is None  # null: cleared
    assert rows["a/2.txt"]["label"] == "cat"


def test_update_from_csv_errors(files: Path, tmp_path: Path) -> None:
    _labeled(files)
    no_path = tmp_path / "no_path.csv"
    no_path.write_text("file,label\na/1.txt,x\n")
    assert "no 'path' column" in run_cli("data", "update", "cats", "--from", str(no_path), ok=False)
    dup = tmp_path / "dup.csv"
    dup.write_text("path,label\na/1.txt,x\na/1.txt,y\n")
    assert "duplicate path" in run_cli("data", "update", "cats", "--from", str(dup), ok=False)
    missing = tmp_path / "missing.csv"
    missing.write_text("path,label\nnope.txt,x\n")
    assert "not in cats" in run_cli("data", "update", "cats", "--from", str(missing), ok=False)
    assert "is not a file" in run_cli("data", "update", "cats", "--from", "nope.csv", ok=False)
    assert fb.dataset("cats").version == 1


def test_update_drop_and_combined(files: Path, tmp_path: Path) -> None:
    _labeled(files)
    out = run_cli("data", "update", "cats", "--drop", "score", "--info", "license=MIT")
    assert "dropped score; info, v" in out
    ds = fb.dataset("cats")
    assert "score" not in ds.files().column_names and ds.info() == {"license": "MIT"}
    assert "specify one of" in run_cli(
        "data", "update", "cats", "--drop", "label", "--tag", "x", ok=False
    )
