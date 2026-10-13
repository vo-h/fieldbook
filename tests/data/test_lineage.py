"""Lineage (fieldbook/data/lineage.py): inputs, commands, run tags."""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook.cli import app
from fieldbook.data import DatasetError
from fieldbook.data.lineage import current_command

runner = CliRunner()


def _make(name: str, files: Path, **inputs) -> fb.data.Dataset:
    ds = fb.dataset(name)
    with ds.edit(**inputs) as edit:
        edit.add_dir(files)
    return ds


def test_inputs_are_reads_inside_the_edit(files: Path) -> None:
    raw = _make("raw", files)
    other = _make("other", files)
    other.files()  # read before the edit: not an input
    clean = fb.dataset("clean")
    with clean.edit() as edit:
        edit.add("seed.txt", files / "3.jpg")
    with clean.edit() as edit:
        for path in raw.files(prefix="a/")["path"].to_pylist():  # unpinned: resolves to v1
            with raw.open(path) as f:
                edit.add(path, files / path)
                f.read()
        clean.files()  # reading the dataset being edited is not an input
    assert clean.log()[0].inputs == ("test/raw@1",)
    raw.files()  # after the edit: not recorded anywhere
    assert clean.log()[0].inputs == ("test/raw@1",)


def test_explicit_inputs(files: Path) -> None:
    raw = _make("raw", files)
    raw.tag("v1-final", 1)
    with raw.edit() as edit:
        edit.remove("3.jpg")
    with fb.dataset("clean").edit(inputs=[raw, "raw@v1-final", "hf://datasets/x"]) as edit:
        edit.add("x.txt", files / "a" / "1.txt")
    assert fb.dataset("clean").log()[0].inputs == (
        "hf://datasets/x",
        "test/raw@1",
        "test/raw@2",
    )
    with pytest.raises(DatasetError, match="not found"):
        fb.dataset("clean").edit(inputs=["nope@1"])
    with pytest.raises(TypeError):
        fb.dataset("clean").edit(inputs=[42])


def test_failed_edit_stops_recording(files: Path) -> None:
    raw = _make("raw", files)
    clean = _make("clean", files)
    with pytest.raises(RuntimeError):
        with clean.edit() as edit:
            raw.files()
            raise RuntimeError
    with clean.edit() as edit:
        edit.remove("3.jpg")
    assert clean.log()[0].inputs == ()


def test_command_is_recorded_and_masked(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["/venv/bin/fieldbook", "data", "add", "cats", "./imgs", "--api-key", "s3cr3t"]
    )
    assert current_command() == "fieldbook data add cats ./imgs --api-key '***'"
    monkeypatch.setattr(sys, "argv", ["clean.py", "--token=abc", "--min-size", "5"])
    assert current_command() == "clean.py '--token=***' --min-size 5"
    ds = _make("cats", files)
    assert ds.log()[0].command == "clean.py '--token=***' --min-size 5"
    monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
    assert current_command() == ""


def test_lineage_walks_inputs(files: Path) -> None:
    raw = _make("raw", files)
    with raw.edit() as edit:
        edit.remove("3.jpg")
    labels = _make("labels", files, inputs=["hf://datasets/labels"])
    clean = _make("clean", files, inputs=["raw@1", labels])
    _make("features", files, inputs=[clean, "raw@1"])
    refs = [(n.depth, n.ref, n.status) for n in fb.dataset("features").lineage()]
    assert refs == [
        (0, "test/features@1", "ok"),
        (1, "test/clean@1", "ok"),
        (2, "test/labels@1", "ok"),
        (3, "hf://datasets/labels", "external"),
        (2, "test/raw@1", "ok"),
        (1, "test/raw@1", "repeated"),
    ]
    raw.delete_versions([1], grace=timedelta(0))
    statuses = {n.ref: n.status for n in fb.dataset("features").lineage()}
    assert statuses["test/raw@1"] == "deleted"


def test_cli_versions_and_lineage(files: Path) -> None:
    _make("raw", files)
    result = runner.invoke(app, ["data", "add", "clean", str(files)])
    assert result.exit_code == 0, result.output
    with fb.dataset("clean").edit(inputs=["raw@1"]) as edit:
        edit.remove("3.jpg")
    versions = runner.invoke(app, ["data", "ls", "clean", "--versions"]).output
    assert versions.splitlines()[0].split()[-2:] == ["inputs", "command"]
    assert "raw@1" in versions.splitlines()[1]
    lines = runner.invoke(app, ["data", "ls", "clean", "--lineage"]).output.splitlines()
    assert lines[0].startswith("clean@2  ") and lines[1].startswith("  raw@1  ")
    assert runner.invoke(app, ["data", "ls", "clean@1", "--lineage"]).output.count("\n") == 1
