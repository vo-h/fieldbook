from pathlib import Path

import pytest
from typer.testing import CliRunner

import fieldbook
from fieldbook.cli import app

runner = CliRunner()


def run_cli(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.stdout


def test_version() -> None:
    assert run_cli("version").strip() == fieldbook.__version__


def test_data_workflow(files: Path, tmp_path: Path) -> None:
    assert "added 3 items, v1" in run_cli("data", "add", "cats", str(files), "-m", "initial")
    assert "added 1 items, v2" in run_cli(
        "data", "add", "cats", str(files / "a" / "1.txt"), "--prefix", "extra/"
    )
    assert "removed 1 items, v3" in run_cli("data", "rm", "cats", "3.jpg")
    assert "tag raw -> v1" in run_cli("data", "update", "cats", "--tag", "raw", "--version", "1")

    datasets = run_cli("data", "ls").splitlines()
    assert datasets[0].split() == ["dataset", "kind", "version", "items", "size"]
    assert datasets[1].split()[:4] == ["cats", "files", "v3", "3"]
    history = run_cli("data", "ls", "cats", "--versions").splitlines()
    assert history[0].split()[:4] == ["version", "date", "author", "items"]
    assert history[1].startswith("v3") and history[3].startswith("v1")
    assert "raw" in history[3] and "initial" in history[3]
    assert run_cli("data", "ls", "cats", "--tags").splitlines()[1].split() == ["raw", "v1"]

    assert "would delete 0 versions" in run_cli("data", "rm", "--old", "--dry-run")


def test_data_errors(files: Path) -> None:
    result = runner.invoke(app, ["data", "ls", "missing", "--versions"])
    assert result.exit_code == 1
    assert "not found" in result.output
    result = runner.invoke(app, ["data", "add", "cats", str(files / "nope")])
    assert result.exit_code == 1


def test_update_and_tags(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    assert "v2" in run_cli(
        "data", "update", "cats", "3.jpg", "--set", "label=tabby", "--set", "score=0.5"
    )
    row = fieldbook.dataset("cats").index().to_table(filter="path = '3.jpg'").to_pylist()[0]
    assert (row["label"], row["score"]) == ("tabby", 0.5)
    result = runner.invoke(app, ["data", "update", "cats", "3.jpg", "--set", "bad"])
    assert result.exit_code == 1

    run_cli("data", "update", "cats", "--tag", "best", "--version", "1")
    assert "tag best -> v2" in run_cli("data", "update", "cats", "--tag", "best")
    assert fieldbook.dataset("cats").tags() == {"best": 2}
    assert "removed tag best" in run_cli("data", "rm", "cats", "--tag", "best")
    assert fieldbook.dataset("cats").tags() == {}


def test_rm_versions_and_dataset(files: Path) -> None:
    for name in ["1.txt", "2.txt"]:
        run_cli("data", "add", "cats", str(files / "a" / name))
    run_cli("data", "add", "cats", str(files / "3.jpg"))
    out = run_cli("data", "rm", "cats@1,2", "--now")
    assert "deleted 2 versions (test/cats v1, v2)" in out and "; freed " in out
    assert [v.version for v in fieldbook.dataset("cats").log()] == [3]

    result = runner.invoke(app, ["data", "rm", "cats@3"])
    assert result.exit_code == 1 and "latest version" in result.output

    result = runner.invoke(app, ["data", "rm", "cats", "--all"], input="n\n")
    assert result.exit_code == 1 and fieldbook.dataset("cats").exists
    out = run_cli("data", "rm", "cats", "--all", "--dry-run")
    assert "would delete dataset test/cats" in out and "space is freed" in out
    out = run_cli("data", "rm", "cats", "--all", "--yes")
    assert "deleted dataset test/cats; freed " in out
    assert not fieldbook.dataset("cats").exists
    assert "deleted 0 versions" in run_cli("data", "rm", "--old", "--now")


def test_rm_argument_errors(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    for args in (
        ["cats"],
        ["cats", "--all", "--old"],
        ["cats@1", "3.jpg"],
        ["--all"],
        ["cats", "3.jpg", "--dry-run"],
        ["cats@x"],
    ):
        result = runner.invoke(app, ["data", "rm", *args])
        assert result.exit_code != 0, args


def test_ls_items_and_stats(files: Path) -> None:
    run_cli("data", "add", "cats", str(files), "--prefix", "train/")
    run_cli("data", "update", "cats", "train/3.jpg", "--set", "label=tabby")
    assert run_cli("data", "ls", "cats", "--prefix", "train/a/").split() == [
        "train/a/1.txt",
        "train/a/2.txt",
    ]
    long = run_cli("data", "ls", "cats@2", "-l", "--where", "label = 'tabby'").splitlines()
    assert long[0].split() == ["path", "size", "type", "label"]
    assert long[1].split()[0] == "train/3.jpg" and long[1].split()[-1] == "tabby"
    assert len(run_cli("data", "ls", "cats", "--limit", "1").splitlines()) == 1
    stats = run_cli(
        "data", "ls", "cats", "--stats", "--prefix", "train/", "--depth", "1"
    ).splitlines()
    assert stats[0].split() == ["dir", "items", "share", "size"]
    assert stats[1].split()[:3] == [".", "1", "33.3%"]
    assert stats[-1].split()[:3] == ["total", "3", "100.0%"]
    assert run_cli("data", "ls", "cats@1", "--stats").splitlines()[1].split()[:2] == ["3", "100.0%"]
    result = runner.invoke(app, ["data", "ls", "missing"])
    assert result.exit_code == 1


def test_ls_argument_errors(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    for args in (
        ["cats", "--versions", "--tags"],
        ["cats", "--depth", "1"],
        ["cats", "--stats", "-l"],
        ["cats", "--versions", "--prefix", "a/"],
        ["cats@1", "--versions"],
        ["--versions"],
    ):
        result = runner.invoke(app, ["data", "ls", *args])
        assert result.exit_code != 0, args


def test_cat(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    result = runner.invoke(app, ["data", "cat", "cats", "3.jpg"])
    assert result.exit_code == 0 and result.stdout_bytes == b"\xff\xd8three"
    assert runner.invoke(app, ["data", "cat", "cats@1", "a/1.txt"]).stdout_bytes == b"one"
    result = runner.invoke(app, ["data", "cat", "cats", "missing.jpg"])
    assert result.exit_code == 1 and "not found" in result.output


def test_add_multiple_sources_with_metadata(files: Path, tmp_path: Path) -> None:
    extra = tmp_path / "extra.png"
    extra.write_bytes(b"png")
    out = run_cli(
        "data",
        "add",
        "cats",
        str(files / "a"),
        str(extra),
        "--prefix",
        "train/",
        "--set",
        "label=tabby",
        "--set",
        "score=1",
        "-m",
        "batch",
    )
    assert "added 3 items, v1" in out
    rows = fieldbook.dataset("cats").files().to_pylist()
    assert [r["path"] for r in rows] == ["train/1.txt", "train/2.txt", "train/extra.png"]
    assert {(r["label"], r["score"]) for r in rows} == {("tabby", 1)}
    assert fieldbook.dataset("cats").log()[0].message == "batch"

    result = runner.invoke(app, ["data", "add", "cats", str(extra), "--prefix", "train/"])
    assert result.exit_code == 1 and "already exists" in result.output
    assert fieldbook.dataset("cats").version == 1


def test_add_tags_the_new_version(files: Path) -> None:
    assert "added 3 items, v1, tagged raw" in run_cli(
        "data", "add", "cats", str(files), "--tag", "raw"
    )
    out = run_cli("data", "add", "cats", str(files / "a"), "--prefix", "more/", "--tag", "batch-2")
    assert "v2, tagged batch-2" in out
    assert fieldbook.dataset("cats").tags() == {"raw": 1, "batch-2": 2}

    result = runner.invoke(
        app, ["data", "add", "cats", str(files / "3.jpg"), "--prefix", "x/", "--tag", "raw"]
    )
    assert result.exit_code == 1 and "update --tag" in result.output
    assert fieldbook.dataset("cats").version == 2, "nothing is committed"
    result = runner.invoke(app, ["data", "add", "dogs", str(files), "--tag", "123"])
    assert result.exit_code == 1 and not fieldbook.dataset("dogs").exists


def test_add_and_update_argument_errors(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    for args in (
        ["add", "cats"],
        ["add", "cats", str(files / "3.jpg"), "--set", "bad"],
        ["update", "cats"],
        ["update", "cats", "3.jpg"],
        ["update", "cats", "--tag", "x", "--file", str(files / "3.jpg")],
        ["update", "cats", "3.jpg", "--file", str(files)],
        ["update", "cats", "3.jpg", "--tag", "x"],
        ["update", "cats", "--tag", "x", "--set", "a=1"],
        ["update", "cats", "3.jpg", "--set", "a=1", "--version", "1"],
        ["update", "cats", "--tag", "x", "--version", "9"],
    ):
        result = runner.invoke(app, ["data", *args])
        assert result.exit_code != 0, args


def test_update_file(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    out = run_cli(
        "data",
        "update",
        "cats",
        "3.jpg",
        "--file",
        str(files / "a" / "1.txt"),
        "--set",
        "label=x",
        "-m",
        "fix image",
    )
    assert "updated 3.jpg, v" in out
    assert runner.invoke(app, ["data", "cat", "cats", "3.jpg"]).stdout_bytes == b"one"
    assert runner.invoke(app, ["data", "cat", "cats@1", "3.jpg"]).stdout_bytes == b"\xff\xd8three"


def test_friendly_errors(files: Path) -> None:
    (files / ".DS_Store").write_bytes(b"x")
    assert "added 3 items" in run_cli("data", "add", "cats", str(files), "--set", "label=tabby")
    assert "added 4 items" in run_cli("data", "add", "hidden", str(files), "--hidden")
    for args in (
        ["ls", "cats", "--where", "nope = 1"],
        ["ls", "cats", "--stats", "--by", "nope"],
        ["update", "cats", "3.jpg", "--set", "added_at=1"],
    ):
        result = runner.invoke(app, ["data", *args])
        assert result.exit_code == 1, args
        assert "Traceback" not in result.output and result.output.startswith("error:"), args
    assert "updated" in run_cli("data", "update", "cats", "3.jpg", "--set", "label=5")


def test_no_option_reaches_another_projects_datasets() -> None:
    """Projects do not read each other's datasets (sharing.md §4)."""
    result = runner.invoke(app, ["data", "--project", "vision", "ls"])
    assert result.exit_code != 0 and "No such option" in result.output


def test_info_revert_diff_verify(files: Path) -> None:
    run_cli("data", "add", "cats", str(files), "--info", "license=MIT", "--info", "source=web")
    assert "updated info, v" in run_cli("data", "update", "cats", "--info", "source=")
    assert len(fieldbook.dataset("cats").log()) == 2  # one version an edit
    assert run_cli("data", "ls", "cats", "--info").splitlines()[1:] == ["license  MIT"]
    assert run_cli("data", "ls", "cats@1", "--info").count("\n") == 3

    run_cli("data", "update", "cats", "3.jpg", "--set", "label=tabby", "--info", "license=GPL")
    diff = run_cli("data", "ls", "cats", "--diff", "1").splitlines()
    assert diff[1].split() == ["3.jpg", "modified", "label"]
    assert diff[-1] == "0 added, 0 removed, 1 modified"

    assert " = 1" in run_cli("data", "update", "cats", "--revert", "1")
    assert len(fieldbook.dataset("cats").log()) == 4
    assert run_cli("data", "ls", "cats", "--diff", "1").splitlines()[-1].startswith("0 added")

    out = run_cli("data", "ls", "--verify", "--deep")
    assert "checked 3 files (content hashes) in 1 datasets: 0 missing, 0 corrupt" in out


def test_verify_exit_code(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    for path in fieldbook.storage.local_path("datasets/cats.lance/data").rglob("*"):
        if path.is_file():
            path.unlink()  # the data files: the files' bytes are in them
    result = runner.invoke(app, ["data", "ls", "cats", "--verify"])
    assert result.exit_code == 1 and "missing  test/cats@v1:" in result.output


def test_new_argument_errors(files: Path) -> None:
    run_cli("data", "add", "cats", str(files))
    for args in (
        ["ls", "cats", "--diff", "1", "--versions"],
        ["ls", "cats", "--diff", "1", "--where", "size > 1"],
        ["ls", "cats", "--deep"],
        ["ls", "cats@1", "--verify"],
        ["ls", "--info"],
        ["update", "cats", "--revert", "1", "--tag", "x"],
        ["update", "cats", "--revert", "1", "--info", "a=b"],
        ["update", "cats", "--info", "bad"],
        ["update", "cats", "--set", "a=1"],
        ["update", "missing", "--info", "a=b"],
    ):
        result = runner.invoke(app, ["data", *args])
        assert result.exit_code == 1, args
        assert "Traceback" not in result.output, args


def test_invalid_project_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("FIELDBOOK_PROJECT", "../x")
    result = runner.invoke(app, ["data", "ls"])
    assert result.exit_code == 1 and "invalid project name" in result.output


def test_rm_old_with_retention_options(files: Path) -> None:
    run_cli("data", "add", "cats", str(files / "a" / "1.txt"))
    for i in range(4):
        run_cli("data", "update", "cats", "1.txt", "--set", f"step={i}")
    run_cli("data", "update", "cats", "--tag", "keep-me", "--version", "1")
    out = run_cli("data", "rm", "cats", "--old", "--keep", "2", "--days", "0", "--dry-run")
    assert "would delete 2 versions (test/cats v2, v3)" in out
    assert "deleted 2 versions" in run_cli(
        "data", "rm", "cats", "--old", "--keep", "2", "--days", "0"
    )
    versions = [
        line.split()[0] for line in run_cli("data", "ls", "cats", "--versions").splitlines()[1:]
    ]
    assert versions == ["v5", "v4", "v1"]  # latest 2, plus the tagged version
    result = runner.invoke(app, ["data", "rm", "cats@4", "--keep", "1"])
    assert result.exit_code == 1 and "--keep and --days apply to --old" in result.output
    result = runner.invoke(app, ["data", "rm", "--old", "--keep", "0"])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "args",
    [
        ["data", "ls", "bad/name"],
        ["data", "ls", ".hidden"],
        ["data", "cat", "bad name", "a.txt"],
        ["data", "add", "bad/name", "."],
        ["data", "query", "-d", "bad/name", "-q", "SELECT 1"],
        ["data", "update", "bad/name", "--info", "a=b"],
        ["data", "rm", "bad/name", "--all", "-y"],
    ],
)
def test_invalid_names_are_reported_not_crashed(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "invalid dataset name" in result.output, result.output
    assert "Traceback" not in result.output


def test_setup_errors_print_one_line_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import fieldbook.cli as cli
    from fieldbook.images import registry

    def broken(*args, **kwargs):
        raise registry.RegistryError(
            "set FIELDBOOK_REGISTRY_USERNAME and FIELDBOOK_REGISTRY_PASSWORD"
        )

    monkeypatch.setattr(cli, "app", broken)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert (
        capsys.readouterr().err
        == "error: set FIELDBOOK_REGISTRY_USERNAME and FIELDBOOK_REGISTRY_PASSWORD\n"
    )
    monkeypatch.setenv("FIELDBOOK_DEBUG", "1")
    with pytest.raises(registry.RegistryError):  # the traceback, on request
        cli.main()


def test_bugs_keep_their_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    import fieldbook.cli as cli

    def bug(*args, **kwargs):
        raise TypeError("a bug")

    monkeypatch.setattr(cli, "app", bug)
    with pytest.raises(TypeError):
        cli.main()


def test_unsupported_storage_is_one_line(tmp_path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    fieldbook = Path(sys.executable).parent / "fieldbook"
    env = {
        **os.environ,
        "FIELDBOOK_STORAGE_ROOT": "gs://some-bucket/fieldbook",
    }  # other clouds: deferred
    result = subprocess.run(
        [str(fieldbook), "data", "ls"], env=env, capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert "an AWS S3 bucket (s3://); other clouds are not supported yet" in result.stderr


def test_fieldbook_build_prints_the_image_or_one_line_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """D2: `fieldbook build` reports the image it found or built."""
    import subprocess

    from fieldbook import images

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    ref = "1.dkr.ecr.us-west-2.amazonaws.com/fieldbook:env-1@sha256:1"
    monkeypatch.setattr(
        images, "image", lambda repo, cfg, local=False: images.Image(ref, None, ("AWS", "pw"))
    )
    result = runner.invoke(app, ["build", "--local"])
    assert result.exit_code == 0 and result.output.strip() == ref

    def missing(repo, cfg, local=False):  # noqa: ANN001, ANN202
        raise images.ImageError("set image_registry to an ECR repository for cloud runs")

    monkeypatch.setattr(images, "image", missing)
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 1 and "error: set image_registry" in result.output


def test_starting_fieldbook_loads_no_heavy_libraries() -> None:
    """Every `fieldbook` command starts by importing the CLI; data, tracking, and cloud libraries
    load only in the commands that use them (`fieldbook config` took 3 s when they did)."""
    import subprocess
    import sys

    heavy = ("pyarrow", "lance", "pandas", "mlflow", "sky", "duckdb", "botocore")
    code = f"import sys, fieldbook.cli; print(sorted(m for m in {heavy!r} if m in sys.modules))"
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "[]"
