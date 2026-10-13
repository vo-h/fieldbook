"""Code snapshots (fieldbook/code/snapshot.py): file selection, secret scan, size, checkout."""

import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import (
    SCRIPT,
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook.code import snapshot


def test_snapshots_are_deduplicated(repo: Path, gitleaks: Path) -> None:
    run_cli("run", "clean.py")
    with fb.dataset("clean").edit() as edit:
        edit.remove("3.jpg")
    run_cli("run", "clean.py")  # fails (paths exist) but still snapshots
    first = list((fb.storage.local_path("code")).iterdir())
    (repo / "new.py").write_text("x = 1\n")
    run_cli("run", "clean.py")
    assert len(first) == 1 and len(list(fb.storage.local_path("code").iterdir())) == 2


def test_tree_hash_leaves_the_users_index_alone(repo: Path) -> None:
    subprocess.run(["git", "add", "clean.py"], cwd=repo, check=True)
    before = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
    assert snapshot.tree_hash(repo) == snapshot.tree_hash(repo)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True)
    assert before.stdout == after.stdout


def test_secret_aborts_the_run(repo: Path, gitleaks: Path) -> None:
    (repo / "config.py").write_text('KEY = "FAKE_SECRET"\n')
    code, out = run_cli("run", "clean.py")
    assert code == 1 and "config.py:1  fake" in out and "FAKE_SECRET" not in out
    assert not fb.dataset("clean").exists  # script never started
    assert not (fb.storage.local_path("code")).exists()


def test_missing_gitleaks_or_repo(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert "brew install gitleaks" in run_cli("run", "clean.py")[1]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "s.py").write_text("")
    monkeypatch.chdir(outside)
    assert "git init" in run_cli("run", "s.py")[1]


def test_code_checkout(repo: Path, gitleaks: Path, tmp_path: Path) -> None:
    run_cli("run", "clean.py")
    digest = fb.dataset("clean").log()[0].code
    code, out = run_cli("log", "code", digest[:8], str(tmp_path / "restored"))
    assert code == 0, out
    assert (tmp_path / "restored" / "clean.py").read_text() == SCRIPT
    assert "not empty" in run_cli("log", "code", digest[:8], str(tmp_path / "restored"))[1]
    assert "at least 7" in run_cli("log", "code", "abc", str(tmp_path / "x"))[1]
    assert "no code snapshot" in run_cli("log", "code", "0000000", str(tmp_path / "x"))[1]


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")
def test_real_gitleaks_finds_a_token(repo: Path) -> None:
    (repo / "config.py").write_text('aws_key = "AKIAIOSFODNN7EXAMPLQ"\n')
    code, out = run_cli("run", "clean.py")
    assert code == 1 and "config.py:1  aws-access-token" in out and "AKIA" not in out


def test_fieldbookignore_excludes_tracked_and_untracked_files(repo: Path) -> None:
    (repo / "ckpt").mkdir()
    (repo / "ckpt" / "tracked.pt").write_text("t")
    (repo / "ckpt" / "new.pt").write_text("n")
    (repo / "notes.bin").write_text("b")
    subprocess.run(["git", "add", "ckpt/tracked.pt"], cwd=repo, check=True)
    before = snapshot.tree_hash(repo)
    (repo / ".fieldbookignore").write_text("ckpt/\n*.bin\n")
    names = snapshot.files(repo)
    assert names == [".fieldbookignore", ".gitignore", "clean.py"]
    assert snapshot.tree_hash(repo) != before
    (repo / "ckpt" / "new.pt").write_text("changed")  # ignored: same snapshot
    assert snapshot.tree_hash(repo) == snapshot.tree_hash(repo, names)


def test_the_projects_local_folder_is_never_in_a_snapshot(repo: Path) -> None:
    """.fieldbook/ holds local data, databases, and this machine's settings, even when the
    .gitignore does not list it."""
    for folder in (repo / ".fieldbook", repo / "sub" / ".fieldbook"):
        folder.mkdir(parents=True)
        (folder / "config.toml").write_text('storage_root = "~/big"\n')
    (repo / "sub" / "keep.py").write_text("")
    assert snapshot.files(repo) == [".gitignore", "clean.py", "sub/keep.py"]


def test_oversized_snapshot_aborts_before_running(repo: Path, gitleaks: Path) -> None:
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "a.pt").write_bytes(b"\0" * (2 * 2**20))
    (repo / "checkpoints" / "b.pt").write_bytes(b"\0" * 2**20)
    code, out = run_cli("run", "--max-snapshot-size", "1", "clean.py")
    assert code == 1 and "would be 3 MB (limit 1 MB)" in out
    assert out.index("checkpoints/a.pt") < out.index("checkpoints/b.pt")  # largest first
    assert ".fieldbookignore" in out
    assert not fb.dataset("clean").exists and not fb.storage.local_path("code").exists()
    (repo / ".fieldbookignore").write_text("checkpoints/\n")
    assert run_cli("run", "--max-snapshot-size", "1", "clean.py")[0] == 0


def test_symlinks_outside_the_repository(repo: Path, gitleaks: Path) -> None:
    (repo.parent / "shared.yaml").write_text("lr: 0.1\n")
    (repo / "config.yaml").symlink_to("../shared.yaml")  # outside
    (repo / "abs.yaml").symlink_to(repo / "clean.py")  # absolute: a different path elsewhere
    (repo / "alias.py").symlink_to("clean.py")  # inside: fine
    from fieldbook.code import snapshot

    names = snapshot.files(repo)
    assert snapshot.outside_links(repo, names) == (
        f"abs.yaml -> {repo / 'clean.py'}", "config.yaml -> ../shared.yaml",
    )  # fmt: skip
    with pytest.raises(snapshot.SnapshotError, match="config.yaml -> ../shared.yaml"):
        snapshot.take(repo, refuse_outside_links=True)  # cloud runs: before anything is stored
    assert len(snapshot.take(repo).outside_links) == 2  # local runs: a warning
