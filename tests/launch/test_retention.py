"""Freeing what old runs left behind (fieldbook/launch/retention.py; `fieldbook log rm`)."""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

from fieldbook import notes, storage
from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.launch import records, retention


def run_id(days_ago: float, tail: str) -> str:
    then = time.gmtime(time.time() - days_ago * 86400)
    return time.strftime("%Y%m%d-%H%M%S", then) + f"-{tail}"


def leave(run: str, *, checkpoint: int = 100, kept: int = 10) -> None:
    """What a run leaves in storage: a checkpoint, its output and a file, its record."""
    files = storage.fs(settings())
    for path, size in ((f"checkpoints/{run}/last.ckpt", checkpoint),
                       (f"runs/{run}/output.log.gz", kept)):  # fmt: skip
        if size:
            storage.makedirs(files, path.rsplit("/", 1)[0])
            files.pipe_file(path, b"0" * size)
    records.start(settings(), run, {"command": "train.py"})
    records.end(settings(), run, 0)


def test_old_runs_bulk_is_freed_and_their_record_stays(repo: Path) -> None:
    cfg, files = settings(), storage.fs(settings())
    ages = ((40, "aaaaaa"), (50, "bbbbbb"), (2, "cccccc"), (45, "dddddd"))
    old, good, recent, going = (run_id(days, tail) for days, tail in ages)
    for run in (old, good, recent, going):
        leave(run)
    notes.add(good, "the one we ship", verdict="good")
    files.rm(f"runs/{going}.end.0")  # no end, and a sign of life: still going
    records.beat(cfg, going)
    planned = retention.prune(cfg, dry_run=True)
    assert planned.checkpoints == {old: 100} and planned.kept == {old: 10}
    assert planned.spared == [good] and planned.total == 110
    assert files.exists(f"checkpoints/{old}/last.ckpt")  # a dry run deletes nothing
    done = retention.prune(cfg)
    assert done.checkpoints == {old: 100}
    assert not files.exists(f"checkpoints/{old}") and not files.exists(f"runs/{old}")
    assert files.exists(f"runs/{old}.json") and files.exists(f"runs/{old}.end.0")  # the record
    for kept in (good, recent, going):
        assert files.exists(f"checkpoints/{kept}/last.ckpt")
    assert retention.prune(cfg, retention=timedelta(days=1)).checkpoints == {recent: 100}
    assert retention.prune(cfg).total == 0


def test_the_rm_command(repo: Path) -> None:
    files = storage.fs(settings())
    old, recent = run_id(40, "aaaaaa"), run_id(2, "cccccc")
    leave(old)
    leave(recent, checkpoint=2048)
    code, out = run_cli("log", "rm", "--old", "--dry-run")
    assert code == 0 and f"checkpoints: 1 (100 B): {old}" in out and "would free 110 B" in out
    assert run_cli("log", "rm", "--old")[0] == 1  # asks first; no answer is a no
    assert files.exists(f"checkpoints/{old}/last.ckpt")
    code, out = run_cli("log", "rm", "--old", "--yes")
    assert code == 0 and "deleted; freed 110 B" in out and not files.exists(f"checkpoints/{old}")
    code, out = run_cli("log", "rm", recent[:15], "--yes")  # one run, whatever its age
    assert code == 0 and "checkpoints: 1 (2.0 KB)" in out
    assert not files.exists(f"checkpoints/{recent}") and recent in run_cli("log")[1]
    assert "nothing to delete" in run_cli("log", "rm", "--old", "--days", "0")[1]
    assert run_cli("log", "rm")[0] == 1 and run_cli("log", "rm", recent, "--old")[0] == 1


def test_images_nothing_has_used_are_deleted(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)

    def image(tag: str | None, pushed: int, pulled: int | None = None, size: int = 1000) -> dict:
        found = {"imagePushedAt": now - timedelta(days=pushed), "imageSizeInBytes": size}
        if tag:
            found["imageTags"] = [tag]
        if pulled is not None:
            found["lastRecordedPullTime"] = now - timedelta(days=pulled)
        return found

    deleted = []

    class ECR:
        def get_paginator(self, name: str):  # noqa: ANN201
            pages = [{"imageDetails": [
                image("env-unused", 90, 60), image("env-pulled", 90, 3), image("env-new", 1),
                image("buildcache", 90), image(None, 90), image("env-older", 200, size=500),
            ]}]  # fmt: skip
            return type("P", (), {"paginate": lambda self, **kw: pages})()

        def batch_delete_image(self, repositoryName: str, imageIds: list) -> None:  # noqa: N803
            deleted.extend(i["imageTag"] for i in imageIds)

    monkeypatch.setenv(
        "FIELDBOOK_IMAGE_REGISTRY", "123456789012.dkr.ecr.us-west-2.amazonaws.com/fb"
    )
    monkeypatch.setattr(aws, "client", lambda service, region=None: ECR())
    planned = retention.prune(settings(), dry_run=True)
    assert planned.images == {"env-unused": 1000, "env-older": 500} and deleted == []
    retention.prune(settings())
    assert sorted(deleted) == ["env-older", "env-unused"]  # not one pulled lately, the newest,
    # the build cache, or an untagged image
    monkeypatch.setattr(
        aws, "client", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("denied"))
    )
    assert "denied" in retention.prune(settings(), dry_run=True).image_error
