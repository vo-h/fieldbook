import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.code import snapshot
from fieldbook.data.cleanup import prune

DAY = 86400


def stored(folder: str, name: str, age_days: float) -> Path:
    path = fb.storage.local_path(f"{folder}/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    old = time.time() - age_days * DAY
    os.utime(path, (old, old))
    return path


def save_version(ds: fb.data.Dataset, files: Path, code: str, packages: str, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CODE", code)
    monkeypatch.setenv("PLATFORM_ENV", json.dumps({"packages": packages}))
    with ds.edit() as edit:
        edit.add(f"{code}.txt", files / "3.jpg")
    monkeypatch.delenv("PLATFORM_CODE")
    monkeypatch.delenv("PLATFORM_ENV")


def test_unused_snapshots_and_package_lists_are_freed(
    files: Path, monkeypatch: pytest.MonkeyPatch, mlflow_runs
) -> None:
    for name in ("ver", "run", "old", "new", "expired"):
        stored("code", f"{name}.tar.gz", age_days=0 if name == "new" else 30)
        stored("packages", f"p-{name}.txt", age_days=0 if name == "new" else 30)
    ds = fb.dataset("cats")
    save_version(ds, files, "expired", "p-expired", monkeypatch)  # v1, deleted by prune
    save_version(ds, files, "ver", "p-ver", monkeypatch)  # v2, kept
    mlflow_runs.log(
        [], tags={"fieldbook.code": "run", "fieldbook.env": json.dumps({"packages": "p-run"})}
    )

    monkeypatch.setattr("fieldbook.data.cleanup.datetime", _Later())
    dry = prune(keep_latest=1, retention=timedelta(days=7), grace=timedelta(0), dry_run=True)
    assert dry.code_snapshots == ["expired", "old"] and dry.package_lists == ["p-expired", "p-old"]
    assert fb.storage.local_path("code/old.tar.gz").exists()  # dry run deletes nothing

    report = prune(keep_latest=1, retention=timedelta(days=7), grace=timedelta(0))
    assert report.versions == {"test/cats": [1]}
    assert report.code_snapshots == ["expired", "old"]
    remaining = sorted(p.name for p in fb.storage.local_path("code").iterdir())
    assert remaining == ["new.tar.gz", "run.tar.gz", "ver.tar.gz"]


class _Later:
    """`datetime.now()` 60 days ahead, so every version is older than the retention."""

    def now(self, tz=None):
        from datetime import UTC, datetime

        return datetime.now(UTC) + timedelta(days=60)


def test_reuse_restarts_the_retention_period(tmp_path: Path) -> None:
    digest = snapshot.store_packages(["a==1"])
    path = fb.storage.local_path(f"packages/{digest}.txt")
    old = time.time() - 30 * DAY
    os.utime(path, (old, old))
    assert snapshot.store_packages(["a==1"]) == digest
    assert path.stat().st_mtime > time.time() - 60
