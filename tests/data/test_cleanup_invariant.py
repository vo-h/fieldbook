"""Cleanup safety: after every operation that references stored files, the most aggressive
cleanup must not free anything a remaining version uses, and must keep tagged versions and
versions used by runs.

New operations that create references to stored files belong in this test.
"""

from datetime import timedelta
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.data import DatasetError, verify
from fieldbook.data.cleanup import prune

NOW = timedelta(0)


def clean_up_and_check(protected: dict[str, set[int]]) -> None:
    prune(keep_latest=1, retention=NOW, grace=NOW)
    report = verify(deep=True)
    assert report.ok, (report.missing, report.corrupt)
    for key, versions in protected.items():
        name = key.split("/")[1]
        remaining = {v.version for v in fb.dataset(name).log()}
        assert versions <= remaining, (key, versions, remaining)


def test_cleanup_never_frees_files_in_use(files: Path, tmp_path: Path, mlflow_runs) -> None:
    other = tmp_path / "other.txt"
    other.write_text("replacement")
    raw, clean = fb.dataset("raw"), fb.dataset("clean")
    protected: dict[str, set[int]] = {}

    with raw.edit() as edit:  # add
        edit.add_dir(files)
    raw.tag("first", 1)
    protected["test/raw"] = {1}
    clean_up_and_check(protected)

    with raw.edit() as edit:  # update content and metadata
        edit.update("a/1.txt", other, label="x")
    clean_up_and_check(protected)

    with raw.edit() as edit:  # move and remove
        edit.move("a/", "b/")
        edit.remove("3.jpg")
    clean_up_and_check(protected)

    raw.revert(1)  # revert to a version whose files later versions dropped
    clean_up_and_check(protected)

    with clean.edit() as edit:  # copy from a non-latest version
        edit.copy_from(fb.dataset("raw", version="first"))
    clean_up_and_check(protected)

    used = raw.version
    with raw.edit() as edit:
        edit.remove("a/2.txt")
    run = mlflow_runs.log([f"test/raw@{used}"])  # a run read raw@used
    protected["test/raw"] |= {used}
    clean_up_and_check(protected)

    raw.untag("first")  # raw@1 is no longer protected; clean still uses its files
    protected["test/raw"] -= {1}
    clean_up_and_check(protected)

    with pytest.raises(DatasetError, match="runs used"):
        raw.delete(grace=NOW)
    mlflow_runs.delete(run)
    raw.delete(grace=NOW)
    clean_up_and_check({})
    assert len(clean.files()) == 3
