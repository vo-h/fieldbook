"""A bucket as the storage root (storage.md), against moto's S3 server."""

import subprocess
import uuid
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import fieldbook as fb
from fieldbook import setup_project, storage
from fieldbook.cli import app
from fieldbook.code import snapshot

pytest.importorskip("moto")
runner = CliRunner()


@pytest.fixture(scope="module")
def s3_endpoint() -> str:
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)  # bound directly
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture(autouse=True)
def bucket_root(s3_endpoint: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A new bucket per test as the storage root, with moto's fake credentials."""
    import boto3

    for name, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)
    bucket = f"fieldbook-{uuid.uuid4().hex[:12]}"
    boto3.client("s3", endpoint_url=s3_endpoint).create_bucket(Bucket=bucket)
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", f"s3://{bucket}/fieldbook")
    monkeypatch.setenv("FIELDBOOK_STORAGE_ENDPOINT", s3_endpoint)
    monkeypatch.setenv("FIELDBOOK_STORAGE_REGION", "us-east-1")
    return f"s3://{bucket}/fieldbook"


def test_datasets_live_in_the_bucket(files: Path, bucket_root: str) -> None:
    pets = fb.dataset("pets")
    with pets.edit() as edit:
        edit.add_dir(files)
    assert storage.location("x") == f"{bucket_root}/test/x"
    assert pets.files().num_rows == 3
    with pets.open("a/1.txt") as f:
        assert f.read() == b"one"
    assert pets.read_many(["3.jpg", "a/2.txt"]) == [b"\xff\xd8three", b"two"]
    stored = storage.fs().find("")
    assert stored and all(path.startswith("datasets/pets.lance/") for path in stored)
    with pets.edit() as edit:
        edit.remove("3.jpg")
    pets.tag("first", 1)
    assert pets.version == 2 and pets.tags() == {"first": 1}
    assert fb.dataset("pets", version="first").files().num_rows == 3
    assert fb.data.list_datasets() == ["pets"]


def test_concurrent_edits_rebase(files: Path) -> None:
    with fb.dataset("pets").edit() as edit:
        edit.add_dir(files)
    first, second = fb.dataset("pets").edit(), fb.dataset("pets").edit()
    first.remove("3.jpg")
    second.remove("a/1.txt")
    first.commit()
    second.commit()  # conflicts at version 2, then rebases onto it
    assert fb.dataset("pets").files()["path"].to_pylist() == ["a/2.txt"]


def test_cleanup_frees_space_in_the_bucket(tmp_path: Path) -> None:
    import os

    pets = fb.dataset("pets")
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(200_000))
    with pets.edit() as edit:
        edit.add("big.bin", big)
        edit.add("small.txt", tmp_path / "big.bin")
    with pets.edit() as edit:
        edit.remove("big.bin")
    before = storage.fs().du("datasets/pets.lance")
    report = fb.data.prune(keep_latest=1, retention=timedelta(0), grace=timedelta(0), referenced={})
    assert report.compacted == ["test/pets"] and report.freed > 150_000
    assert storage.fs().du("datasets/pets.lance") < before - 150_000
    assert pets.read_many(["small.txt"]) == [big.read_bytes()]
    assert fb.data.verify(deep=True).ok


def test_query_reads_the_bucket(tmp_path: Path) -> None:
    pq.write_table(pa.table({"id": [1, 2]}), tmp_path / "p.parquet")
    with fb.dataset("sales").edit() as edit:
        edit.add_records(tmp_path / "p.parquet")
    result = runner.invoke(app, ["data", "query", "-d", "sales", "-q", "SELECT sum(id) FROM sales"])
    assert result.exit_code == 0 and "3" in result.output, result.output


def test_code_snapshots_go_to_the_bucket(tmp_path: Path, gitleaks: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "train.py").write_text("print('hi')\n")
    snap = snapshot.take(repo)
    assert storage.fs().exists(snapshot.archive_key(snap.hash))
    assert snapshot.take(repo).hash == snap.hash  # the same code again: refreshed, not touched
    snapshot.checkout(snap.hash[:12], tmp_path / "out")
    assert (tmp_path / "out" / "train.py").read_text() == "print('hi')\n"


def test_init_checks_the_bucket(monkeypatch: pytest.MonkeyPatch, bucket_root: str) -> None:
    assert "atomic commits work" in setup_project._storage()
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://no-such-bucket-fieldbook/x")
    line = setup_project._storage()
    assert "s3://no-such-bucket-fieldbook/x/test: write failed" in line and "see design" in line


def test_unsupported_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "hdfs://cluster/data")
    result = runner.invoke(app, ["data", "ls"])
    assert result.exit_code == 1 and "unsupported storage root" in result.output


def test_init_detects_a_store_that_ignores_conditional_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s3fs

    monkeypatch.setattr(s3fs.S3FileSystem, "call_s3", lambda self, *args, **kwargs: {})
    line = setup_project._storage()
    assert "conditional write failed" in line and "concurrent dataset commits" in line


@pytest.fixture
def open_data(s3_endpoint: str) -> str:
    """Someone else's bucket with a few objects (one hidden, one under a folder)."""
    import boto3

    client = boto3.client("s3", endpoint_url=s3_endpoint)
    bucket = f"open-{uuid.uuid4().hex[:10]}"
    client.create_bucket(Bucket=bucket)
    objects = {"cats/a/1.txt": b"one", "cats/a/2.txt": b"two", "cats/3.jpg": b"\xff\xd8three"}
    objects |= {"cats/.hidden": b"x", "dogs/4.txt": b"four"}
    for key, body in objects.items():
        client.put_object(Bucket=bucket, Key=key, Body=body)
    return f"s3://{bucket}"


def test_adding_from_a_bucket_reads_its_objects_into_the_dataset(open_data: str) -> None:
    import hashlib

    result = runner.invoke(app, ["data", "add", "cats", f"{open_data}/cats/", "--prefix", "raw/"])
    assert result.exit_code == 0, result.output
    assert "reading" in result.output and "added 3 items" in result.output
    cats = fb.dataset("cats")
    rows = {r["path"]: r for r in cats.files().to_pylist()}
    assert sorted(rows) == ["raw/3.jpg", "raw/a/1.txt", "raw/a/2.txt"]  # .hidden skipped
    assert rows["raw/a/1.txt"]["hash"] == hashlib.sha256(b"one").hexdigest()
    assert (rows["raw/a/1.txt"]["size"], rows["raw/3.jpg"]["mime_type"]) == (3, "image/jpeg")
    with cats.open("raw/a/2.txt") as f:
        assert f.read() == b"two"
    assert cats.log()[0].inputs == (f"{open_data}/cats",)  # where it came from
    with fb.dataset("one").edit() as edit:  # a single object, through the Python API
        edit.add_dir(f"{open_data}/dogs/4.txt")
    assert [r["path"] for r in fb.dataset("one").files().to_pylist()] == ["4.txt"]
    result = runner.invoke(app, ["data", "add", "none", f"{open_data}/nothing-here/"])
    assert result.exit_code == 1 and "no objects found" in result.output


def test_tables_in_a_bucket_are_imported_as_rows(open_data: str, s3_endpoint: str) -> None:
    import boto3

    s3_client = boto3.client("s3", endpoint_url=s3_endpoint)
    bucket = open_data.removeprefix("s3://")
    s3_client.put_object(Bucket=bucket, Key="sales/a.csv", Body=b"id,amount\n1,2.5\n2,4\n")
    s3_client.put_object(Bucket=bucket, Key="sales/b.tsv", Body=b"id\tamount\n3\t1\n")
    result = runner.invoke(app, ["data", "add", "sales", f"{open_data}/sales/"])
    assert result.exit_code == 0 and "added 3 rows" in result.output, result.output
    sales = fb.dataset("sales")
    assert sales.kind == "records" and sales.to_table("amount > 1").num_rows == 2
    assert sales.log()[0].inputs == (f"{open_data}/sales",)


def test_a_large_file_is_streamed_into_the_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import sys

    monkeypatch.setattr(sys.modules["fieldbook.data.dataset"], "STREAM_ABOVE", 1000)
    content = os.urandom(6_000_000)  # above Lance's 4 MB: a data file of its own
    (tmp_path / "big.bin").write_bytes(content)
    (tmp_path / "small.txt").write_text("small")
    ds = fb.dataset("big")
    with ds.edit() as edit:
        edit.add("big.bin", tmp_path / "big.bin")
        edit.add("small.txt", tmp_path / "small.txt")
    assert ds.read_many(["big.bin", "small.txt"]) == [content, b"small"]
    assert fb.data.verify(deep=True).ok
