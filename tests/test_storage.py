import hashlib
from pathlib import Path

import pytest

from fieldbook import storage
from fieldbook.config import normalize_root, settings


def test_normalize_root() -> None:
    home = str(Path.home().resolve())
    assert normalize_root("file://~/data") == f"file://{home}/data"
    assert normalize_root("~/data") == f"file://{home}/data"
    assert normalize_root("s3://bucket/prefix/") == "s3://bucket/prefix"


def test_default_root(clean_env: Path) -> None:
    """Without a storage root, data stays in the project's .fieldbook folder."""
    assert settings().storage_root == f"file://{(clean_env / '.fieldbook' / 'data').resolve()}"


def test_paths_are_relative_to_the_projects_folder(storage_root: Path) -> None:
    """Projects share a storage root; each keeps everything in `<root>/<project>/`."""
    fs = storage.fs()
    with fs.open("x/y.txt", "wb") as f:
        f.write(b"hi")
    folder = storage_root.resolve() / "test"  # FIELDBOOK_PROJECT=test (conftest)
    assert (folder / "x" / "y.txt").read_bytes() == b"hi"
    assert storage.uri("x/y.txt") == f"file://{folder}/x/y.txt"
    assert storage.local_path("x/y.txt") == folder / "x" / "y.txt"
    assert storage.project_root() == f"file://{folder}"


def test_lance_is_imported_on_first_use() -> None:
    """Lance takes over a second to import; commands that do not open datasets skip it."""
    import subprocess
    import sys

    code = (
        "import sys, fieldbook.cli, fieldbook.data.lineage\n"
        "assert 'lance' not in sys.modules\n"
        "import fieldbook as fb\n"
        "fb.dataset('x').exists\n"
        "assert 'lance' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_buckets_region_is_looked_up_once_per_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: kept in the state folder, so later commands do not ask the network."""
    from fieldbook.clouds import aws

    lookups = []
    monkeypatch.setattr(aws, "bucket_region", lambda bucket: lookups.append(bucket) or "eu-west-1")
    monkeypatch.setattr(storage, "_regions", {})
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://some-bucket/fieldbook")
    assert storage.s3_region(settings()) == "eu-west-1"
    monkeypatch.setattr(storage, "_regions", {})  # a new process
    assert storage.s3_region(settings()) == "eu-west-1"
    assert lookups == ["some-bucket"]


def test_unreadable_datasets_are_not_reported_missing(tmp_path: Path) -> None:
    """B10: only "not found" means missing; permission or credential errors are errors."""
    import fieldbook as fb
    from fieldbook.data import DatasetError

    assert not fb.dataset("nothing").exists
    ds = fb.dataset("locked")
    with ds.edit() as edit:
        (tmp_path / "a.txt").write_text("a")
        edit.add("a.txt", tmp_path / "a.txt")
    folder = Path(ds._real_uri.removeprefix("file://"))
    folder.chmod(0)
    try:
        with pytest.raises(DatasetError, match="cannot read dataset"):
            assert fb.dataset("locked").exists
    finally:
        folder.chmod(0o755)


def test_adding_from_a_bucket_downloads_to_a_local_root(storage_root: Path, monkeypatch) -> None:
    """The objects are read into the dataset's table, and hashed as they arrive."""
    import uuid

    import boto3
    from moto.server import ThreadedMotoServer

    import fieldbook as fb

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        for name, value in {"AWS_ACCESS_KEY_ID": "t", "AWS_SECRET_ACCESS_KEY": "t",
                            "AWS_DEFAULT_REGION": "us-east-1"}.items():  # fmt: skip
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("FIELDBOOK_STORAGE_ENDPOINT", endpoint)  # where the source bucket is
        bucket = f"open-{uuid.uuid4().hex[:10]}"
        client = boto3.client("s3", endpoint_url=endpoint)
        client.create_bucket(Bucket=bucket)
        client.put_object(Bucket=bucket, Key="set/a.txt", Body=b"alpha")
        with fb.dataset("pulled").edit() as edit:
            assert edit.add_dir(f"s3://{bucket}/set") == 1
        ds = fb.dataset("pulled")
        with ds.open("a.txt") as f:
            assert f.read() == b"alpha"
        assert ds.log()[0].inputs == (f"s3://{bucket}/set",)
        (row,) = ds.files().to_pylist()
        assert (row["size"], row["hash"]) == (5, hashlib.sha256(b"alpha").hexdigest())
        assert not (storage_root / "test" / "objects").exists()
    finally:
        server.stop()


def test_folders_are_created_on_local_disk_only() -> None:
    """In a bucket, under permissions for the project's folder only (a run machine's role),
    s3fs answers a request for a folder by trying to create the bucket (seen on a real
    machine: the run's output and end were not kept). Code writes through
    `storage.makedirs`, which does nothing in a bucket."""
    import re
    from pathlib import Path

    import fsspec

    class Bucket(fsspec.AbstractFileSystem):
        protocol = "s3"

        def makedirs(self, path, exist_ok=False):  # noqa: ANN001, ANN202
            raise PermissionError("s3:CreateBucket")

    storage.makedirs(Bucket(), "runs/x")  # nothing asked of the bucket
    direct = [
        f"{path.name}:{number}"
        for path in Path(storage.__file__).parent.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\b(files|fs)\.makedirs\(", line)
        and path.name != "storage.py"  # which checks for local disk
    ]
    assert direct == []


def test_a_fresh_look_forgets_earlier_listings() -> None:
    """Seen on a real run: `fieldbook log wait` kept seeing its first listing of the bucket, so
    a run that had finished looked as if it had stopped giving signs of life."""
    import fsspec

    forgotten = []

    class Bucket(fsspec.AbstractFileSystem):
        protocol = "s3"

        def invalidate_cache(self, path=None):  # noqa: ANN001, ANN202
            forgotten.append(path)

    bucket = Bucket()
    assert storage.fresh(bucket) is bucket and forgotten == [None]
    from fsspec.implementations.dirfs import DirFileSystem

    storage.fresh(DirFileSystem(path="root/project", fs=bucket))  # the project's folder view
    assert len(forgotten) == 2
