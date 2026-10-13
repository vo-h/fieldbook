"""Files in someone else's S3 bucket as the source of an add (data.md §5).

`fieldbook data add NAME s3://bucket/prefix` reads them into the dataset's table: the bytes
pass through the machine doing the add, whatever the storage root (a dataset's files are
rows of its table, not objects S3 could copy). Public buckets need no AWS login.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fieldbook import storage
from fieldbook.config import Settings

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

SCHEME = "s3://"


def is_bucket(src: object) -> bool:
    return isinstance(src, str) and src.startswith(SCHEME)


@dataclass(frozen=True)
class RemoteFile:
    fs: AbstractFileSystem  # reads the source bucket
    key: str  # bucket/key
    size: int

    @property
    def uri(self) -> str:
        return f"{SCHEME}{self.key}"


def filesystem(cfg: Settings) -> AbstractFileSystem:
    """An S3 filesystem for a source bucket: this machine's AWS login if it has one, else
    anonymous. Buckets are looked up in their own regions."""
    try:
        import botocore.session
        import s3fs
    except ImportError:
        raise storage.StorageError("s3:// sources need pip install 'fieldbook[s3]'") from None
    options: dict = {"cache_regions": True}
    if cfg.storage_endpoint:  # an S3-compatible endpoint (tests): the source is there too
        options = {"endpoint_url": cfg.storage_endpoint}
    anonymous = botocore.session.get_session().get_credentials() is None
    return s3fs.S3FileSystem(anon=anonymous, **options)


def files(
    src: str, prefix: str, cfg: Settings, *, include_hidden: bool = False
) -> list[tuple[str, RemoteFile]]:
    """(item path, file) for the object at `src`, or every object under it, sorted by path.
    As for local directories, names starting with "." are skipped unless asked for."""
    fs = filesystem(cfg)
    root = src.removeprefix(SCHEME).rstrip("/")
    try:
        found = fs.find(root, detail=True)
    except FileNotFoundError:
        found = {}
    except PermissionError as e:
        raise storage.StorageError(f"{src}: access denied ({e})") from None
    if not found:
        raise storage.StorageError(f"{src}: no objects found")
    items = []
    for key, info in sorted(found.items()):
        if key.endswith("/"):
            continue  # a folder marker
        relative = key[len(root) :].lstrip("/") or key.rsplit("/", 1)[-1]
        if include_hidden or not any(part.startswith(".") for part in relative.split("/")):
            items.append((prefix + relative, RemoteFile(fs, key, int(info.get("size") or 0))))
    return items


def bucket(src: str) -> str:
    return src.removeprefix(SCHEME).split("/", 1)[0]
