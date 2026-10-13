"""Storage module: resolves the storage root and provides one filesystem interface.

All platform reads and writes go through URIs and fsspec, never hard-coded local paths
(storage.md §1). Each project keeps everything in its own folder of the storage root,
`<root>/<project>/`, so several projects can share a bucket with permissions per project;
paths here are relative to that folder. The root is local (`file://`) or an AWS S3 bucket
(`s3://`; other clouds are deferred). Credentials come from AWS's standard sources, never
from fieldbook's settings (storage.md §3).
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.dirfs import DirFileSystem

from fieldbook.clouds import aws
from fieldbook.config import Settings, settings

# Packages that provide each scheme to fsspec, for error messages.
EXTRAS = {"s3": "s3"}


class StorageError(Exception):
    pass


def root(cfg: Settings | None = None) -> str:
    """The storage root URI, shared by projects, e.g. `s3://acme-data` or
    `file:///Users/alice/vision/.fieldbook/data`."""
    return (cfg or settings()).storage_root


def project_root(cfg: Settings | None = None) -> str:
    """The project's folder in the storage root, e.g. `s3://acme-data/vision`."""
    cfg = cfg or settings()
    return f"{cfg.storage_root}/{cfg.project}"


def scheme(cfg: Settings | None = None) -> str:
    return root(cfg).split("://", 1)[0]


def fsspec_options(cfg: Settings | None = None) -> dict:
    """Location settings for the root's fsspec filesystem (s3fs)."""
    cfg = cfg or settings()
    kind = scheme(cfg)
    if kind == "file":
        return {"auto_mkdir": True}  # create parent directories on write, as object stores do
    if kind == "s3":
        options: dict = {}
        if cfg.storage_endpoint:
            options["endpoint_url"] = cfg.storage_endpoint
        if region := s3_region(cfg):
            options["client_kwargs"] = {"region_name": region}
        return options
    return {}


def s3_region(cfg: Settings) -> str | None:
    """The region of an S3 root: `storage_region`, else, on AWS (no endpoint), the bucket's
    own region. Requests signed for another region are refused as "Access Denied", so a
    bucket outside the default region would otherwise fail."""
    if cfg.storage_region or cfg.storage_endpoint or scheme(cfg) != "s3":
        return cfg.storage_region
    bucket = root(cfg).removeprefix("s3://").split("/", 1)[0]
    if bucket not in _regions:
        _regions[bucket] = _known_region(cfg, bucket)
    return _regions[bucket]


def _known_region(cfg: Settings, bucket: str) -> str | None:
    """A bucket's region never changes, so it is looked up once per machine and kept in the
    state folder; every later `fieldbook` command reads it from there."""
    path = cfg.state_dir / "bucket-regions.json"
    try:
        known = json.loads(path.read_text())
    except (OSError, ValueError):
        known = {}
    if bucket in known:
        return known[bucket]
    region = aws.bucket_region(bucket)
    if region:  # not when offline: try again next time
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**known, bucket: region}, indent=2, sort_keys=True))
        except OSError:
            pass
    return region


_regions: dict[str, str | None] = {}


def lance_options(cfg: Settings | None = None) -> dict[str, str]:
    """The same location settings, as Lance's `storage_options`."""
    cfg = cfg or settings()
    kind = scheme(cfg)
    options: dict[str, str] = {}
    if kind == "s3":
        if cfg.storage_endpoint:
            options["endpoint"] = cfg.storage_endpoint
            if cfg.storage_endpoint.startswith("http://"):
                options["allow_http"] = "true"
        if region := s3_region(cfg):
            options["region"] = region
    return options


def base_fs(cfg: Settings | None = None):
    """The root's fsspec filesystem and the root's path in it."""
    cfg = cfg or settings()
    try:
        return url_to_fs(root(cfg), **fsspec_options(cfg))
    except ImportError:
        extra = EXTRAS.get(scheme(cfg), "all")
        raise StorageError(
            f"{scheme(cfg)}:// storage needs pip install 'fieldbook[{extra}]'"
        ) from None


def fs(cfg: Settings | None = None) -> DirFileSystem:
    """An fsspec filesystem whose paths are relative to the project's folder."""
    cfg = cfg or settings()
    base, path = base_fs(cfg)
    path = f"{path}/{cfg.project}"
    if scheme(cfg) == "file":
        base.makedirs(path, exist_ok=True)  # buckets are created by their owner, not here
    return DirFileSystem(path=path, fs=base)


def uri(rel: str, cfg: Settings | None = None) -> str:
    """Full URI of a path relative to the project's folder."""
    return f"{project_root(cfg)}/{rel.lstrip('/')}"


def local_path(rel: str, cfg: Settings | None = None) -> Path | None:
    """Local filesystem path of a project-relative path, or None if the root is not local."""
    r = project_root(cfg)
    if not r.startswith("file://"):
        return None
    return Path(r.removeprefix("file://")) / rel


def location(rel: str, cfg: Settings | None = None) -> str:
    """What other tools (Lance, DuckDB, PIL, Hugging Face, ...) accept: a plain path for a
    local root, otherwise the URI (e.g. `s3://...`)."""
    local = local_path(rel, cfg)
    return str(local) if local is not None else uri(rel, cfg)


def check(cfg: Settings | None = None) -> str:
    """Check a bucket root (storage.md §6): credentials, write, read, list, and
    delete, and, on S3 APIs, that "write only if absent" is enforced, which concurrent
    dataset commits rely on. Returns a summary; raises StorageError naming the failed step."""
    cfg = cfg or settings()
    if scheme(cfg) == "file":
        return f"storage: {project_root(cfg)}"
    probe = f"tmp/fieldbook-check-{uuid.uuid4().hex}"
    steps = []
    try:
        files = fs(cfg)
        steps.append("write")
        try:
            files.pipe(probe, b"ok")
        except PermissionError:  # a project's viewer may read only (sharing.md §2)
            try:
                files.ls("", detail=False)
            except FileNotFoundError:  # nothing stored yet
                pass
            return f"storage: {project_root(cfg)} (read-only access: listing works)"
        steps.append("read")
        if files.cat(probe) != b"ok":
            raise StorageError("read back different content")
        steps.append("list")
        files.ls("tmp", detail=False)
        if scheme(cfg) == "s3":
            steps.append("conditional write")
            _check_conditional_put(cfg, probe)
        steps.append("delete")
        files.rm(probe)
    except Exception as e:  # credentials, permissions, network: report, do not crash init
        if steps:  # the probe may exist: do not leave it behind
            try:
                fs(cfg).rm(probe)
            except Exception:
                pass
        detail = str(e) if isinstance(e, StorageError) else f"{type(e).__name__}: {e}"
        step = steps[-1] if steps else "connect"
        raise StorageError(f"storage: {project_root(cfg)}: {step} failed: {detail}") from None
    return f"storage: {project_root(cfg)} (write, read, list, delete, and atomic commits work)"


def _check_conditional_put(cfg: Settings, probe: str) -> None:
    base, path = base_fs(cfg)
    bucket, key, _ = base.split_path(f"{path}/{cfg.project}/{probe}")
    try:
        base.call_s3("put_object", Bucket=bucket, Key=key, Body=b"again", IfNoneMatch="*")
    except Exception as e:
        # Refused because the object exists: S3's 412 PreconditionFailed, which s3fs versions
        # report differently (older ones as OSError "pre-conditions ... did not hold").
        refused = ("PreconditionFailed", "412", "pre-conditions you specified did not hold")
        if any(text in repr(e) for text in refused) or isinstance(e, FileExistsError):
            return
        raise
    raise StorageError(
        "the store accepted a second write of an existing object despite If-None-Match, so "
        "concurrent dataset commits could be lost; use AWS S3"
    )


def refresh(files, path: str) -> None:
    """Give an existing file a new modification time (cleanup's grace periods count from it).
    Object stores cannot touch an existing object: on S3 it is copied onto itself (one
    request, nothing downloaded); objects too large for that (over 5 GB) are rewritten
    through a temporary copy."""
    try:
        files.touch(path, truncate=False)
        return
    except (NotImplementedError, ValueError):  # s3fs
        pass
    base = getattr(files, "fs", files)
    if hasattr(base, "call_s3"):
        full = f"{files.path}/{path}" if hasattr(files, "path") else path
        bucket, key, _ = base.split_path(full)
        try:
            base.call_s3("copy_object", Bucket=bucket, Key=key, MetadataDirective="REPLACE",
                         CopySource={"Bucket": bucket, "Key": key},
                         Metadata={"refreshed": str(int(time.time()))})  # fmt: skip
            return
        except Exception as e:  # e.g. over 5 GB, which needs a multipart copy
            if "InvalidRequest" not in repr(e) and "EntityTooLarge" not in repr(e):
                raise
    tmp = f"tmp/{uuid.uuid4().hex}"
    files.copy(path, tmp)
    move(files, tmp, path)


def modified_time(info: dict) -> float:
    """Last-modified time (epoch seconds) from an fsspec info dict."""
    if "mtime" in info:
        return float(info["mtime"])
    for key in ("LastModified", "last_modified", "updated"):  # S3, Azure, GCS
        value = info.get(key)
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value is not None:
            return value.timestamp()
    raise KeyError("no modification time in file info")


def is_local(files: AbstractFileSystem) -> bool:
    """Whether `files` (possibly a DirFileSystem) is on local disk."""
    base = getattr(files, "fs", files)
    protocol = base.protocol if isinstance(base.protocol, str) else base.protocol[0]
    return protocol in ("file", "local")


def fresh(files: AbstractFileSystem) -> AbstractFileSystem:
    """`files`, with what it remembers of earlier listings forgotten. s3fs keeps listings
    for the life of the process, so a loop that watches for another machine's files (a
    run's end, its next piece of output) would otherwise see its first look for ever."""
    getattr(files, "fs", files).invalidate_cache()
    return files


def makedirs(files: AbstractFileSystem, path: str) -> None:
    """Create a folder of the store on local disk. Never in a bucket: object stores have no
    folders, and s3fs, asked for one where it may not look at the bucket itself (a run
    machine's role, a member's policy: only the project's folder), tries to create the
    bucket and fails."""
    if is_local(files):
        files.makedirs(path, exist_ok=True)


def move(files: AbstractFileSystem, src: str, dst: str) -> None:
    """Move within the store. On local disk the destination's folder is created first:
    older fsspec versions do not (object stores have no folders)."""
    if is_local(files) and "/" in dst:
        files.makedirs(dst.rsplit("/", 1)[0], exist_ok=True)
    files.mv(src, dst)
