"""Versioned datasets (data.md).

A dataset is one Lance table. In a files dataset a row is an item: its path and metadata,
and the file's bytes in a blob column (`data`). In a records dataset a row is a record of
tabular data (`records.py`). Each committed edit creates a new Lance version; versions
share the data they do not change.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import mimetypes
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from fieldbook import storage
from fieldbook.config import NAME_RE, Settings, settings
from fieldbook.data import lineage as _lineage
from fieldbook.data import records as _records
from fieldbook.data import smoke as _smoke
from fieldbook.data._lazy import lance, lance_commit, lance_dataset
from fieldbook.data.remote import RemoteFile, is_bucket
from fieldbook.data.remote import files as bucket_files
from fieldbook.launch import agents as _agents
from fieldbook.launch import context

if TYPE_CHECKING:
    from fieldbook.data.audit import VerifyReport
    from fieldbook.data.cleanup import CleanupReport

DATASETS_DIR = "datasets"
BASE_COLUMNS = ("path", "hash", "size", "mime_type", "added_at")
DATA = "data"  # the column holding each file's bytes (Lance's blob encoding)
RESERVED_COLUMNS = (*BASE_COLUMNS, DATA)
FILE_FORMAT = "2.2"  # Lance's file format: the first with its blob layout
BATCH_FILES = 2000  # files read into memory at a time when saving...
BATCH_BYTES = 256 * 1024 * 1024  # ...or this much, whichever comes first
CHECKOUT_BATCH = 256  # files fetched together when writing a dataset out
# A local file above this size is not read into memory to save it: Lance is given its
# address and copies it in a piece at a time (a 2 GB file: 0.13 GB of memory, not 4 GB).
# Smaller files are faster read here (20,000 of 4 KB: 1.4 s against 2.1 to 2.5 s).
STREAM_ABOVE = 64 * 1024 * 1024
HASH_CHUNK = 16 * 1024 * 1024
INTERNAL_KEY = "fieldbook.internal"  # on a commit that is a step of an edit, not a version
_ORIGIN, _CONTENT = "_origin", "_content"  # a staged row's bookkeeping (see Edit)
BASE_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("hash", pa.string()),
        ("size", pa.int64()),
        ("mime_type", pa.string()),
        ("added_at", pa.timestamp("us", tz="UTC")),
    ]
)
COMMIT_MESSAGE_KEY = "__lance_commit_message"
MAX_COMMIT_ATTEMPTS = 20
# Naming this many paths in one filter costs more than reading every path (27 microseconds
# a path): above it, rows are found by reading the path column whole.
REWRITE_ABOVE_REMOVALS = 5000
INFO_KEY = b"fieldbook.info"
_KEEP: Any = object()  # sentinel: an edit leaves the schema unchanged
DEFAULT_WORKERS = 8
REMOTE_WORKERS = 32  # copies inside S3 are requests that wait on S3, not on this machine
_NAME_RE = NAME_RE


class DatasetError(Exception):
    pass


class InvalidNameError(DatasetError, ValueError):
    """A dataset or tag name that is not allowed (also a ValueError, as before)."""


class DatasetNotFoundError(DatasetError):
    pass


class ConflictError(DatasetError):
    """A concurrent edit changed items that this edit also changes."""


@dataclass(frozen=True)
class VersionInfo:
    version: int
    timestamp: datetime
    message: str
    author: str
    run_id: str
    rows: int
    tags: tuple[str, ...]
    inputs: tuple[str, ...] = ()  # "project/name@version" or outside sources (§2.11)
    command: str = ""
    code: str = ""  # code snapshot hash, when saved in a launched run
    git: str = ""  # `<commit> <branch> clean|dirty` at the snapshot, when saved in a launched run
    env: str = ""  # environment record (JSON), when saved in a launched run
    agent: str = ""  # the coding agent that saved it for the author (runs.md §20)


def dataset(name: str, version: int | str | None = None) -> Dataset:
    """Open a dataset in the current project. It is created by its first edit. Projects do
    not read each other's datasets (sharing.md §4)."""
    from fieldbook.config import require_project

    require_project()
    return Dataset(name, version)


def list_datasets(cfg: Settings | None = None) -> list[str]:
    """The current project's datasets."""
    fs = storage.fs(cfg or settings())
    if not fs.exists(DATASETS_DIR):
        return []
    names = [p.rsplit("/", 1)[-1] for p in fs.ls(DATASETS_DIR, detail=False)]
    return sorted(n.removesuffix(".lance") for n in names if n.endswith(".lance"))


PARALLEL_READS = 16  # concurrent reads of version records; each is a round trip in a bucket


class Dataset:
    def __init__(
        self,
        name: str,
        version: int | str | None = None,
        *,
        cfg: Settings | None = None,
    ):
        if not _NAME_RE.match(name):
            raise InvalidNameError(
                f"invalid dataset name: {name!r} (letters, digits, '.', '_', '-'; "
                "must start with a letter or digit)"
            )
        self.cfg = cfg or settings()
        self.name = name
        self.project = self.cfg.project
        self.ref = version
        self._real_uri = storage.location(f"{DATASETS_DIR}/{name}.lance", self.cfg)
        self._files = storage.fs(self.cfg)
        self._lookup: tuple[int, dict[str, int]] | None = None  # (version, path -> row address)
        self._pinned: lance.LanceDataset | None = None

    def __repr__(self) -> str:
        ref = "" if self.ref is None else f"@{self.ref}"
        return f"Dataset({self.project}/{self.name}{ref})"

    @property
    def _scratch(self) -> str | None:
        return _smoke.scratch_id()

    @property
    def _scratch_rel(self) -> str | None:
        scratch = self._scratch
        return _smoke.scratch_path(scratch, self.name) if scratch else None

    @property
    def in_scratch(self) -> bool:
        """Whether this view reads a smoke run's scratch area: an unpinned view of a
        dataset that the smoke run has saved (runs.md §9)."""
        return (
            self.ref is None
            and self._scratch_rel is not None
            and self._files.exists(self._scratch_rel)
        )

    @property
    def _table_uri(self) -> str:
        """The table this view reads: the scratch copy if `in_scratch`, else the dataset."""
        if self.in_scratch:
            return storage.location(self._scratch_rel, self.cfg)
        return self._real_uri

    def _require_writable(self) -> None:
        if os.environ.get("FIELDBOOK_SCRATCH") and not _smoke.writes_to_scratch():
            raise DatasetError("FIELDBOOK_SCRATCH is a read-only view of a kept scratch area")

    def _write_uri(self) -> str:
        if _smoke.writes_to_scratch():
            return storage.location(self._scratch_rel, self.cfg)
        return self._real_uri

    # --- reading -------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """Whether the dataset exists. Errors other than "not found" (credentials,
        permissions, network) are raised, not reported as a missing dataset."""
        try:
            _open(self._table_uri, self.cfg)
        except FileNotFoundError:
            return False
        except (ValueError, OSError) as e:
            if _is_not_found(e):
                return False
            raise DatasetError(f"cannot read dataset {self.key}: {e}") from None
        return True

    def _lance(self, version: int | str | None = None) -> lance.LanceDataset:
        try:
            return _open(self._table_uri, self.cfg, version)
        except (FileNotFoundError, ValueError, OSError) as e:
            if not self.exists:
                raise DatasetNotFoundError(f"dataset {self.project}/{self.name} not found") from e
            raise DatasetError(f"version {version!r} of {self.name} not found") from e

    @property
    def version(self) -> int:
        """The version this view shows (the latest if unpinned). Lance also numbers the
        steps of an edit and its own housekeeping (a compaction); those are not versions
        of the dataset, and the view then shows the version they belong to."""
        return _shown(self.index())

    def index(self) -> lance.LanceDataset:
        """The dataset index at this view's version."""
        ref = self.ref if self.ref is not None else _lineage.pinned(self.name)
        if ref is None or (self.ref is None and self.in_scratch):
            return self._lance()
        if self._pinned is None:
            pinned = self._lance(ref)
            if not _is_version(pinned, pinned.version):
                raise DatasetError(
                    f"{ref} is not a version of {self.name} (a step of an edit, or "
                    f"housekeeping): `fieldbook data ls {self.name} --versions` lists them"
                )
            self._pinned = pinned
        return self._pinned

    @property
    def kind(self) -> str:
        """What the dataset holds: "files" (a row is a file) or "records" (a row is a
        record of tabular data). Decided by what was first added, and kept."""
        return _records.kind_of(self.index().schema)

    def _files_only(self, index: lance.LanceDataset, instead: str) -> None:
        if _records.kind_of(index.schema) == _records.RECORDS:
            raise DatasetError(_records.NOT_FILES.format(name=self.name, instead=instead))

    def to_table(
        self,
        where: str | None = None,
        *,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> pa.Table:
        """The rows of a records dataset, optionally only those matching a SQL filter
        (e.g. `"price > 10 AND region = 'EU'"`), only some columns, or the first `limit`.
        A smoke run reads a sample."""
        index = self.index()
        if _records.kind_of(index.schema) != _records.RECORDS:
            raise DatasetError(
                f"{self.name} holds files, not records: list them with files(), read them "
                "with open() and read_many()"
            )
        if (n := _smoke.sample_size()) is not None:
            limit = min(limit or n * _smoke.ROWS_PER_ITEM, n * _smoke.ROWS_PER_ITEM)
        try:
            table = index.to_table(
                columns=list(columns) if columns is not None else None, filter=where, limit=limit
            )
        except (ValueError, OSError) as e:
            raise DatasetError(f"invalid filter or columns: {_first_line(e)}") from None
        _lineage.record_read(self, _shown(index))
        return table

    def files(self, prefix: str = "", where: str | None = None) -> pa.Table:
        """Index rows (without the files' bytes) whose path starts with `prefix`,
        optionally filtered by a SQL expression over index columns (e.g. `"label = 'cat'
        AND size > 1000"`), sorted by path (saves append rows, so the stored order is not).
        """
        index = self.index()
        self._files_only(index, _records.READ_ROWS)
        # Lance applies the prefix while reading, which is far faster than loading every
        # row and filtering afterwards (1M items: 0.06 s instead of 1.6 s).
        conditions = [f"({where})"] if where else []
        if prefix:
            conditions.append("starts_with(path, '{}')".format(prefix.replace("'", "''")))
        try:
            table = index.to_table(columns=_columns(index), filter=" AND ".join(conditions) or None)
        except (ValueError, OSError) as e:
            raise DatasetError(f"invalid filter {where!r}: {_first_line(e)}") from None
        table = table.sort_by("path")
        _lineage.record_read(self, _shown(index))
        if (n := _smoke.sample_size()) is not None:
            table = _smoke.sample(table, prefix, n)
        return table

    def stats(
        self,
        prefix: str = "",
        where: str | None = None,
        *,
        depth: int = 0,
        by: Sequence[str] = (),
    ) -> pa.Table:
        """Item count and total size, optionally grouped.

        `depth` groups by the first `depth` directory levels below `prefix` (e.g. depth=1
        on `train/cat/1.jpg` gives `train`); files above that level are grouped as `.`.
        `by` groups by index columns (e.g. `["label"]`). Both can be combined, e.g. to
        see class balance per split.
        """
        table = self.files(prefix, where)
        keys = list(by)
        unknown = [k for k in keys if k not in table.column_names]
        if unknown:
            raise DatasetError(
                f"unknown columns {unknown}; available: {', '.join(table.column_names)}"
            )
        if depth:
            dirs = [_dir_prefix(p[len(prefix) :], depth) for p in table["path"].to_pylist()]
            table = table.append_column("dir", pa.array(dirs, pa.string()))
            keys.insert(0, "dir")
        if not keys:
            return pa.table(
                {
                    "files": pa.array([table.num_rows], pa.int64()),
                    "bytes": pa.array([pc.sum(table["size"]).as_py() or 0], pa.int64()),
                }
            )
        grouped = table.group_by(keys).aggregate([("path", "count"), ("size", "sum")])
        grouped = grouped.select([*keys, "path_count", "size_sum"])
        grouped = grouped.rename_columns([*keys, "files", "bytes"])
        return grouped.sort_by([(k, "ascending") for k in keys])

    def _where(self, paths: Sequence[str]) -> list[int]:
        """The row address of each item, from a path lookup table cached per version.

        Pinned views load the table once. Unpinned views check for a newer version on
        each call; pin a version in training loops for consistent, faster reads.
        """
        index = self.index()
        self._files_only(index, _records.READ_ROWS)
        if self._lookup is None or self._lookup[0] != index.version:
            table = index.to_table(columns=["path"], with_row_address=True)
            lookup = dict(
                zip(table["path"].to_pylist(), table["_rowaddr"].to_pylist(), strict=True)
            )
            self._lookup = (index.version, lookup)
        _lineage.record_read(self, _shown(index))
        try:
            return [self._lookup[1][path] for path in paths]
        except KeyError as e:
            raise KeyError(e.args[0]) from None

    def open(self, path: str, mode: str = "rb"):  # noqa: ANN201
        """Open an item's content for reading (a file-like object). One request a file: a
        training loop reads a batch with `read_many`."""
        if "r" not in mode or "+" in mode:
            raise ValueError("dataset items are read-only")
        (blob,) = self.index().take_blobs(DATA, addresses=self._where([path]))
        return blob

    def read_many(self, paths: Sequence[str]) -> list[bytes]:
        """The content of several items, in the order given, fetched together."""
        addresses = self._where(paths)
        found = dict(self.index().read_blobs(DATA, addresses=addresses))
        return [found[address] for address in addresses]

    def checkout(self, target: str | Path, prefix: str = "") -> int:
        """Write this view's files under `prefix` into the folder `target`, at their
        paths, for tools that need real files. Returns how many were written."""
        target = Path(target)
        paths = self.files(prefix)["path"].to_pylist()
        for start in range(0, len(paths), CHECKOUT_BATCH):
            batch = paths[start : start + CHECKOUT_BATCH]
            for path, content in zip(batch, self.read_many(batch), strict=True):
                out = target / path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(content)
        return len(paths)

    def info(self) -> dict[str, str]:
        """Dataset-level info (e.g. description, license, source) at this view's version."""
        return read_info(self.index().schema)

    def schema(self) -> str | None:
        """The dataset schema (Pandera YAML) at this view's version, or None."""
        from fieldbook.data.schema import read_schema

        return read_schema(self.index().schema)

    def diff(self, other: int | str, prefix: str = "") -> pa.Table:
        """Changes from version `other` to this view's version: for files, see
        `audit.diff`; for records, the rows removed and added (`audit.diff_records`)."""
        from fieldbook.data.audit import diff, diff_records

        if self.kind == _records.RECORDS:
            if prefix:
                raise DatasetError(f"{self.name} holds records: its rows have no paths")
            return diff_records(Dataset(self.name, other, cfg=self.cfg), self)
        return diff(Dataset(self.name, other, cfg=self.cfg), self, prefix)

    def verify(self, *, deep: bool = False) -> VerifyReport:
        """Check stored files for every version. See `audit.verify`."""
        from fieldbook.data.audit import verify

        return verify(self, deep=deep)

    def tags(self) -> dict[str, int]:
        return {tag: info["version"] for tag, info in self._lance().tags.list().items()}

    def log(self) -> list[VersionInfo]:
        """Version history, newest first.

        Each version's commit record is a separate read. In a bucket every read is a round
        trip, so they are read in parallel, each thread with its own handle (a Lance handle
        cannot be shared between threads); on local disk they are read in order from one
        handle."""
        ds = self._lance()
        tags_by_version = _tags_by_version(ds)
        entries = ds.versions()
        if storage.scheme(self.cfg) == "file" or len(entries) <= 2:
            history = [_read_version_info(ds, entry, tags_by_version) for entry in entries]
        else:
            handles = threading.local()

            def read(entry: dict[str, Any]) -> VersionInfo:
                if not hasattr(handles, "ds"):
                    handles.ds = _open(self._table_uri, self.cfg)
                return _read_version_info(handles.ds, entry, tags_by_version)

            with ThreadPoolExecutor(max_workers=min(PARALLEL_READS, len(entries))) as pool:
                history = list(pool.map(read, entries))
        # A step of an edit, or Lance's own housekeeping, is not a version of the dataset.
        return sorted((h for h in history if h is not None), key=lambda h: h.version, reverse=True)

    def lineage(self) -> list[_lineage.LineageNode]:
        """This view's version and its inputs, recursively (§2.11)."""
        return _lineage.lineage(self)

    def _version_info(self, version: int) -> VersionInfo:
        """Info for one version; DatasetError if the dataset or version does not exist."""
        ds = self._lance()  # DatasetNotFoundError if the dataset does not exist
        entry = next((v for v in ds.versions() if v["version"] == version), None)
        if entry is None:
            raise DatasetError(f"version {version} of {self.key} not found")
        info = _read_version_info(ds, entry, _tags_by_version(ds))
        if info is None:
            raise DatasetError(f"version {version} of {self.key} is a step of an edit, "
                               "not a version of its own")  # fmt: skip
        return info

    # --- writing -------------------------------------------------------------

    def edit(self, message: str = "", *, inputs: Iterable[Dataset | str] = ()) -> Edit:
        """Start an edit. Used as a context manager; commits on successful exit.

        Dataset versions read while the edit is open are recorded as its inputs. `inputs`
        adds others: datasets (at their current version), `name@version` strings, or
        sources outside the platform such as `hf://...` (§2.11).
        """
        if self.ref is not None:
            raise DatasetError("cannot edit a pinned version; open the dataset without a version")
        self._require_writable()
        return Edit(self, message, inputs)

    def revert(self, version: int | str, message: str | None = None) -> int:
        """Create a new version identical to `version` (number or tag). Returns its number."""
        if self.ref is not None:
            raise DatasetError("cannot revert a pinned version; open the dataset without a version")
        self._require_writable()
        latest = self._lance()
        target = Dataset(self.name, version, cfg=self.cfg).index()  # refuses what is no version
        if target.version == _shown(latest):
            return target.version
        if _smoke.skip(f"revert {self.key} to v{target.version}"):
            return latest.version
        transaction = lance_dataset.Transaction(  # the old version's data, not a copy of it
            read_version=latest.version,
            operation=lance.LanceOperation.Restore(target.version),
            transaction_properties=_properties(self, message or f"revert to v{target.version}", ()),
        )
        try:
            version = lance.LanceDataset.commit(
                self._write_uri(), transaction, max_retries=0,
                storage_options=storage.lance_options(self.cfg),
            ).version  # fmt: skip
            _committed(self, version)
            return version
        except lance_commit.CommitConflictError:
            raise ConflictError(f"{self.name} changed during the revert; try again") from None

    def tag(self, tag: str, version: int | None = None, *, move: bool = False) -> None:
        """Name a version (default: latest). `move=True` repoints an existing tag."""
        validate_tag(tag)
        self._require_writable()
        ds = self._lance()
        target = version if version is not None else _shown(ds)
        if target not in {v["version"] for v in ds.versions()} or not _is_version(ds, target):
            raise DatasetError(f"version {target} of {self.name} not found")
        if tag in ds.tags.list():
            if not move:
                raise DatasetError(
                    f"tag {tag!r} already exists on {self.name}; use move to repoint"
                )
            if _smoke.skip(f"tag {self.key}@v{target} as {tag}"):
                return
            ds.tags.update(tag, target)
        elif _smoke.skip(f"tag {self.key}@v{target} as {tag}"):
            return
        else:
            try:
                ds.tags.create(tag, target)
            except ValueError as e:  # created by someone else since the check above
                if "already exists" not in str(e):
                    raise
                raise DatasetError(
                    f"tag {tag!r} already exists on {self.name}; use move to repoint"
                ) from None

    def untag(self, tag: str) -> None:
        """Remove a tag. The version itself is kept."""
        self._require_writable()
        ds = self._lance()
        if tag not in ds.tags.list():
            raise DatasetError(f"tag {tag!r} not found on {self.name}")
        if not _smoke.skip(f"remove tag {tag} from {self.key}"):
            ds.tags.delete(tag)

    def delete(self, **options: Any) -> CleanupReport:
        """Delete the dataset with all its versions and tags, and free stored files no
        other dataset uses. Options: see `cleanup.delete_dataset`."""
        from fieldbook.data.cleanup import delete_dataset

        return delete_dataset(self, **options)

    def delete_versions(self, versions: Sequence[int], **options: Any) -> CleanupReport:
        """Delete specific versions and free stored files no remaining version uses.
        Options: see `cleanup.delete_versions`."""
        from fieldbook.data.cleanup import delete_versions

        return delete_versions(self, versions, **options)

    def prune(self, **options: Any) -> CleanupReport:
        """Delete versions not kept by the retention rules. Options: see `cleanup.prune`."""
        from fieldbook.data.cleanup import prune

        return prune(self, **options)

    def _record_version(
        self,
        message: str,
        index: lance.LanceDataset | None = None,
        replaced: Sequence[int] = (),
        by: Sequence[Any] = (),
    ) -> int:
        """Make the table a version of the dataset, with a record, holding the same items:
        as it is now (after Lance's own compaction, which leaves no record), or with the
        data files `replaced` given up for the new ones `by` (our compaction). The second
        is refused (Lance's CommitConflictError) if anything was saved after `index`."""
        index = index or self._lance()
        fragments = index.get_fragments()
        anchor = min(fragments, key=lambda f: f.fragment_id) if fragments else None
        operation = lance.LanceOperation.Update(
            removed_fragment_ids=list(replaced),
            updated_fragments=[anchor.metadata]
            if anchor is not None and anchor.fragment_id not in replaced else [],
            new_fragments=list(by),
        )  # fmt: skip
        transaction = lance_dataset.Transaction(
            read_version=index.version, operation=operation,
            transaction_properties=_properties(self, message, ()),
        )  # fmt: skip
        return lance.LanceDataset.commit(
            self._write_uri(), transaction, storage_options=storage.lance_options(self.cfg),
            **({"max_retries": 0} if replaced else {}),
        ).version  # fmt: skip

    def _remove_table(self) -> None:
        self._files.rm(f"{DATASETS_DIR}/{self.name}.lance", recursive=True)

    @property
    def key(self) -> str:
        """`project/name`, unique within a storage root."""
        return f"{self.project}/{self.name}"


class Edit:
    """A set of changes committed as one new dataset version (data.md §5).

    Changes are staged in one map from item path to its pending state: the item's new
    row, or None (a tombstone) if it is removed. This is the staging model of lakeFS and
    the git index. Every operation reads an item's current state (pending, else the saved
    version) and writes its new state, so operations compose without special cases.
    A pending row also says where it comes from: `_origin`, the saved row it continues
    (so a move or a change of metadata keeps the row and its stored file), and
    `_content`, where new bytes are to be read from. Changes of metadata to items the edit
    does not otherwise touch are kept apart, as the changed fields only (`_patches`): a
    relabel of a hundred thousand items reads none of their rows.

    A save never writes a file's bytes unless they are new (data.md §6): new files are
    appended, removed ones marked deleted, and changed metadata written as new column
    values for those rows, all in one Lance commit. Only an edit that adds a column or
    drops one takes a commit more, which is not a version of its own (`INTERNAL_KEY`).
    Every save also names the dataset's first data file as updated, so that Lance refuses
    the second of two concurrent saves; fieldbook then decides at path level whether it
    conflicts or is redone on the latest version.
    """

    def __init__(self, ds: Dataset, message: str, inputs: Iterable[Dataset | str] = ()):
        self.ds = ds
        self.message = message
        self.version: int | None = None
        self.inputs: set[str] = {_input_ref(i, ds) for i in inputs}
        # The version the edit starts from. Only its paths are loaded (to check what exists);
        # rows are fetched for the items the edit moves or replaces, and the whole index
        # (without the files) only if there is a schema to validate.
        # A rerun's first edit of a dataset starts where the original run's did (runs.md §17).
        start = None if ds.in_scratch else _lineage.pinned(ds.name, "FIELDBOOK_PIN_BASES")
        if start is None:
            self._base = _latest(ds)
        else:
            self._base = _open(ds._real_uri, ds.cfg, start) if start else None
        self._base_version = self._base.version if self._base is not None else None
        self._base_full: pa.Table | None = None
        # What the dataset holds, files or records: a new dataset's first operation decides.
        self._kind = None if self._base is None else _records.kind_of(self._base.schema)
        # Operations on rows, in the order given: ("add", source, delimiter),
        # ("remove", filter), ("update", filter, new values).
        self._row_ops: list[tuple] = []
        self.rows_added = 0
        files = self._base is not None and self._kind == _records.FILES
        paths = self._base.to_table(columns=["path"])["path"] if files else []
        self._base_index: dict[str, int] = {p: i for i, p in enumerate(_pylist(paths))}
        self._pending: dict[str, dict[str, Any] | None] = {}
        self._patches: dict[str, dict[str, Any]] = {}  # saved items: only the fields changed
        self._set_columns: set[str] = set()  # user columns set explicitly in this edit
        self._info: dict[str, str | None] = {}
        self._drops: set[str] = set()
        self._schema: str | None = _KEEP
        self._workers = DEFAULT_WORKERS
        self._progress: Callable[[int], None] | None = None
        self._appended: tuple[tuple[str, ...], list] | None = None  # (columns, fragments)
        self._closed = False
        _lineage.open_edit(self)

    def __enter__(self) -> Edit:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            _lineage.close_edit(self)

    # --- current state -------------------------------------------------------

    def _exists(self, path: str) -> bool:
        if path in self._pending:
            return self._pending[path] is not None
        return path in self._base_index

    def _rows(self, paths: Sequence[str]) -> list[dict[str, Any]]:
        """Current rows of existing items: pending state, else the saved row with the
        fields this edit changed, marked as continuing that row."""
        saved = [p for p in paths if p not in self._pending]
        loaded = {row["path"]: row for row in self._saved_rows(saved)}
        for path, row in loaded.items():
            row.update(self._patches.get(path, {}))
            row[_ORIGIN], row[_CONTENT] = path, None
        return [dict(self._pending[p]) if p in self._pending else loaded[p] for p in paths]

    def _stage(self, path: str, row: dict[str, Any] | None) -> None:
        self._pending[path] = row
        self._patches.pop(path, None)  # a staged row already holds them

    def _saved_rows(self, paths: Iterable[str]) -> list[dict[str, Any]]:
        """Rows of the starting version for those of `paths` it holds, read by position,
        without the files' bytes."""
        positions = sorted(self._base_index[p] for p in paths if p in self._base_index)
        if not positions:
            return []
        return self._base.take(positions, columns=_columns(self._base)).to_pylist()

    def _base_table(self) -> pa.Table | None:
        """The whole index of the starting version, without the files, loaded on first use."""
        if self._base_full is None and self._base is not None:
            self._base_full = self._base.to_table(columns=_columns(self._base))
        return self._base_full

    def _paths(self) -> Iterator[str]:
        """Paths of all items that currently exist."""
        yield from (p for p in self._base_index if p not in self._pending)
        yield from (p for p, row in self._pending.items() if row is not None)

    def _require(self, paths: Sequence[str]) -> None:
        missing = [p for p in paths if not self._exists(p)]
        if missing:
            raise DatasetError(f"items not in {self.ds.name}: {missing[:10]}")

    def _require_new(self, paths: Sequence[str], freed: set[str] = frozenset()) -> None:
        """Refuse paths given twice or that already exist (unless being moved away)."""
        seen: set[str] = set()
        for path in paths:
            if path in seen:
                raise DatasetError(f"{path!r} is already added in this edit")
            if self._exists(path) and path not in freed:
                if self._pending.get(path) is not None:
                    raise DatasetError(f"{path!r} is already added in this edit")
                raise DatasetError(
                    f"{path!r} already exists in {self.ds.name}; use update to change it"
                )
            seen.add(path)

    # --- operations ----------------------------------------------------------

    def add(self, path: str, src: str | Path, **fields: Any) -> None:
        """Add a new item. Its content is read when the edit is saved.

        Fails if the path already exists; change existing items with `update`.
        """
        self.add_many([(path, src)], **fields)

    def add_many(
        self,
        items: Iterable[tuple[str, str | Path | RemoteFile]],
        *,
        workers: int = DEFAULT_WORKERS,
        progress: Callable[[int], None] | None = None,
        **fields: Any,
    ) -> int:
        """Add several new items; their content is read with `workers` threads when the
        edit is saved.

        All paths and local files are checked first. `progress(n)` is called as files are
        read. Returns the number of items added.
        """
        self._use(_records.FILES)
        _check_fields(fields)
        pending = [
            (_validate_path(path), src if isinstance(src, RemoteFile) else Path(src))
            for path, src in items
        ]
        self._require_new([path for path, _ in pending])
        for _, src in pending:
            if isinstance(src, Path) and not src.is_file():
                raise FileNotFoundError(f"{src} is not a file")
        now = datetime.now(UTC)
        for path, src in pending:
            mime, _ = mimetypes.guess_type(path)
            self._stage(path, {
                "path": path, "hash": None, "size": None,
                "mime_type": mime or "application/octet-stream", "added_at": now,
                **fields, _ORIGIN: None, _CONTENT: src,
            })  # fmt: skip
        self._workers, self._progress = workers, progress
        self._set_columns.update(fields)
        return len(pending)

    def copy_from(
        self, src: Dataset, where: str | None = None, *, prefix: str = "", **fields: Any
    ) -> int:
        """Add the items of another dataset version that match `where` (SQL over its index),
        under `prefix`. Their files are copied when the edit is saved. Metadata is kept;
        `fields` override it.

        The source version is recorded as an input (§2.11). Returns the number of items.
        """
        self._use(_records.FILES)
        _check_fields(fields)
        if (src.project, src.cfg.storage_root) != (self.ds.project, self.ds.cfg.storage_root):
            raise DatasetError("items can only be copied between datasets of one project")
        rows = src.files(where=where).to_pylist()
        if not rows:
            raise DatasetError(f"no items in {src.key} match {where!r}")
        source = _Snapshot(src._table_uri, src.cfg, src.index().version)  # as just listed
        now = datetime.now(UTC)
        new = [
            {**row, "path": _validate_path(prefix + row["path"]), "added_at": now, **fields,
             _ORIGIN: None, _CONTENT: _Stored(source, row["path"])}
            for row in rows
        ]  # fmt: skip
        self._require_new([row["path"] for row in new])
        for row in new:
            self._stage(row["path"], row)
        self._set_columns.update(fields)
        return len(new)

    def add_dir(
        self,
        src: str | Path,
        prefix: str = "",
        *,
        include_hidden: bool = False,
        workers: int = DEFAULT_WORKERS,
        progress: Callable[[int], None] | None = None,
        **fields: Any,
    ) -> int:
        """Add every file under `src`, keyed by `prefix` + relative path. Returns the count.

        Files and directories whose names start with "." (e.g. `.DS_Store`, `.git/`) are
        skipped unless `include_hidden` is set.

        `src` may be an S3 address (`s3://bucket/prefix`, or one object): its objects are
        read into the dataset (data.md §5), and the address is recorded as an input of
        the version.
        """
        if is_bucket(src):
            items = bucket_files(src, prefix, self.ds.cfg, include_hidden=include_hidden)
            self.inputs.add(src.rstrip("/"))
            workers = max(workers, REMOTE_WORKERS)  # requests, not local disk reads
            return self.add_many(items, workers=workers, progress=progress, **fields)
        items = list(dir_items(src, prefix, include_hidden=include_hidden))
        return self.add_many(items, workers=workers, progress=progress, **fields)

    def update(self, path: str, src: str | Path | None = None, **fields: Any) -> None:
        """Change an existing item: replace its content with `src`, set user columns, or
        both. Columns not set are kept."""
        self._use(_records.FILES)
        path = _validate_path(path)
        _check_fields(fields)
        self._require([path])
        self._set_columns.update(fields)
        if src is None and path not in self._pending:
            self._patches.setdefault(path, {}).update(fields)  # no row is read for this
            return
        (row,) = self._rows([path])
        row.update(fields)
        if src is not None:
            if not Path(src).is_file():
                raise FileNotFoundError(f"{src} is not a file")
            row.update({"hash": None, "size": None, _ORIGIN: None, _CONTENT: Path(src)})
        self._stage(path, row)

    def move(self, old: str, new: str) -> int:
        """Rename an item, or every item under a directory (`old` and `new` both ending in
        "/"). Content, metadata, and `added_at` are kept. Returns the number of items moved.
        """
        self._use(_records.FILES)
        is_dir = old.endswith("/")
        if is_dir != new.endswith("/"):
            raise ValueError("old and new must both be item paths or both end with '/'")
        if is_dir:
            sources = sorted(p for p in self._paths() if p.startswith(old))
        else:
            sources = [old] if self._exists(old) else []
        if not sources:
            raise DatasetError(f"no items at {old!r} in {self.ds.name}")
        targets = [_validate_path(new + p[len(old) :] if is_dir else new) for p in sources]
        self._require_new(targets, freed=set(sources))
        rows = self._rows(sources)
        for source in sources:
            if source in self._base_index:
                self._stage(source, None)
            else:
                del self._pending[source]  # added in this edit: only its name changes
        for target, row in zip(targets, rows, strict=True):
            self._stage(target, {**row, "path": target})
        return len(sources)

    def remove(self, path: str) -> None:
        self._use(_records.FILES)
        path = _validate_path(path)
        self._require([path])
        if path in self._base_index:
            self._stage(path, None)
        else:
            del self._pending[path]  # added in this edit: nothing to remove from the base

    def set_info(self, **info: str | None) -> None:
        """Set dataset-level info (e.g. description, license, source). None removes a key."""
        self._check_open()
        for key, value in info.items():
            self._info[key] = None if value is None else str(value)

    def set_schema(self, schema: Any) -> None:
        """Set the dataset schema: a pandas `DataFrameSchema` or its YAML. None removes it.

        This and every later version must match the schema to be committed.
        """
        self._check_open()
        from fieldbook.data.schema import to_yaml

        self._schema = None if schema is None or schema == "" else to_yaml(schema)

    def drop_columns(self, *names: str) -> None:
        """Remove user columns from the dataset. Earlier versions keep them."""
        self._check_open()
        _check_fields(dict.fromkeys(names))
        self._drops.update(names)

    # --- records -------------------------------------------------------------

    def add_records(self, source: Any, *, delimiter: str | None = None) -> None:
        """Add rows to a records dataset: from a Parquet, JSON-lines, or delimited text
        file (`.csv`, `.tsv`; any extension once `delimiter` is stated), a folder or an
        s3:// address of such files, a pyarrow Table, a pandas DataFrame, or a list of
        dicts. Columns the dataset lacks are added; rows lacking a column get null. Read
        when the edit is saved; `rows_added` then says how many."""
        self._use(_records.RECORDS)
        if _records._in_memory(source) is None:
            _records._files(source, delimiter, self.ds.cfg)  # refuse what is no table now
        if is_bucket(source):
            self.inputs.add(source.rstrip("/"))
        self._row_ops.append(("add", source, delimiter))

    def remove_records(self, where: str) -> None:
        """Remove the rows matching a SQL filter, e.g. `"label IS NULL"`."""
        self._use(_records.RECORDS)
        self._row_ops.append(("remove", where))

    def update_records(self, where: str, **fields: Any) -> None:
        """Set columns to the given values in the rows matching a SQL filter. A column
        the dataset lacks is added (null in the other rows)."""
        self._use(_records.RECORDS)
        if not fields:
            raise ValueError("update_records needs at least one column=value")
        self._set_columns.update(fields)
        self._row_ops.append(("update", where, fields))

    def _sources(self) -> list[tuple[Any, str | None]]:
        return [(op[1], op[2]) for op in self._row_ops if op[0] == "add"]

    def _constants(self) -> dict[str, Any]:
        """Every column an `update_records` of this edit sets, with a value given for it."""
        return {k: v for op in self._row_ops if op[0] == "update" for k, v in op[2].items()}

    def _use(self, kind: str) -> None:
        """Refuse an operation of the other kind: a dataset holds files or records."""
        self._check_open()
        self._require_kind(kind)

    def _require_kind(self, kind: str) -> None:
        if self._kind is None:
            self._kind = kind
        elif self._kind != kind and kind == _records.RECORDS:
            raise DatasetError(_records.NOT_RECORDS.format(name=self.ds.name))
        elif self._kind != kind:
            raise DatasetError(_records.NOT_FILES.format(
                name=self.ds.name, instead="add rows with add_records, change them with "
                "update_records and remove_records"))  # fmt: skip

    def _changed(self) -> str:
        if self._kind == _records.RECORDS:
            return "rows changed"
        return f"{len(self.touched)} items changed"

    def _check_open(self) -> None:
        if self._closed:
            raise DatasetError("this edit is already committed; start a new edit")

    @property
    def touched(self) -> set[str]:
        return set(self._pending) | set(self._patches)

    def _has_changes(self) -> bool:
        return (
            bool(self._pending)
            or bool(self._patches)
            or bool(self._row_ops)
            or bool(self._info)
            or bool(self._drops)
            or self._schema is not _KEEP
        )

    # --- committing ----------------------------------------------------------

    def commit(self) -> int:
        """Commit the changes as a new version and return its number.

        If another writer committed first, the edit is rebased onto the latest version as
        long as that writer did not change any item this edit changes.
        """
        self._check_open()
        self._closed = True
        _lineage.close_edit(self)
        if not self._has_changes():
            self.version = self._base_version
            return self._base_version or 0
        if _smoke.sample_size() is not None and not _smoke.writes_to_scratch():
            if self._kind == _records.RECORDS:  # every check still runs
                _records.check(self, self._base)
            else:
                self._apply(self._base_table())
            _smoke.skip(f"new version of {self.ds.key} ({self._changed()})")
            self.version = self._base_version
            return self._base_version or 0
        base_version = self._base_version
        if _smoke.writes_to_scratch() and not self.ds.in_scratch and self._base is not None:
            # The first save of a smoke run or a rerun works on a copy of the dataset that
            # shares its files: nothing is written to the dataset itself (runs.md §9).
            clone = self._base.shallow_clone(
                self.ds._write_uri(), self._base.version,
                storage_options=storage.lance_options(self.ds.cfg),
            )  # fmt: skip
            base_version = clone.version
        # What this edit's changes were derived from: compared with the latest version when
        # another writer commits first. Only these are read, not the whole index.
        base = _Seen.of_rows(self._saved_rows(self.touched), self._base)
        for _ in range(MAX_COMMIT_ATTEMPTS):
            try:
                if base_version is None and self._kind == _records.RECORDS:
                    self.version = _records.create(self)
                elif base_version is None:
                    self._apply(None)
                    self.version = self._create()
                else:
                    index = _open(self.ds._write_uri(), self.ds.cfg, base_version)
                    self._require_kind(_records.kind_of(index.schema))  # created meanwhile
                    if self._kind == _records.RECORDS:
                        self.version = _records.save(self, index)
                        _committed(self.ds, self.version)
                        return self.version
                    if self._declares_schema(index.schema):
                        self._apply(index.to_table(columns=_columns(index)))
                    self.version = self._save(index)
                _committed(self.ds, self.version)
                return self.version
            except (lance_commit.CommitConflictError, OSError) as e:
                latest = _open_if_exists(self.ds._write_uri(), self.ds.cfg)
                # Creating a dataset that another writer created first surfaces as an
                # OSError rather than a commit conflict.
                created_concurrently = base_version is None and latest is not None
                if not isinstance(e, lance_commit.CommitConflictError) and not created_concurrently:
                    raise
                seen = _Seen.of_index(latest, self.touched)
                if self._drops or (self._schema is not _KEEP and base.schema != seen.schema):
                    raise ConflictError(
                        f"a concurrent edit to {self.ds.name} was committed while this edit "
                        "changes columns or the schema; retry the edit"
                    ) from None
                info = lambda s: {k: s.info.get(k) for k in self._info}  # noqa: E731
                if base.rows != seen.rows or info(base) != info(seen):
                    raise ConflictError(
                        f"a concurrent edit to {self.ds.name} changed items this edit changes"
                    ) from None
                base_version = latest.version if latest is not None else None
                base = seen
        raise DatasetError(f"could not commit to {self.ds.name}: too many concurrent edits")

    def _declares_schema(self, index_schema: pa.Schema) -> bool:
        from fieldbook.data import schema

        return self._schema is not _KEEP or schema.read_schema(index_schema) is not None

    def _apply(self, table: pa.Table | None) -> pa.Table:
        """The new version's index (without the files): `table` with the staged changes
        applied, checked against the dataset's schema. Rebasing guarantees the touched
        items of `table` match those the changes were derived from."""
        table = table if table is not None else BASE_SCHEMA.empty_table()
        set_and_dropped = self._drops & self._set_columns
        if set_and_dropped:
            raise DatasetError(f"columns both set and dropped: {sorted(set_and_dropped)}")
        rows = [_public(row) for row in self._pending.values() if row is not None]
        if self._patches:
            patched = table.filter(
                pc.is_in(table["path"], value_set=pa.array(sorted(self._patches), pa.string()))
            )
            rows += [{**row, **self._patches[row["path"]]} for row in patched.to_pylist()]
        touched = pa.array(sorted(self.touched), pa.string())
        kept = table.filter(pc.invert(pc.is_in(table["path"], value_set=touched)))
        parts = [kept]
        if rows:
            parts.append(_conform(_table_from_rows(rows), table.schema, self.ds.name))
        result = pa.concat_tables(parts, promote_options="permissive")
        unknown = self._drops - set(result.column_names)
        if unknown:
            raise DatasetError(f"columns not in {self.ds.name}: {sorted(unknown)}")
        result = result.drop_columns(sorted(self._drops))
        result = result.replace_schema_metadata(self._metadata(table.schema))
        from fieldbook.data import schema

        if (declared := schema.read_schema(result.schema)) is not None:
            schema.validate(result.sort_by("path"), declared, self.ds.name)
        return result

    def _metadata(self, current: pa.Schema) -> dict[bytes, bytes] | None:
        """The table's metadata after this edit: the dataset's info and schema."""
        from fieldbook.data import schema

        info = {**read_info(current), **self._info}
        info = {k: v for k, v in info.items() if v is not None}
        metadata = dict(current.metadata or {})
        metadata.pop(INFO_KEY, None)
        if info:
            metadata[INFO_KEY] = json.dumps(info, sort_keys=True).encode()
        if self._schema is not _KEEP:
            metadata.pop(schema.SCHEMA_KEY, None)
            if self._schema is not None:
                metadata[schema.SCHEMA_KEY] = self._schema.encode()
        return metadata or None

    def _record(self) -> dict[str, str]:
        return _properties(self.ds, self.message, sorted(self.inputs))

    # --- writing -------------------------------------------------------------

    def _new_rows(self) -> list[dict[str, Any]]:
        return [row for row in self._pending.values() if row is not None and row[_CONTENT]]

    def _create(self) -> int:
        """The dataset's first version: one write of every staged file."""
        rows = self._new_rows()
        fields = _table_from_rows([_public(row) for row in rows]).schema if rows else BASE_SCHEMA
        user = [f for f in fields if f.name not in BASE_COLUMNS and f.name not in self._drops]
        schema = pa.schema([*BASE_SCHEMA, *user, lance.blob_field(DATA)],
                           metadata=self._metadata(pa.schema([])))  # fmt: skip
        reader = pa.RecordBatchReader.from_batches(schema, self._batches(rows, schema))
        return _create_table(self.ds, schema, reader, self._record())

    def _save(self, index: lance.LanceDataset) -> int:
        """Commit the staged changes on `index`'s version (data.md §6). Raises Lance's
        CommitConflictError if another version was committed after it."""
        uri, options = self.ds._write_uri(), storage.lance_options(self.ds.cfg)
        staged = [row for row in self._pending.values() if row is not None]
        new_rows = [row for row in staged if row[_CONTENT]]
        # Saved rows that go on: their new values, by the path they are saved under.
        changes: dict[str, dict[str, Any]] = {p: dict(f) for p, f in self._patches.items()}
        for row in staged:
            if not row[_CONTENT]:
                changes[row[_ORIGIN]] = {k: v for k, v in _public(row).items()
                                         if k == "path" or k not in BASE_COLUMNS}  # fmt: skip
        removed = set(self._pending) - set(changes)  # saved rows nothing continues
        where = _find(index, set(changes) | removed)
        removed &= set(where)
        index = self._restructure(index, uri)
        existing = set(index.schema.names)
        named = {k for values in changes.values() for k in values}
        named |= {k for row in new_rows for k in _public(row)}
        added = sorted(named - existing)
        moved = any(v.get("path", p) != p for p, v in changes.items())
        changed = sorted(c for c in {k for v in changes.values() for k in v} & existing
                         if c != "path" or moved)  # fmt: skip
        if added:
            operation = self._add_columns(index, added, changes, new_rows)
            if not (changed or removed or new_rows):
                # values for a new column and nothing else (the usual first relabel): the
                # whole edit is this one commit
                return self._commit(index, operation, self._record()).version
            index = self._commit(index, operation, {INTERNAL_KEY: "1"})
        operation = self._change_rows(index, uri, options, changes, changed, removed, where,
                                      new_rows)  # fmt: skip
        return self._commit(index, operation, self._record()).version

    def _restructure(self, index: lance.LanceDataset, uri: str) -> lance.LanceDataset:
        """Drop columns and change the dataset's info and schema, as steps of the save,
        and return the table after them."""
        set_and_dropped = self._drops & self._set_columns
        if set_and_dropped:
            raise DatasetError(f"columns both set and dropped: {sorted(set_and_dropped)}")
        unknown = self._drops - set(index.schema.names)
        if unknown:
            raise DatasetError(f"columns not in {self.ds.name}: {sorted(unknown)}")
        metadata = self._metadata(index.schema)
        if self._drops or metadata != (index.schema.metadata or None):
            # Lance's own calls, which keep the columns' identities (an operation built
            # from an Arrow schema renumbers them, and reads then return other columns'
            # data). They cannot be told which version they follow, so that is checked.
            if _open(uri, self.ds.cfg).version != index.version:
                raise lance_commit.CommitConflictError(f"{self.ds.name} changed since it was read")
            if self._drops:
                index.drop_columns(sorted(self._drops))
            if metadata != (index.schema.metadata or None):
                index.update_schema_metadata(
                    {k.decode(): v.decode() for k, v in (metadata or {}).items()}, replace=True
                )
            index = _open(uri, self.ds.cfg)
        return index

    def _commit(self, index: lance.LanceDataset, operation: Any,
                record: dict[str, str]) -> lance.LanceDataset:  # fmt: skip
        """One step of a save, refused if anything was committed after `index`'s version."""
        transaction = lance_dataset.Transaction(
            read_version=index.version, operation=operation, transaction_properties=record
        )
        return lance.LanceDataset.commit(
            self.ds._write_uri(), transaction, max_retries=0,
            storage_options=storage.lance_options(self.ds.cfg),
        )  # fmt: skip

    def _add_columns(
        self,
        index: lance.LanceDataset,
        added: list[str],
        changes: dict[str, dict[str, Any]],
        new_rows: list[dict[str, Any]],
    ) -> Any:
        """New columns, with their values for the saved rows that were given one (null
        for the rest)."""
        given = {p: v for p, v in changes.items() if any(name in v for name in added)}
        samples = [*given.values(), *(_public(row) for row in new_rows)]
        columns: dict[str, pa.Array] = {"path": pa.array(list(given) or [""], pa.string())}
        for name in added:
            kind = pa.array([s[name] for s in samples if s.get(name) is not None]).type
            values = [v.get(name) for v in given.values()] or [None]
            columns[name] = _as_type(values, pa.string() if pa.types.is_null(kind) else kind,
                                     name, self.ds.name)  # fmt: skip
        values = pa.table(columns)  # a path that matches nothing still adds the columns
        fragments, schema = [], None
        for fragment in index.get_fragments():
            merged, schema = fragment.merge(values, left_on="path")  # Lance's own schema:
            fragments.append(merged)  # converting it to Arrow and back would renumber it
        if schema is None:  # no data files (every item was removed): Lance's own call
            index.add_columns(pa.schema([pa.field(n, columns[n].type) for n in added]))
            return lance.LanceOperation.Update()
        return lance.LanceOperation.Merge(fragments, schema)

    def _change_rows(
        self,
        index: lance.LanceDataset,
        uri: str,
        options: dict[str, str],
        changes: dict[str, dict[str, Any]],
        changed: list[str],
        removed: set[str],
        where: dict[str, tuple[int, int]],
        new_rows: list[dict[str, Any]],
    ) -> Any:
        """One operation: new values for existing columns of saved rows, rows marked
        deleted, and new rows appended."""
        fragments = {f.fragment_id: f for f in index.get_fragments()}
        updated: dict[int, Any] = {}
        modified: list[int] = []
        if changed:
            given = {p: v for p, v in changes.items() if any(c in v for c in changed)}
            current = _current(index, sorted(given), changed)
            columns: dict[str, pa.Array] = {
                "_rowid": pa.array([current[p]["_rowid"] for p in given], pa.uint64())
            }
            for name in changed:
                values = [v[name] if name in v else current[p][name] for p, v in given.items()]
                columns[name] = _as_type(values, index.schema.field(name).type, name, self.ds.name)
            values = pa.table(columns)
            for fragment_id in sorted({where[p][0] for p in given}):
                updated[fragment_id], modified = fragments[fragment_id].update_columns(
                    values, left_on="_rowid", right_on="_rowid"
                )
        gone: list[int] = []
        offsets: dict[int, list[int]] = {}
        for path in removed:
            offsets.setdefault(where[path][0], []).append(where[path][1])
        for fragment_id, rows in offsets.items():
            left = fragments[fragment_id].delete_rows(rows)
            if left is None:  # every row of this data file is removed
                gone.append(fragment_id)
                updated.pop(fragment_id, None)
            else:
                base = updated.get(fragment_id, left)
                updated[fragment_id] = replace(base, deletion_file=left.deletion_file)
        if fragments:  # named in every save: see the class docstring
            anchor = min(fragments)
            if anchor not in updated and anchor not in gone:
                updated[anchor] = fragments[anchor].metadata
        appended = self._append(index, uri, options, new_rows) if new_rows else []
        return lance.LanceOperation.Update(
            removed_fragment_ids=gone, updated_fragments=list(updated.values()),
            new_fragments=appended, fields_modified=list(modified),
        )  # fmt: skip

    def _append(self, index: lance.LanceDataset, uri: str, options: dict[str, str],
                rows: list[dict[str, Any]]) -> list:  # fmt: skip
        """Write the new files as data files of their own, not yet part of any version.
        Kept if the save has to be redone on a later version: files are written once."""
        names = tuple(index.schema.names)
        if self._appended is None or self._appended[0] != names:
            # the table's own columns, in its order
            schema = pa.schema(
                [f if f.name != DATA else lance.blob_field(DATA) for f in index.schema]
            )
            reader = pa.RecordBatchReader.from_batches(schema, self._batches(rows, schema))
            written = lance.fragment.write_fragments(
                reader,
                uri,
                schema=schema,
                mode="append",
                storage_options=options,
                external_blob_mode="ingest",  # a large file is given by its address
            )
            self._appended = (names, written)
        return self._appended[1]

    def _batches(self, rows: list[dict[str, Any]], schema: pa.Schema) -> Iterator[pa.RecordBatch]:
        """The new rows with their files' bytes, read a batch at a time so that a million
        files never sit in memory together."""
        names = [f.name for f in schema if f.name != DATA]

        def load(row: dict[str, Any]) -> dict[str, Any]:
            source, loaded = row[_CONTENT], _public(row)
            if isinstance(source, Path) and source.stat().st_size > STREAM_ABOVE:
                digest, size = hashlib.sha256(), 0
                with source.open("rb") as f:
                    while chunk := f.read(HASH_CHUNK):
                        digest.update(chunk)
                        size += len(chunk)
                # its address: Lance copies the file in, a piece at a time
                return {**loaded, "hash": digest.hexdigest(), "size": size,
                        DATA: source.resolve().as_uri()}  # fmt: skip
            content = _read(source)
            if isinstance(row[_CONTENT], _Stored):  # a copy: its hash and size are known
                return {**loaded, DATA: content}
            return {**loaded, "hash": hashlib.sha256(content).hexdigest(),
                    "size": len(content), DATA: content}  # fmt: skip

        def batch(loaded: list[dict[str, Any]]) -> pa.RecordBatch:
            columns = {
                name: _as_type([row.get(name) for row in loaded], schema.field(name).type,
                               name, self.ds.name)
                for name in names
            }  # fmt: skip
            columns[DATA] = lance.blob_array([row[DATA] for row in loaded])
            return pa.record_batch(columns, schema=schema)

        held, size = [], 0
        ordered = sorted(rows, key=lambda row: row["path"])
        with ThreadPoolExecutor(max_workers=max(1, self._workers)) as pool:
            for row in pool.map(load, ordered):
                held.append(row)
                size += 0 if isinstance(row[DATA], str) else len(row[DATA])
                if self._progress is not None:
                    self._progress(1)
                if len(held) >= BATCH_FILES or size >= BATCH_BYTES:
                    yield batch(held)
                    held, size = [], 0
        if held:
            yield batch(held)


def _create_table(ds: Dataset, schema: pa.Schema, reader: Any, record: dict[str, str]) -> int:
    """A dataset's first version, refused if the dataset exists by the time it is
    committed. Lance's own `write_dataset` is not used: when two processes create a
    dataset at the same instant, it commits the second as an overwrite of the first, and
    the first one's rows are gone (seen: 2 of 40 rounds of four writers)."""
    uri, options = ds._write_uri(), storage.lance_options(ds.cfg)
    written = lance.fragment.write_fragments(
        reader, uri, schema=schema, mode="create", data_storage_version=FILE_FORMAT,
        storage_options=options, external_blob_mode="ingest",
    )  # fmt: skip
    transaction = lance_dataset.Transaction(
        read_version=0, operation=lance.LanceOperation.Overwrite(schema, written),
        transaction_properties=record,
    )  # fmt: skip
    created = lance.LanceDataset.commit(uri, transaction, max_retries=0, storage_options=options)
    return created.version


class _Snapshot:
    """One version of a dataset's table, read by path: where a copy's bytes come from. It
    reads the table it was listed from, which for a smoke run is its scratch copy."""

    def __init__(self, uri: str, cfg: Settings, version: int) -> None:
        self.uri, self.cfg, self.version = uri, cfg, version
        self._index: lance.LanceDataset | None = None
        self._where: dict[str, int] = {}
        self._lock = threading.Lock()

    def read(self, path: str) -> bytes:
        with self._lock:
            if self._index is None:
                self._index = _open(self.uri, self.cfg, self.version)
                table = self._index.to_table(columns=["path"], with_row_address=True)
                self._where = dict(
                    zip(table["path"].to_pylist(), table["_rowaddr"].to_pylist(), strict=True)
                )
            index = self._index
        return index.read_blobs(DATA, addresses=[self._where[path]])[0][1]


@dataclass(frozen=True)
class _Stored:
    """A file already stored in a dataset version: the content of a copied item."""

    source: _Snapshot
    path: str


def _read(source: Path | RemoteFile | _Stored) -> bytes:
    if isinstance(source, _Stored):
        return source.source.read(source.path)
    if isinstance(source, RemoteFile):
        return source.fs.cat_file(source.key)
    return source.read_bytes()


def _is_version(index: lance.LanceDataset, version: int) -> bool:
    """Whether a Lance version is a version of the dataset: one fieldbook recorded, not a
    step of an edit (`INTERNAL_KEY`) or Lance's housekeeping (no record at all)."""
    key = (index.uri, version)
    if key not in _versions_known:
        props = index.read_transaction(version).transaction_properties or {}
        _versions_known[key] = "author" in props and not props.get(INTERNAL_KEY)
    return _versions_known[key]


_versions_known: dict[tuple[str, int], bool] = {}  # a version's record never changes


def _shown(index: lance.LanceDataset) -> int:
    """The dataset version an open table shows: its own number, or, for a table opened at
    Lance's housekeeping (same content, compacted), the version before it."""
    if _is_version(index, index.version):
        return index.version
    earlier = sorted((v["version"] for v in index.versions() if v["version"] < index.version),
                     reverse=True)  # fmt: skip
    return next((v for v in earlier if _is_version(index, v)), index.version)


def _public(row: dict[str, Any]) -> dict[str, Any]:
    """A staged row without the edit's own bookkeeping."""
    return {k: v for k, v in row.items() if k not in (_ORIGIN, _CONTENT)}


def _columns(index: lance.LanceDataset) -> list[str]:
    """Every column but the files' bytes."""
    return [name for name in index.schema.names if name != DATA]


def _as_type(values: list[Any], kind: pa.DataType, name: str, dataset_name: str) -> pa.Array:
    """Values as a column of the dataset's type. Values that convert cleanly (5 into a text
    column, 1 into a float column) are stored with that type; those that cannot (text into
    a number column, 1.5 into a whole-number column) raise a DatasetError."""
    try:
        given = pa.array(values)  # as given, then converted: asking for the type directly
        # would cut 1.5 down to 1 without a word
        if pa.types.is_null(given.type):
            return pa.nulls(len(values), kind)
        return given if given.type == kind else pc.cast(given, kind)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        try:
            found = pa.array([v for v in values if v is not None]).type
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            found = "mixed"
        raise DatasetError(
            f"column {name!r} in {dataset_name} has type {kind}; cannot store {found} values in it"
        ) from None


def _find(index: lance.LanceDataset, paths: set[str]) -> dict[str, tuple[int, int]]:
    """(data file, row within it) of each of `paths` that the version holds."""
    if not paths:
        return {}
    filter_ = _path_in(sorted(paths)) if len(paths) <= REWRITE_ABOVE_REMOVALS else None
    table = index.to_table(columns=["path"], filter=filter_, with_row_address=True)
    found = zip(table["path"].to_pylist(), table["_rowaddr"].to_pylist(), strict=True)
    return {path: (address >> 32, address & 0xFFFFFFFF) for path, address in found if path in paths}


def _current(index: lance.LanceDataset, paths: list[str], columns: list[str]) -> dict[str, dict]:
    """The saved values of `columns`, and the row ID, for each of `paths`."""
    filter_ = _path_in(paths) if len(paths) <= REWRITE_ABOVE_REMOVALS else None
    wanted = sorted({"path", *columns})
    table = index.to_table(columns=wanted, filter=filter_, with_row_id=True)
    keep = set(paths)
    return {row["path"]: row for row in table.to_pylist() if row["path"] in keep}


def _open_if_exists(uri: str, cfg: Settings) -> lance.LanceDataset | None:
    try:
        return _open(uri, cfg)
    except (FileNotFoundError, ValueError, OSError):
        return None


@dataclass(frozen=True)
class _Seen:
    """What a concurrent save must not have changed for an edit to be redone on top of it:
    the edit's touched items, the dataset's info, and its schema."""

    rows: list[dict[str, Any]]
    info: dict[str, str]
    schema: str | None

    @classmethod
    def of_rows(cls, rows: list[dict[str, Any]], index: lance.LanceDataset | None) -> _Seen:
        from fieldbook.data.schema import read_schema

        if index is None:
            return cls([], {}, None)
        rows = sorted(rows, key=lambda r: r["path"])
        return cls(rows, read_info(index.schema), read_schema(index.schema))

    @classmethod
    def of_index(cls, index: lance.LanceDataset | None, touched: set[str]) -> _Seen:
        """Read from the index: only the touched items, unless they are so many that naming
        them in a filter costs more than reading everything (C3's measurement)."""
        from fieldbook.data.schema import read_schema

        if index is None:
            return cls([], {}, None)
        if not touched:
            rows = []
        elif len(touched) > REWRITE_ABOVE_REMOVALS:
            rows = _rows(index.to_table(columns=_columns(index)), touched)
        else:
            rows = _rows(
                index.to_table(columns=_columns(index), filter=_path_in(sorted(touched))), touched
            )
        return cls(rows, read_info(index.schema), read_schema(index.schema))


def _latest(ds: Dataset) -> lance.LanceDataset | None:
    try:
        return _open(ds._table_uri, ds.cfg)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _properties(ds: Dataset, message: str, inputs: Sequence[str]) -> dict[str, str]:
    """A version's commit record (§2.11), stored as Lance transaction properties."""
    run = context.current()
    properties = {
        COMMIT_MESSAGE_KEY: message,
        "author": ds.cfg.author or getpass.getuser(),
        "agent": _agents.current(),
        "run_id": run.run_id,
        "command": _lineage.current_command(),
        "code": run.code,
        "git": run.git,
        "env": run.env,
    }
    if inputs:
        properties["inputs"] = json.dumps(list(inputs))
    return properties


def _committed(ds: Dataset, version: int) -> None:
    if _smoke.writes_to_scratch():
        _smoke.note(f"saved {_lineage.version_ref(ds, version)} (scratch, not the dataset)")
    _lineage.record_output(ds, version)


def _path_in(paths: Sequence[str]) -> str:
    """A filter matching these paths (SQL string literals, quotes doubled)."""
    quoted = ",".join("'" + path.replace("'", "''") + "'" for path in paths)
    return f"path IN ({quoted})"


def _tags_by_version(ds: lance.LanceDataset) -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for tag, info in ds.tags.list().items():
        found.setdefault(info["version"], []).append(tag)
    return found


def _read_version_info(
    ds: lance.LanceDataset, entry: dict[str, Any], tags_by_version: dict[int, list[str]]
) -> VersionInfo | None:
    """A version's record; None for a commit that is not a version of the dataset: a step
    of an edit (`INTERNAL_KEY`), or one fieldbook did not write (Lance's compaction)."""
    props = ds.read_transaction(entry["version"]).transaction_properties or {}
    if props.get(INTERNAL_KEY) or "author" not in props:
        return None
    return VersionInfo(
        version=entry["version"],
        timestamp=entry["timestamp"],
        message=props.get(COMMIT_MESSAGE_KEY, ""),
        author=props.get("author", ""),
        agent=props.get("agent", ""),
        run_id=props.get("run_id", ""),
        rows=int(entry["metadata"].get("total_rows", 0)),
        tags=tuple(sorted(tags_by_version.get(entry["version"], []))),
        inputs=tuple(json.loads(props["inputs"])) if props.get("inputs") else (),
        command=props.get("command", ""),
        code=props.get("code", ""),
        git=props.get("git", ""),
        env=props.get("env", ""),
    )


def _input_ref(source: Dataset | str, editing: Dataset) -> str:
    """Normalize an explicit input: datasets and `[project/]name[@version|tag]` strings
    resolve to `project/name@version`; strings containing `://` are kept as given. Datasets
    of other projects are refused (sharing.md §4)."""
    if isinstance(source, str):
        if "://" in source:
            return source
        target, _, ref = source.partition("@")
        project, _, name = target.rpartition("/")
        if project and project != editing.project:
            raise DatasetError(f"input {source!r} is in another project; projects do not "
                               "read each other's datasets")  # fmt: skip
        source = Dataset(name, (int(ref) if ref.isdigit() else ref) or None, cfg=editing.cfg)
    if not isinstance(source, Dataset):
        raise TypeError(f"inputs must be datasets or strings, not {type(source).__name__}")
    if not source.exists:
        raise DatasetNotFoundError(f"input dataset {source.key} not found")
    return _lineage.version_ref(source, source.version)


def dir_items(
    src: str | Path, prefix: str = "", *, include_hidden: bool = False
) -> Iterator[tuple[str, Path]]:
    """(item path, file) pairs for every file under `src`, sorted by path."""
    src = Path(src)
    for file in sorted(p for p in src.rglob("*") if p.is_file()):
        relative = file.relative_to(src)
        if include_hidden or not any(part.startswith(".") for part in relative.parts):
            yield prefix + relative.as_posix(), file


def read_info(schema: pa.Schema) -> dict[str, str]:
    raw = (schema.metadata or {}).get(INFO_KEY)
    return json.loads(raw) if raw else {}


def _is_not_found(error: Exception) -> bool:
    """Lance reports a missing dataset as a ValueError saying "not found"; other failures
    (access denied, network) use other words."""
    return "not found" in str(error).lower()


def _open(uri: str, cfg: Settings, version: int | str | None = None) -> lance.LanceDataset:
    """A dataset index, with the storage root's location settings (endpoint, region)."""
    return lance.dataset(uri, version=version, storage_options=storage.lance_options(cfg))


def _pylist(column: Any) -> list:
    return column.to_pylist() if hasattr(column, "to_pylist") else list(column)


def _rows(table: pa.Table | None, paths: set[str]) -> list[dict[str, Any]]:
    if table is None:
        return []
    selected = table.filter(pc.is_in(table["path"], value_set=pa.array(sorted(paths), pa.string())))
    return sorted(selected.to_pylist(), key=lambda r: r["path"])


def validate_tag(tag: str) -> str:
    if not _NAME_RE.match(tag) or tag.isdigit():
        raise InvalidNameError(
            f"invalid tag: {tag!r} (letters, digits, '.', '_', '-'; not all digits)"
        )
    return tag


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a table from rows that may have different keys; missing values become null.

    `pa.Table.from_pylist` infers columns from the first row only, which would drop
    fields that appear only in later rows.
    """
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows])


def _conform(table: pa.Table, schema: pa.Schema, dataset_name: str) -> pa.Table:
    """Cast columns of new rows to the dataset's existing column types.

    Values that convert cleanly (e.g. 5 into a text column, 1 into a float column) are
    stored with the existing type; values that cannot convert raise a DatasetError.
    """
    for name in table.column_names:
        if name not in schema.names:
            continue
        target = schema.field(name).type
        column = table[name]
        if column.type == target or pa.types.is_null(target) or pa.types.is_null(column.type):
            continue
        try:
            table = table.set_column(table.column_names.index(name), name, pc.cast(column, target))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            raise DatasetError(
                f"column {name!r} in {dataset_name} has type {target}; "
                f"cannot store {column.type} values in it"
            ) from None
    return table


def _first_line(error: Exception) -> str:
    return str(error).splitlines()[0].split(", /")[0]


def _dir_prefix(path: str, depth: int) -> str:
    """The first `depth` directory levels of a path, or "." for a file above that level."""
    parts = path.split("/")[:-1]
    return "/".join(parts[:depth]) if parts else "."


def _validate_path(path: str) -> str:
    p = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or p.is_absolute()
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(f"invalid item path: {path!r}")
    return path


def _check_fields(fields: dict[str, Any]) -> None:
    reserved = set(fields) & set(RESERVED_COLUMNS)
    if reserved:
        raise ValueError(f"reserved columns cannot be set: {sorted(reserved)}")
