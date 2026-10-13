"""Deleting versions and datasets, and freeing storage (data.md §7).

A dataset's files live in its table, and versions share the data they do not change, so
space is freed by Lance: when a version is deleted, the data files only it used go with
it. Removing an item only marks its row; its bytes are freed once every version that has
the item is gone, and, for data files that still hold other rows, once the table is
compacted. Data written in the last week by an edit that has not committed (one still
running, or one that failed) is left alone unless asked for (`grace` of zero).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.compute as pc

from fieldbook import storage
from fieldbook.config import Settings, settings
from fieldbook.data import smoke
from fieldbook.data._lazy import lance, lance_commit
from fieldbook.data.dataset import (
    BATCH_BYTES,
    BATCH_FILES,
    DATA,
    DATASETS_DIR,
    STREAM_ABOVE,
    Dataset,
    DatasetError,
    DatasetNotFoundError,
    list_datasets,
)
from fieldbook.data.records import RECORDS, kind_of

GRACE = timedelta(hours=24)
KEEP_LATEST = 10
RETENTION = timedelta(days=7)
# A table is compacted (rewritten without its removed rows, small data files merged) when
# this share of its rows is marked removed, or it has this many data files.
COMPACT_REMOVED = 0.2
COMPACT_FILES = 100
SMALL = 2000  # rows: a data file with fewer is merged with others when there are many


@dataclass
class CleanupReport:
    versions: dict[str, list[int]] = field(default_factory=dict)  # "project/name" -> versions
    datasets: list[str] = field(default_factory=list)  # "project/name"
    freed: int = 0  # bytes; known only by doing it, so 0 in a dry run
    compacted: list[str] = field(default_factory=list)  # "project/name"
    scratch: list[str] = field(default_factory=list)  # smoke-run scratch areas deleted
    code_snapshots: list[str] = field(default_factory=list)  # unused snapshot hashes
    package_lists: list[str] = field(default_factory=list)  # unused package-list hashes
    kept_for_runs: dict[str, list[int]] = field(default_factory=dict)  # retained: used by runs
    dry_run: bool = False


def delete_versions(
    ds: Dataset,
    versions: Sequence[int],
    *,
    grace: timedelta = GRACE,
    referenced: Mapping[str, set[int]] | None = None,
    dry_run: bool = False,
) -> CleanupReport:
    """Delete specific versions of a dataset.

    The latest version, tagged versions, and versions used by runs cannot be deleted.
    `referenced` defaults to the run records in MLflow (`records.referenced_versions`).
    """
    dry_run = dry_run or smoke.skip("deleting versions")
    _require_unpinned(ds)
    referenced = _run_references(ds.cfg, referenced)
    targets = sorted(set(versions))
    existing = {info.version for info in ds.log()}
    missing = [v for v in targets if v not in existing]
    if missing:
        raise DatasetError(f"versions not found in {ds.name}: {missing}")
    if ds.version in targets:
        raise DatasetError(
            f"cannot delete the latest version (v{ds.version}) of {ds.name}; "
            "make a new edit instead"
        )
    tagged = {tag: v for tag, v in ds.tags().items() if v in targets}
    if tagged:
        listed = ", ".join(f"v{v} ({tag})" for tag, v in sorted(tagged.items()))
        raise DatasetError(f"tagged versions cannot be deleted: {listed}; remove the tags first")
    used = sorted(set(targets) & set(referenced.get(ds.key, {})))
    if used:
        raise DatasetError(
            f"versions used by runs cannot be deleted: {_describe_runs(referenced, ds.key, used)}; "
            "delete those runs in MLflow first"
        )
    report = CleanupReport(versions={ds.key: targets}, dry_run=dry_run)
    if not dry_run:
        report.freed = _remove(ds, set(targets), grace)
    return report


def delete_dataset(
    ds: Dataset,
    *,
    grace: timedelta = GRACE,
    referenced: Mapping[str, Iterable[int]] | None = None,
    dry_run: bool = False,
) -> CleanupReport:
    """Delete a dataset with all its versions and tags, unless runs used any version."""
    dry_run = dry_run or smoke.skip("deleting the dataset")
    _require_unpinned(ds)
    if not ds.exists:
        raise DatasetNotFoundError(f"dataset {ds.key} not found")
    referenced = _run_references(ds.cfg, referenced)
    used = sorted(referenced.get(ds.key, {}))
    if used:
        raise DatasetError(
            f"{ds.key} cannot be deleted; runs used {_describe_runs(referenced, ds.key, used)}. "
            "Delete those runs in MLflow first"
        )
    report = CleanupReport(datasets=[ds.key], dry_run=dry_run)
    if not dry_run:
        report.freed = _size(ds)
        ds._remove_table()
    return report


def prune(
    ds: Dataset | None = None,
    *,
    keep_latest: int = KEEP_LATEST,
    retention: timedelta = RETENTION,
    grace: timedelta = GRACE,
    referenced: Mapping[str, set[int]] | None = None,
    dry_run: bool = False,
    cfg: Settings | None = None,
) -> CleanupReport:
    """Delete versions not kept by the retention rules, and compact tables that have many
    removed rows or many small data files.

    A version is kept if it is tagged, among the latest `keep_latest`, newer than
    `retention`, or used by a run (`referenced`, default: the run records in MLflow).
    With no dataset, prunes every dataset and also removes stale scratch areas and code
    snapshots and package lists nothing refers to.
    """
    dry_run = dry_run or smoke.skip("pruning")
    if ds is not None:
        _require_unpinned(ds)
        cfg = ds.cfg
        datasets = [ds]
    else:
        cfg = cfg or settings()
        datasets = list(_all_datasets(cfg))
    referenced = _run_references(cfg, referenced)
    report = CleanupReport(dry_run=dry_run)
    cutoff = datetime.now(UTC) - retention
    expired: dict[str, set[int]] = {}
    for d in datasets:
        stale = _expired(d, keep_latest, cutoff, referenced, report)
        if stale:
            expired[d.key] = set(stale)
            report.versions[d.key] = stale
        if dry_run:
            continue
        report.freed += _remove(d, set(stale), grace)
        # A compacted table is a version of its own (same items): earlier versions keep
        # the old data files, and then expire by the same rules.
        if _worth_compacting(d) and _compact(d):
            report.compacted.append(d.key)
            again = _expired(d, keep_latest, cutoff, referenced, report)
            if again:
                expired.setdefault(d.key, set()).update(again)
                report.versions[d.key] = sorted(expired[d.key])
            report.freed += _remove(d, set(again), grace)
    if ds is None:
        fs = storage.fs(cfg)
        report.scratch = _stale_scratch(fs, time.time() - grace.total_seconds())
        if not dry_run:
            for path in report.scratch:
                fs.rm(path, recursive=True)
        _free_environment(report, datasets, expired, max(grace, retention), cfg)
    return report


def _expired(
    ds: Dataset,
    keep_latest: int,
    cutoff: datetime,
    referenced: Mapping[str, Iterable[int]],
    report: CleanupReport,
) -> list[int]:
    """The versions the retention rules do not keep; those kept only because a run used
    them are noted in the report."""
    history = {info.version: info.timestamp for info in ds.log()}
    ordered = sorted(history)
    kept = (
        set(ordered[-keep_latest:])
        | set(ds.tags().values())
        | {v for v, ts in history.items() if _aware(ts) >= cutoff}
    )
    for_runs = sorted(set(referenced.get(ds.key, {})) & (set(ordered) - kept))
    if for_runs:
        report.kept_for_runs[ds.key] = for_runs
    return [v for v in ordered if v not in kept and v not in for_runs]


def _remove(ds: Dataset, versions: set[int], grace: timedelta) -> int:
    """Delete `versions`, and with them the Lance versions that are not versions of the
    dataset (steps of edits, housekeeping) and are no longer the latest. Returns the bytes
    freed: the data files only those versions used."""
    index = ds.index()
    shown = {info.version for info in ds.log()}
    internal = {v["version"] for v in index.versions()} - shown - {index.version}
    gone = sorted(versions | internal)
    if not gone:
        return 0
    before = _size(ds)
    index.cleanup_old_versions(
        older_than=timedelta(0), versions=gone, delete_unverified=grace <= timedelta(0)
    )
    return max(before - _size(ds), 0)


def _size(ds: Dataset) -> int:
    path = f"{DATASETS_DIR}/{ds.name}.lance"
    files = storage.fresh(ds._files)  # Lance has just deleted files that s3fs remembers
    return int(files.du(path)) if files.exists(path) else 0


def _worth_compacting(ds: Dataset) -> bool:
    fragments = ds.index().get_fragments()
    rows = sum(f.physical_rows for f in fragments)
    removed = sum(f.num_deletions for f in fragments)
    return len(fragments) >= COMPACT_FILES or (rows > 0 and removed / rows >= COMPACT_REMOVED)


COMPACTED = "compacted: removed items' space freed, data files merged"


def _compact(ds: Dataset) -> bool:
    """Rewrite the table's data files that hold removed rows, or are small among many,
    without the removed rows. False if there was nothing to rewrite, or a save came in
    meanwhile (the next cleanup does it).

    For files, the kept rows are read in batches and written as new data files, which
    one commit swaps for the old. Lance's own compaction is not used for them: it fetches
    a blob column one file a request (670 files of 4 KB in a bucket: 46 to 51 s, against
    1.4 s read in one batch). It is used for records, which have no such column, and for
    data files holding a file too large to read into memory.
    """
    index = ds.index()
    if kind_of(index.schema) == RECORDS:
        index.optimize.compact_files()
        ds._record_version(COMPACTED)
        return True
    fragments = index.get_fragments()
    many = len(fragments) >= COMPACT_FILES
    chosen = [f for f in fragments if f.num_deletions or (many and f.physical_rows < SMALL)]
    if not chosen:
        return False
    rows = index.scanner(
        columns=[name for name in index.schema.names if name != DATA],
        fragments=chosen, with_row_address=True,
    ).to_table()  # fmt: skip
    large = pc.filter(rows["_rowaddr"], pc.greater(rows["size"], STREAM_ABOVE))
    theirs = {address >> 32 for address in large.to_pylist()}
    ours = [f.fragment_id for f in chosen if f.fragment_id not in theirs]
    if ours:
        keep = pc.is_in(pc.shift_right(rows["_rowaddr"], pa.scalar(32, pa.uint64())),
                        value_set=pa.array(ours, pa.uint64()))  # fmt: skip
        schema = pa.schema([f if f.name != DATA else lance.blob_field(DATA) for f in index.schema])
        reader = pa.RecordBatchReader.from_batches(schema, _kept(index, rows.filter(keep), schema))
        written = lance.fragment.write_fragments(
            reader, ds._write_uri(), schema=schema, mode="append",
            storage_options=storage.lance_options(ds.cfg),
        )  # fmt: skip
        try:
            ds._record_version(COMPACTED, index, replaced=ours, by=written)
        except lance_commit.CommitConflictError:
            return False
    if theirs:
        ds.index().optimize.compact_files()
        ds._record_version(COMPACTED)
    return True


def _kept(index, rows: pa.Table, schema: pa.Schema) -> Iterable[pa.RecordBatch]:  # noqa: ANN001
    """The rows with their files, fetched a batch at a time."""
    sizes = rows["size"].to_pylist()
    start = 0
    while start < rows.num_rows:
        end, total = start, 0
        while end < rows.num_rows and end - start < BATCH_FILES:
            if end > start and total + sizes[end] > BATCH_BYTES:
                break
            total += sizes[end]
            end += 1
        part = rows.slice(start, end - start)
        addresses = part["_rowaddr"].to_pylist()
        found = dict(index.read_blobs(DATA, addresses=addresses))
        columns = [
            lance.blob_array([found[a] for a in addresses]) if field.name == DATA
            else part[field.name].combine_chunks()
            for field in schema
        ]  # fmt: skip
        yield pa.record_batch(columns, schema=schema)
        start = end


def _all_datasets(cfg: Settings) -> Iterable[Dataset]:
    """The project's datasets (storage.md §1)."""
    for name in list_datasets(cfg=cfg):
        yield Dataset(name, cfg=cfg)


def _run_references(
    cfg: Settings, referenced: Mapping[str, Iterable[int]] | None
) -> Mapping[str, Iterable[int]]:
    if referenced is not None:
        return referenced
    from fieldbook.tracking.records import referenced_versions

    return referenced_versions(cfg)


def _describe_runs(referenced: Mapping[str, Iterable[int]], key: str, versions: list[int]) -> str:
    """e.g. "v3 (run 20261005-...)" when run IDs are known, else "v3"."""
    by_version = referenced.get(key, {})
    parts = []
    for v in versions:
        runs = by_version.get(v, []) if isinstance(by_version, Mapping) else []
        parts.append(f"v{v} (run {', '.join(sorted(runs))})" if runs else f"v{v}")
    return ", ".join(parts)


def _free_environment(
    report: CleanupReport,
    datasets: list[Dataset],
    expired: Mapping[str, set[int]],
    keep_for: timedelta,
    cfg: Settings,
) -> None:
    """Delete code snapshots and package lists that no remaining dataset version and no
    active MLflow run refers to (runs.md §4-6). They are kept for `keep_for`
    after their last use, since a long run refers to its snapshot only once it saves a
    version or logs to MLflow."""
    from fieldbook.code.snapshot import CODE_DIR, PACKAGES_DIR
    from fieldbook.tracking.records import referenced_environment

    code, packages = referenced_environment(cfg)
    for d in datasets:
        for info in d.log():
            if info.version in expired.get(d.key, set()):
                continue
            code.add(info.code)
            if info.env:
                packages.add(json.loads(info.env).get("packages"))
    fs = storage.fs(cfg)
    cutoff = time.time() - keep_for.total_seconds()
    for folder, used, found in (
        (CODE_DIR, code, report.code_snapshots),
        (PACKAGES_DIR, packages, report.package_lists),
    ):
        if not fs.exists(folder):
            continue
        for path, info in fs.find(folder, detail=True).items():
            digest = path.rsplit("/", 1)[-1].split(".", 1)[0]
            if digest not in used and storage.modified_time(info) < cutoff:
                found.append(digest)
                if not report.dry_run:
                    fs.rm(path)
    report.code_snapshots.sort()
    report.package_lists.sort()


def _stale_scratch(fs, older_than: float) -> list[str]:
    """Smoke-run scratch areas (kept, or left by a crash) last changed before `older_than`."""
    if not fs.exists(smoke.SCRATCH_DIR):
        return []
    stale = []
    for path in fs.ls(smoke.SCRATCH_DIR, detail=False):
        files = fs.find(path, detail=True)
        newest = max((storage.modified_time(info) for info in files.values()), default=0.0)
        if newest < older_than:
            stale.append(path)
    return sorted(stale)


def _require_unpinned(ds: Dataset) -> None:
    ds._require_writable()
    if ds.ref is not None:
        raise DatasetError("open the dataset without a version to delete or prune")


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.astimezone(UTC)
