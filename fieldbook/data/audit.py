"""Comparing versions and checking stored files (data.md §8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pyarrow as pa

from fieldbook.config import Settings, settings
from fieldbook.data.dataset import DATA, Dataset, list_datasets
from fieldbook.data.records import RECORDS, kind_of

# Columns that do not count as a change: content is compared by hash, and added_at
# records when the path first appeared.
_IGNORED_IN_DIFF = {"path", "hash", "size", "added_at"}


def diff(old: Dataset, new: Dataset, prefix: str = "") -> pa.Table:
    """Items added, removed, or modified between two views of a dataset.

    Returns columns `path`, `change` (added, removed, modified), and `changed`: for
    modified items, a comma-separated list of what changed ("content" and/or column
    names).
    """
    before = {r["path"]: r for r in old.files(prefix).to_pylist()}
    after = {r["path"]: r for r in new.files(prefix).to_pylist()}
    rows = []
    for path in sorted(before.keys() | after.keys()):
        if path not in after:
            rows.append({"path": path, "change": "removed", "changed": ""})
        elif path not in before:
            rows.append({"path": path, "change": "added", "changed": ""})
        else:
            b, a = before[path], after[path]
            changed = ["content"] if b["hash"] != a["hash"] else []
            columns = (b.keys() | a.keys()) - _IGNORED_IN_DIFF
            changed += [c for c in sorted(columns) if b.get(c) != a.get(c)]
            if changed:
                rows.append({"path": path, "change": "modified", "changed": ", ".join(changed)})
    schema = pa.schema([("path", pa.string()), ("change", pa.string()), ("changed", pa.string())])
    return pa.Table.from_pylist(rows, schema=schema)


CHANGE = "_change"  # in a diff of records: "removed" or "added"


def diff_records(old: Dataset, new: Dataset) -> pa.Table:
    """Rows that differ between two views of a records dataset, as `diff` shows lines:
    those only in `old` ("removed") and those only in `new` ("added"), in a `_change`
    column before the rows' own. A changed row is one of each. Rows have no key and no
    order, so whole rows are compared, and as many times as each occurs; a column only
    one version has is null in the other. Sorted so that a changed row's two lines meet.
    """
    from fieldbook.data import query

    con = query.connect()
    query.register(con, old, "old_rows")
    query.register(con, new, "new_rows")
    names = list(dict.fromkeys([*new.index().schema.names, *old.index().schema.names]))
    columns = ", ".join('"{}"'.format(name.replace('"', '""')) for name in names)
    sides = {
        side: f"SELECT {columns} FROM (SELECT * FROM {side}_rows UNION ALL BY NAME "
        f"SELECT * FROM {other}_rows WHERE false)"
        for side, other in (("old", "new"), ("new", "old"))
    }
    return con.sql(
        f"SELECT 'removed' AS {CHANGE}, * FROM ({sides['old']} EXCEPT ALL {sides['new']}) "
        f"UNION ALL "
        f"SELECT 'added' AS {CHANGE}, * FROM ({sides['new']} EXCEPT ALL {sides['old']}) "
        f"ORDER BY {columns}, {CHANGE} DESC"
    ).to_arrow_table()


@dataclass
class VerifyReport:
    datasets: list[str] = field(default_factory=list)  # "project/name"
    objects: int = 0  # distinct stored files checked
    missing: list[str] = field(default_factory=list)  # "project/name@vN:path"
    corrupt: list[str] = field(default_factory=list)  # wrong size, or wrong hash if deep
    deep: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing and not self.corrupt


VERIFY_BATCH = 256  # files read together


def verify(
    ds: Dataset | None = None, *, deep: bool = False, cfg: Settings | None = None
) -> VerifyReport:
    """Check that every file of every version is stored with the recorded size.

    `deep` also reads every file and re-hashes its content. With no dataset, checks every
    dataset. A file that several versions share is checked once, and each problem is
    reported at the newest version and path that has it.
    """
    cfg = ds.cfg if ds is not None else (cfg or settings())
    if ds is not None:
        datasets = [ds]
    else:
        datasets = [Dataset(name, cfg=cfg) for name in list_datasets(cfg=cfg)]
    report = VerifyReport(datasets=[d.key for d in datasets], deep=deep)
    for d in datasets:
        checked: set[int] = set()  # row addresses: versions share the rows they keep
        for info in d.log():
            index = d._lance(info.version)
            if kind_of(index.schema) == RECORDS:
                try:  # rows have no content apart from the table: check it reads
                    if deep:
                        for _ in index.to_batches():
                            pass
                    else:
                        index.count_rows()
                except (OSError, ValueError, pa.ArrowException):
                    report.missing.append(f"{d.key}@v{info.version}: the version cannot be read")
                continue
            try:
                table = index.to_table(columns=["path", "hash", "size"], with_row_address=True)
            except (OSError, ValueError, pa.ArrowException):  # its data files are gone
                report.missing.append(f"{d.key}@v{info.version}: the version cannot be read")
                continue
            rows = [row for row in table.to_pylist() if row["_rowaddr"] not in checked]
            checked.update(row["_rowaddr"] for row in rows)
            report.objects += len(rows)
            for start in range(0, len(rows), VERIFY_BATCH):
                _check(index, rows[start : start + VERIFY_BATCH], f"{d.key}@v{info.version}",
                       deep, report)  # fmt: skip
    return report


def _check(index, rows: list[dict], where: str, deep: bool, report: VerifyReport) -> None:  # noqa: ANN001
    addresses = [row["_rowaddr"] for row in rows]
    try:
        if deep:
            found = dict(index.read_blobs(DATA, addresses=addresses))
            stored = [found.get(address) for address in addresses]
        else:
            stored = index.take_blobs(DATA, addresses=addresses)
    except (OSError, ValueError, pa.ArrowException):  # a data file is gone or unreadable
        report.missing += [f"{where}:{row['path']}" for row in rows]
        return
    for row, content in zip(rows, stored, strict=True):
        name = f"{where}:{row['path']}"
        if content is None:
            report.missing.append(name)
        elif deep:
            if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != row["hash"]:
                report.corrupt.append(name)
        elif content.size() != row["size"]:
            report.corrupt.append(name)
