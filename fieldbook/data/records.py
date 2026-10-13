"""Records datasets: tabular data stored as the rows of the dataset's table (data.md §1).

A dataset holds files or records, decided by what is first added to it and kept from
then on. This module reads tabular sources (Parquet, JSON lines, delimited text) into
rows, and saves an edit's row changes as one Lance commit: new rows appended, rows marked
deleted, and changed columns written anew for the rows that change.
"""

from __future__ import annotations

import contextlib
import csv
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO

import pyarrow as pa
import pyarrow.compute as pc

from fieldbook import storage
from fieldbook.data._lazy import lance, lance_commit
from fieldbook.data.remote import RemoteFile, is_bucket
from fieldbook.data.remote import files as bucket_files

if TYPE_CHECKING:
    from fieldbook.data.dataset import Edit

KIND_KEY = b"fieldbook.kind"  # in the table's metadata; set on a records dataset
FILES, RECORDS = "files", "records"
PARQUET = {".parquet"}
JSON_LINES = {".json", ".jsonl", ".ndjson"}
DELIMITED = {".csv", ".tsv"}
DELIMITERS = ",\t;|"  # those looked for when none is stated
SNIFF_BYTES = 64 * 1024
NOT_FILES = "{name} holds records (the rows of a table), not files: {instead}"
READ_ROWS = "read its rows with to_table(), `fieldbook data ls`, or `fieldbook data query`"
NOT_RECORDS = (
    "{name} holds files, not records: rows are imported into a dataset of their own "
    "(a table added here is stored as a file)"
)


def kind_of(schema: pa.Schema) -> str:
    """What a table holds. A table without the mark (written by something else, or by an
    earlier fieldbook) is told by its columns: files have a `data` column, records none."""
    marked = (schema.metadata or {}).get(KIND_KEY)
    if marked is not None:
        return marked.decode()
    return FILES if "data" in schema.names else RECORDS


def is_tabular(name: str | Path, delimiter: str | None = None) -> bool:
    """Whether a file is imported as rows: by its extension, or any file once a delimiter
    is stated."""
    suffix = PurePosixPath(str(name)).suffix.lower()
    return delimiter is not None or suffix in PARQUET | JSON_LINES | DELIMITED


@dataclass(frozen=True)
class _File:
    name: str
    open: Callable[[], BinaryIO]


def _files(source: Any, delimiter: str | None, cfg: Any) -> list[_File]:
    """The tabular files of a path, a folder, or an s3:// address."""
    from fieldbook.data.dataset import DatasetError, dir_items

    if is_bucket(source):
        found = [(f.uri, f) for _, f in bucket_files(source, "", cfg)]
        files = [_File(uri, lambda f=f: f.fs.open(f.key, "rb")) for uri, f in found]
    elif isinstance(source, RemoteFile):
        files = [_File(source.uri, lambda: source.fs.open(source.key, "rb"))]
    else:
        path = Path(source)
        if not path.exists():
            raise DatasetError(f"file not found: {path}")
        paths = [file for _, file in dir_items(path)] if path.is_dir() else [path]
        files = [_File(str(p), lambda p=p: p.open("rb")) for p in paths]
    other = [f.name for f in files if not is_tabular(f.name, delimiter)]
    if other:
        raise DatasetError(
            f"not a table: {', '.join(other[:5])} (rows are read from Parquet, JSON lines, "
            ".csv, and .tsv; state --delimiter for delimited text under another extension)"
        )
    if not files:
        raise DatasetError(f"no files in {source}")
    return files


def _delimiter(file: _File) -> str:
    """The delimiter of a delimited text file, found from how it starts."""
    with file.open() as f:
        sample = f.read(SNIFF_BYTES).decode("utf-8", errors="replace")
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:  # one column, or nothing to tell by
        return "\t" if file.name.lower().endswith(".tsv") else ","


def _open(file: _File, delimiter: str | None) -> tuple[pa.Schema, Iterator[pa.RecordBatch]]:
    """A file's columns, and its rows a batch at a time."""
    import pyarrow.csv as pa_csv
    import pyarrow.json as pa_json
    import pyarrow.parquet as pq

    from fieldbook.data.dataset import DatasetError

    suffix = PurePosixPath(file.name).suffix.lower()
    try:
        if suffix in PARQUET:
            parquet = pq.ParquetFile(file.open())
            return parquet.schema_arrow, parquet.iter_batches()
        if suffix in JSON_LINES:
            table = pa_json.read_json(file.open())
            return table.schema, iter(table.to_batches())
        options = pa_csv.ParseOptions(delimiter=delimiter or _delimiter(file))
        reader = pa_csv.open_csv(file.open(), parse_options=options)
        return reader.schema, _guarded(reader, file.name)
    except (pa.ArrowException, OSError, UnicodeDecodeError) as e:
        raise DatasetError(f"cannot read {file.name} as a table: {_line(e)}") from None


def _guarded(reader: Any, name: str) -> Iterator[pa.RecordBatch]:
    from fieldbook.data.dataset import DatasetError

    try:
        yield from reader
    except pa.ArrowException as e:  # a value further down that does not fit its column
        raise DatasetError(f"cannot read {name} as a table: {_line(e)}") from None


def _line(error: Exception) -> str:
    return str(error).splitlines()[0]


def _in_memory(source: Any) -> pa.Table | None:
    from fieldbook.data.dataset import _table_from_rows

    if isinstance(source, pa.Table):
        return source
    if isinstance(source, pa.RecordBatch):
        return pa.Table.from_batches([source])
    if isinstance(source, list):
        return _table_from_rows(source)
    if type(source).__name__ == "DataFrame" and hasattr(source, "columns"):
        return pa.Table.from_pandas(source, preserve_index=False)
    return None


def incoming(
    edit: Edit, sources: list[tuple[Any, str | None]]
) -> tuple[pa.Schema, Callable[[], Iterator[pa.RecordBatch]]]:
    """Rows from `sources` (each with its delimiter): their columns, reconciled across
    the sources, and a function that reads them, a batch at a time. Files are opened
    here, so one that cannot be read fails before anything is written."""
    from fieldbook.data.dataset import DatasetError

    opened: list[tuple[pa.Schema, Callable[[], Iterator[pa.RecordBatch]]]] = []
    for source, delimiter in sources:
        table = _in_memory(source)
        if table is not None:
            opened.append((table.schema, lambda t=table: iter(t.to_batches())))
            continue
        for file in _files(source, delimiter, edit.ds.cfg):
            schema, _ = _open(file, delimiter)
            opened.append((schema, lambda f=file, d=delimiter: _open(f, d)[1]))
    try:
        schema = pa.unify_schemas(
            [pa.schema([]), *(pa.schema(list(s)) for s, _ in opened)],
            promote_options="permissive",
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        raise DatasetError(f"the sources' columns do not agree: {_line(e)}") from None

    def batches() -> Iterator[pa.RecordBatch]:
        for _, read in opened:
            yield from read()

    return schema, batches


def _conform(batch: pa.RecordBatch, schema: pa.Schema, name: str) -> pa.RecordBatch:
    """A batch with the table's columns, in its order and types; a column the batch lacks
    is null."""
    from fieldbook.data.dataset import DatasetError

    columns = []
    for field in schema:
        if field.name not in batch.schema.names:
            columns.append(pa.nulls(batch.num_rows, field.type))
            continue
        column = batch.column(field.name)
        try:
            columns.append(column if column.type == field.type else pc.cast(column, field.type))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            raise DatasetError(
                f"column {field.name!r} in {name} has type {field.type}; "
                f"cannot store {column.type} values in it"
            ) from None
    return pa.record_batch(columns, schema=pa.schema(list(schema)))


def _stream(
    edit: Edit, schema: pa.Schema, batches: Callable[[], Iterator[pa.RecordBatch]],
    declared: str | None, failure: list[BaseException],
) -> Iterator[pa.RecordBatch]:  # fmt: skip
    """The new rows as the table stores them, each batch checked against the dataset's
    schema on its way. Lance reports an error raised here as its own, so it is kept in
    `failure` for the caller to raise."""
    from fieldbook.data import schema as _schema

    edit.rows_added = 0
    try:
        for batch in batches():
            batch = _conform(batch, schema, edit.ds.name)
            if declared is not None:
                _schema.validate(pa.Table.from_batches([batch]), declared, edit.ds.name)
            edit.rows_added += batch.num_rows
            yield batch
    except BaseException as e:
        failure.append(e)
        raise


def _declared(edit: Edit, metadata: dict[bytes, bytes] | None) -> str | None:
    from fieldbook.data import schema as _schema

    return _schema.read_schema(pa.schema([], metadata=metadata))


def create(edit: Edit) -> int:
    """A records dataset's first version: one write of every staged row."""
    schema, batches = incoming(edit, edit._sources())
    schema = pa.schema([*schema, *_constant_columns(edit, set(schema.names))])
    metadata = {**(edit._metadata(pa.schema([])) or {}), KIND_KEY: RECORDS.encode()}
    declared = _declared(edit, metadata)
    failure: list[BaseException] = []
    with contextlib.ExitStack() as stack:
        if not _simple(edit):  # else filters come first and match nothing: no rows yet
            _, _, batches = stack.enter_context(_in_order(edit, None, schema))
        stream = _stream(edit, schema, batches, declared, failure)
        return _write_first(edit, schema.with_metadata(metadata), stream, failure)


def _write_first(edit: Edit, schema: pa.Schema, stream: Any, failure: list) -> int:
    from fieldbook.data.dataset import _create_table

    reader = pa.RecordBatchReader.from_batches(schema, stream)
    try:
        return _create_table(edit.ds, schema, reader, edit._record())
    except Exception:
        if failure:
            raise failure[0] from None
        raise


def check(edit: Edit, index: Any | None) -> None:
    """Every check of a save, with nothing written: for a smoke run."""
    from fieldbook.data import schema as _schema

    schema, batches = incoming(edit, edit._sources())
    known = set(schema.names) | set(index.schema.names if index is not None else ())
    extra = [*schema, *_constant_columns(edit, known)]
    if index is not None:
        extra = [f for f in extra if f.name not in index.schema.names]
    target = pa.schema([*(index.schema if index is not None else ()), *extra])
    metadata = edit._metadata(index.schema if index is not None else pa.schema([]))
    declared = _declared(edit, metadata)
    if _simple(edit):
        for _ in _stream(edit, target, batches, declared, []):
            pass
        return
    with _in_order(edit, index, target, declared is not None) as (kept, _, new):
        if declared is not None and kept.num_rows:
            _schema.validate(kept.drop_columns([ORIGIN, ADDRESS]), declared, edit.ds.name)
        for _ in _stream(edit, target, new, declared, []):
            pass


def _constant_columns(edit: Edit, known: set[str]) -> list[pa.Field]:
    """Columns that `update_records` names and nothing has yet: typed by the value given."""
    fields = []
    for name, value in edit._constants().items():
        if name not in known:
            kind = pa.array([value]).type
            fields.append(pa.field(name, pa.string() if pa.types.is_null(kind) else kind))
    return fields


def _simple(edit: Edit) -> bool:
    """Whether the edit's operations can each be worked out from the saved rows alone:
    removals, then at most one update, then additions. Any other order has a filter that
    must see what an earlier operation of the edit left (`_in_order`)."""
    kinds = [op[0] for op in edit._row_ops]
    while kinds and kinds[0] == "remove":
        kinds.pop(0)
    if kinds and kinds[0] == "update":
        kinds.pop(0)
    return all(kind == "add" for kind in kinds)


ORIGIN, ADDRESS = "__origin", "__address"  # of a saved row the edit changes: its ID, its place


@contextlib.contextmanager
def _in_order(
    edit: Edit, index: Any | None, schema: pa.Schema, whole_rows: bool = False
) -> Iterator[tuple[pa.Table, set[int], Callable[[], Iterator[pa.RecordBatch]]]]:
    """The edit's operations applied one after another, each seeing what the earlier ones
    left: a filter reaches rows the edit added and rows it already changed.

    The rows the edit adds or changes are kept in a working Lance table on local disk,
    so its filters are evaluated by Lance, as they are on the dataset. A saved row is
    read into it when an update first matches it; from then on it is matched there, not
    in the dataset. Gives the changed saved rows (`ORIGIN`, `ADDRESS`, and the columns
    updates set, or every column if `whole_rows`), the places of the saved rows removed,
    and a function that reads the rows to add a batch at a time, from the working table:
    it is gone when the `with` block ends.
    """
    import tempfile

    from fieldbook.data.dataset import FILE_FORMAT, DatasetError, _as_type

    name = edit.ds.name
    plain = pa.schema(list(schema))
    working = pa.schema([*plain, pa.field(ORIGIN, pa.uint64()), pa.field(ADDRESS, pa.uint64())])
    touched: set[int] = set()  # saved rows now in the working table, by place
    removed: set[int] = set()

    def saved(where: str, columns: list[str] | None) -> pa.Table | None:
        """Saved rows matching a filter that the edit has not changed or removed yet."""
        if index is None:
            return None
        found = index.to_table(columns=columns, filter=where, with_row_id=True,
                               with_row_address=True)  # fmt: skip
        skip = pa.array(sorted(touched | removed), pa.uint64())
        return found.filter(pc.invert(pc.is_in(found["_rowaddr"], value_set=skip)))

    def marked(batch: pa.RecordBatch) -> pa.RecordBatch:
        batch = _conform(batch, plain, name)
        none = pa.nulls(batch.num_rows, pa.uint64())
        return pa.record_batch([*batch.columns, none, none], schema=working)

    with tempfile.TemporaryDirectory() as folder:
        table = lance.write_dataset(working.empty_table(), folder, data_storage_version=FILE_FORMAT)
        for kind, *given in edit._row_ops:
            try:
                if kind == "add":
                    _, batches = incoming(edit, [tuple(given)])
                    reader = pa.RecordBatchReader.from_batches(working, map(marked, batches()))
                    table = lance.write_dataset(reader, folder, mode="append")
                    continue
                where = given[0]
                if kind == "remove":
                    found = saved(where, [])
                    if found is not None:
                        removed.update(found["_rowaddr"].to_pylist())
                    table.delete(where)
                    continue
                rows = table.to_table(filter=where)
                table.delete(where)
                found = saved(where, None)
                if found is not None and found.num_rows:
                    touched.update(found["_rowaddr"].to_pylist())
                    names = {"_rowid": ORIGIN, "_rowaddr": ADDRESS}
                    found = found.rename_columns([names.get(c, c) for c in found.column_names])
                    found = found.select(working.names).cast(working)
                    rows = pa.concat_tables([rows, found])
                for column, value in given[1].items():
                    constant = _as_type([value], plain.field(column).type, column, name)[0]
                    filled = pa.array([constant.as_py()] * rows.num_rows, plain.field(column).type)
                    rows = rows.set_column(rows.column_names.index(column), column, filled)
                if rows.num_rows:
                    table = lance.write_dataset(rows, folder, mode="append")
            except DatasetError:
                raise
            except (ValueError, OSError) as e:
                raise DatasetError(f"invalid filter: {_line(e).split(', /')[0]}") from None
        columns = None if whole_rows else [ORIGIN, ADDRESS, *sorted(edit._constants())]
        kept = table.to_table(columns=columns, filter=f"{ORIGIN} IS NOT NULL")
        removed |= touched - set(kept[ADDRESS].to_pylist())  # changed, then removed
        yield (
            kept,
            removed,
            lambda: table.to_batches(columns=plain.names, filter=f"{ORIGIN} IS NULL"),
        )


def save(edit: Edit, index: Any) -> int:
    """Commit the staged row changes on `index`'s version. Raises Lance's
    CommitConflictError if another version was committed after it."""
    from fieldbook.data import schema as _schema
    from fieldbook.data.dataset import _KEEP, DatasetError, _as_type, _open

    ds = edit.ds
    uri, options = ds._write_uri(), storage.lance_options(ds.cfg)
    index = edit._restructure(index, uri)
    schema, batches = incoming(edit, edit._sources())
    constants = edit._constants()
    existing = set(index.schema.names)
    new_columns = [f for f in schema if f.name not in existing]
    new_columns += _constant_columns(edit, existing | set(schema.names))
    if new_columns:
        # Lance's own call, which cannot be told which version it follows: checked first.
        if _open(uri, ds.cfg).version != index.version:
            raise lance_commit.CommitConflictError(f"{ds.name} changed since it was read")
        index.add_columns(pa.schema(new_columns))
        index = _open(uri, ds.cfg)
    declared = _declared(edit, index.schema.metadata)
    every_row = declared is not None and edit._schema is not _KEEP  # a new schema
    fragments = {f.fragment_id: f for f in index.get_fragments()}
    updated: dict[int, Any] = {}
    modified: list[int] = []
    gone: list[int] = []

    def remove(fragment_id: int, left: Any) -> None:
        if left is None:  # every row of this data file is removed
            gone.append(fragment_id)
            updated.pop(fragment_id, None)
        elif left.deletion_file != fragments[fragment_id].metadata.deletion_file:
            base = updated.get(fragment_id, left)
            updated[fragment_id] = replace(base, deletion_file=left.deletion_file)

    def change(values: pa.Table, places: Any) -> None:
        nonlocal modified
        files = pc.unique(pc.shift_right(places, pa.scalar(32, pa.uint64())))
        for fragment_id in sorted(files.to_pylist()):
            updated[fragment_id], modified = fragments[fragment_id].update_columns(
                values, left_on="_rowid", right_on="_rowid"
            )

    if _simple(edit):
        removals = [op[1] for op in edit._row_ops if op[0] == "remove"]
        removed = " OR ".join(f"({where})" for where in removals)
        update = next((op for op in edit._row_ops if op[0] == "update"), None)
        try:
            if every_row:
                kept = index.to_table(filter=f"({removed}) IS NOT TRUE" if removed else None)
                _schema.validate(kept, declared, ds.name)
            if update is not None:
                _, where, fields = update
                current = index.to_table(
                    columns=None if declared else sorted(fields), filter=where,
                    with_row_id=True, with_row_address=True,
                )  # fmt: skip
                for name, value in fields.items():
                    kind = index.schema.field(name).type
                    constant = _as_type([value], kind, name, ds.name)[0].as_py()
                    column = pa.array([constant] * current.num_rows, kind)
                    current = current.set_column(current.column_names.index(name), name, column)
                if declared is not None and current.num_rows:
                    rows = current.drop_columns(["_rowid", "_rowaddr"])
                    _schema.validate(rows, declared, ds.name)
                change(current.select(["_rowid", *sorted(fields)]), current["_rowaddr"])
            if removed:
                for fragment_id, fragment in fragments.items():
                    remove(fragment_id, fragment.delete(removed))
        except (ValueError, OSError) as e:
            raise DatasetError(f"invalid filter: {_line(e).split(', /')[0]}") from None
        appended = []
        if edit._sources():
            appended = _append(edit, index, uri, options, batches, declared)
    else:
        plain = pa.schema(list(index.schema))
        whole = declared is not None
        with _in_order(edit, index, plain, whole) as (kept, removed_places, new):
            if declared is not None:
                rows = kept.drop_columns([ORIGIN, ADDRESS])
                if every_row:
                    rest = index.to_table(with_row_address=True)
                    changed = set(kept[ADDRESS].to_pylist())
                    skip = pa.array(sorted(removed_places | changed), pa.uint64())
                    rest = rest.filter(pc.invert(pc.is_in(rest["_rowaddr"], value_set=skip)))
                    rows = pa.concat_tables([rest.drop_columns(["_rowaddr"]), rows])
                if rows.num_rows:
                    _schema.validate(rows, declared, ds.name)
            if kept.num_rows:
                values = kept.select([ORIGIN, *sorted(constants)]).rename_columns(
                    ["_rowid", *sorted(constants)]
                )
                change(values, kept[ADDRESS])
            offsets: dict[int, list[int]] = {}
            for place in removed_places:
                offsets.setdefault(place >> 32, []).append(place & 0xFFFFFFFF)
            for fragment_id, places in offsets.items():
                remove(fragment_id, fragments[fragment_id].delete_rows(places))
            # Written again if the save is redone: these rows depend on the edit's filters.
            edit._appended = None
            appended = _append(edit, index, uri, options, new, declared)
    if fragments:  # named in every save, so that Lance refuses the second of two at once
        anchor = min(fragments)
        if anchor not in updated and anchor not in gone:
            updated[anchor] = fragments[anchor].metadata
    operation = lance.LanceOperation.Update(
        removed_fragment_ids=gone, updated_fragments=list(updated.values()),
        new_fragments=appended, fields_modified=list(modified),
    )  # fmt: skip
    return edit._commit(index, operation, edit._record()).version


def _append(
    edit: Edit, index: Any, uri: str, options: dict[str, str],
    batches: Callable[[], Iterator[pa.RecordBatch]], declared: str | None,
) -> list:  # fmt: skip
    """Write the new rows as data files of their own, not yet part of any version. Kept
    if the save has to be redone on a later version: rows are written once."""
    names = tuple(index.schema.names)
    if edit._appended is None or edit._appended[0] != names:
        schema = pa.schema(list(index.schema))
        failure: list[BaseException] = []
        stream = _stream(edit, schema, batches, declared, failure)
        try:
            written = lance.fragment.write_fragments(
                pa.RecordBatchReader.from_batches(schema, stream), uri, schema=schema,
                mode="append", storage_options=options,
            )  # fmt: skip
        except Exception:
            if failure:
                raise failure[0] from None
            raise
        edit._appended = (names, written)
    return edit._appended[1]
