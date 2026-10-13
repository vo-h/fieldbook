"""`fieldbook data`: versioned datasets."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from fieldbook.cli._common import fail


class _OnFirstUse:
    """A module imported when first used: `fieldbook.data` loads pyarrow, Lance, and pandas
    (about 2 s), which commands other than `fieldbook data` (`config`, `run`) do not need."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        import importlib

        return getattr(importlib.import_module(self._name), attr)


data = _OnFirstUse("fieldbook.data")
cleanup = _OnFirstUse("fieldbook.data.cleanup")

app = typer.Typer(no_args_is_help=True, help="Versioned datasets.")


def _parse_value(raw: str) -> Any:
    """JSON if it parses (5, 0.9, true, null), otherwise the string itself."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_field(spec: str) -> tuple[str, Any]:
    key, sep, raw = spec.partition("=")
    if not sep or not key:
        raise ValueError(f"expected KEY=VALUE, got {spec!r}")
    return key, _parse_value(raw)


def _read_values(file: Path) -> dict[str, dict[str, Any]]:
    """Per-item values from a CSV with a `path` column; empty cells are skipped."""
    with file.open(newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            raise ValueError(f"{file} has no 'path' column")
        values: dict[str, dict[str, Any]] = {}
        for row in reader:
            path = row.pop("path")
            if path in values:
                raise ValueError(f"{file}: duplicate path {path!r}")
            values[path] = {k: _parse_value(v) for k, v in row.items() if k and v}
    return values


def _parse_info(specs: list[str] | None) -> dict[str, str | None]:
    """KEY=VALUE pairs as strings; an empty VALUE removes the key."""
    info: dict[str, str | None] = {}
    for spec in specs or []:
        key, sep, value = spec.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {spec!r}")
        info[key] = value or None
    return info


def _progress(total: int, label: str):
    """A progress bar on stderr for large batches in an interactive terminal, else None."""
    if total < 100 or not sys.stderr.isatty():
        return None
    return typer.progressbar(length=total, label=label, file=sys.stderr)


def _parse_ref(ref: str) -> int | str:
    """A version number or a tag."""
    return int(ref) if ref.isdigit() else ref


def _split_ref(spec: str) -> tuple[str, int | str | None]:
    """Parse NAME, NAME@VERSION, or NAME@TAG."""
    name, _, ref = spec.partition("@")
    return name, _parse_ref(ref) if ref else None


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _echo_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows, strict=True)]
    for row in [headers, *rows]:
        typer.echo("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


@app.command("ls")
def data_ls(
    spec: Annotated[
        str | None, typer.Argument(metavar="[NAME[@VERSION|TAG]]", show_default=False)
    ] = None,
    versions: Annotated[bool, typer.Option("--versions", help="Version history.")] = False,
    tags: Annotated[bool, typer.Option("--tags", help="Tags and their versions.")] = False,
    stats: Annotated[bool, typer.Option("--stats", help="Item count and total size.")] = False,
    diff: Annotated[
        str | None,
        typer.Option(
            "--diff",
            metavar="FROM",
            help="Changes from version or tag FROM to NAME's version (latest or @VERSION).",
        ),
    ] = None,
    info: Annotated[bool, typer.Option("--info", help="Dataset info.")] = False,
    show_lineage: Annotated[
        bool, typer.Option("--lineage", help="The version's inputs, recursively.")
    ] = False,
    show_schema: Annotated[
        bool, typer.Option("--schema", help="Dataset schema (Pandera YAML).")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify", help="Check that stored files exist and match (all datasets if no NAME)."
        ),
    ] = False,
    deep: Annotated[bool, typer.Option("--deep", help="With --verify: re-hash content.")] = False,
    depth: Annotated[
        int, typer.Option(help="With --stats: group by the first N directory levels.")
    ] = 0,
    by: Annotated[
        list[str] | None, typer.Option(help="With --stats: group by a column; repeatable.")
    ] = None,
    prefix: Annotated[str, typer.Option(help="Only paths starting with this prefix.")] = "",
    where: Annotated[
        str | None, typer.Option(help="SQL filter on index columns, e.g. \"label = 'cat'\".")
    ] = None,
    long: Annotated[bool, typer.Option("--long", "-l", help="Show size and metadata.")] = False,
    limit: Annotated[int | None, typer.Option(help="Show at most N items.")] = None,
) -> None:
    """List datasets, or a dataset's items, versions, tags, or changes.

    \b
    fieldbook data ls                     datasets
    fieldbook data ls cats[@3]            items (filter with --prefix, --where)
    fieldbook data ls cats[@3] --stats --depth 1
                                          count and size per folder
    fieldbook data ls cats --versions     version history
    fieldbook data ls cats --tags         tags
    fieldbook data ls cats[@5] --diff 3   changes since v3 (records: - and + rows)
    fieldbook data ls cats --info         description, license, source, ...
    fieldbook data ls cats --schema       schema checked on every commit
    fieldbook data ls cats[@5] --lineage  inputs of the version, recursively
    fieldbook data ls [cats] --verify [--deep]
                                          check stored files
    """
    views = {
        "--versions": versions,
        "--tags": tags,
        "--stats": stats,
        "--diff": diff is not None,
        "--info": info,
        "--schema": show_schema,
        "--lineage": show_lineage,
        "--verify": verify,
    }
    selected = [v for v, on in views.items() if on]
    if len(selected) > 1:
        raise fail(f"use at most one of {', '.join(views)}")
    view = selected[0] if selected else "items"
    name, ref = _split_ref(spec) if spec else (None, None)
    used = {
        "@VERSION": ref is not None,
        "--prefix": bool(prefix),
        "--where": where is not None,
        "--long": long,
        "--limit": limit is not None,
        "--depth": bool(depth),
        "--by": bool(by),
        "--deep": deep,
    }
    allowed = {
        "items": {"@VERSION", "--prefix", "--where", "--long", "--limit"},
        "--stats": {"@VERSION", "--prefix", "--where", "--depth", "--by"},
        "--diff": {"@VERSION", "--prefix", "--limit"},
        "--info": {"@VERSION"},
        "--schema": {"@VERSION"},
        "--lineage": {"@VERSION"},
        "--versions": set(),
        "--tags": set(),
        "--verify": {"--deep"},
    }
    invalid = [option for option, on in used.items() if on and option not in allowed[view]]
    if invalid:
        target = "an item listing" if view == "items" else view
        raise fail(f"{', '.join(invalid)} cannot be used with {target}")
    if name is None and view not in ("items", "--verify"):
        raise fail(f"NAME is required for {view}")
    if name is None and view == "items" and any(used.values()):
        raise fail("NAME is required for these options")
    try:
        if view == "--verify":
            _ls_verify(
                data.verify(data.dataset(name), deep=deep) if name else data.verify(deep=deep)
            )
            return
        if name is None:
            _ls_datasets()
            return
        ds = data.dataset(name, ref)
        if view in ("items", "--stats") and ds.kind == "records":
            if prefix or long or depth or by:
                raise fail("--prefix, --long, --depth, and --by apply to datasets of files")
            if view == "--stats":
                _ls_columns(ds, where)
            else:
                _ls_rows(ds.to_table(where, limit=limit if limit is not None else ROWS_SHOWN))
            return
        if view == "--versions":
            _ls_versions(ds)
        elif view == "--tags":
            _echo_table(["tag", "version"], [[t, f"v{v}"] for t, v in sorted(ds.tags().items())])
        elif view == "--stats":
            _ls_stats(ds.stats(prefix, where, depth=depth, by=by or []))
        elif view == "--diff":
            _ls_diff(ds.diff(_parse_ref(diff), prefix), f"{name}@{diff}", spec, limit)
        elif view == "--info":
            _echo_table(["key", "value"], [[k, v] for k, v in sorted(ds.info().items())])
        elif view == "--lineage":
            _ls_lineage(ds)
        elif view == "--schema":
            typer.echo((ds.schema() or "no schema\n").rstrip("\n"))
        else:
            _ls_items(ds.files(prefix, where), long, limit)
    except data.DatasetError as e:
        raise fail(str(e)) from None


@app.command("cat")
def data_cat(
    spec: Annotated[str, typer.Argument(metavar="NAME[@VERSION|TAG]")],
    path: str,
) -> None:
    """Write an item's content to stdout."""
    name, ref = _split_ref(spec)
    try:
        with data.dataset(name, ref).open(path) as f:
            shutil.copyfileobj(f, sys.stdout.buffer)
    except data.DatasetError as e:
        raise fail(str(e)) from None
    except KeyError:
        raise fail(f"{path!r} not found in {spec}") from None


@app.command("checkout")
def data_checkout(
    spec: Annotated[str, typer.Argument(metavar="NAME[@VERSION|TAG]")],
    target: Annotated[Path, typer.Argument(metavar="DIR", help="Where to write the files.")],
    prefix: Annotated[str, typer.Option(help="Only items under this path.")] = "",
) -> None:
    """Write a dataset's files into a folder, for tools that need real files."""
    name, ref = _split_ref(spec)
    try:
        count = data.dataset(name, ref).checkout(target, prefix)
    except data.DatasetError as e:
        raise fail(str(e)) from None
    typer.echo(f"{spec}: wrote {count} files to {target}")


@app.command("query")
def data_query(
    datasets: Annotated[
        list[str],
        typer.Option(
            "--dataset",
            "-d",
            metavar="[ALIAS=]NAME[@VERSION|TAG]",
            help="Dataset to query, available in SQL as NAME (or ALIAS); repeatable.",
        ),
    ],
    query: Annotated[str, typer.Option("--query", "-q", help="SQL (DuckDB dialect).")],
    limit: Annotated[int, typer.Option(help="Show at most N rows; 0 shows all.")] = 100,
) -> None:
    """Run SQL over dataset versions: a records dataset's rows, or a files dataset's index.

    A files dataset is a table of its paths and metadata (not the files' content).

    \b
    fieldbook data query -d sales@3 -q "SELECT count(*) FROM sales"
    fieldbook data query -d old=sales@2 -d new=sales -q "... FROM new, old"
    fieldbook data query -d sales -q "COPY (SELECT * FROM sales) TO 'out.parquet'"
    """
    from fieldbook.data import query as q

    views: dict[str, tuple[str, int | str | None]] = {}
    for spec in datasets:
        alias, sep, target = spec.rpartition("=")
        name, ref = _split_ref(target)
        view = alias if sep else name
        if view in views:
            raise fail(f"{view!r} is used twice; name one with ALIAS=")
        views[view] = (name, ref)
    try:
        con = q.connect()
    except data.DatasetError as e:
        raise fail(str(e)) from None
    import duckdb

    try:
        for view, (name, ref) in views.items():
            ds = data.dataset(name, ref)
            if not ds.exists:
                raise data.DatasetNotFoundError(f"dataset {ds.key} not found")
            q.register(con, ds, view)
        result = con.sql(query)
        if result is None:
            return
        rows = result.fetchmany(limit + 1) if limit else result.fetchall()
    except (data.DatasetError, duckdb.Error) as e:
        raise fail(str(e)) from None
    shown = rows[:limit] if limit else rows
    _echo_table(list(result.columns), [["" if v is None else str(v) for v in row] for row in shown])
    if limit and len(rows) > limit:
        typer.echo(f"(first {limit} rows; use --limit 0 for all)", err=True)


def _ls_datasets() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def row(name: str) -> list[str]:
        ds = data.dataset(name)
        if ds.kind == "records":
            return [name, "records", f"v{ds.version}", str(ds.index().count_rows()), "-"]
        total = ds.stats().to_pylist()[0]
        return [name, "files", f"v{ds.version}", str(total["files"]),
                _human_size(total["bytes"])]  # fmt: skip

    names = data.list_datasets()
    with ThreadPoolExecutor(max_workers=min(8, len(names) or 1)) as pool:  # read in parallel
        rows = list(pool.map(row, names))
    _echo_table(["dataset", "kind", "version", "items", "size"], rows)


def _ls_versions(ds: data.Dataset) -> None:
    rows = [
        [
            f"v{v.version}",
            v.timestamp.strftime("%Y-%m-%d %H:%M"),
            v.author + (f" [{v.agent}]" if v.agent else ""),
            str(v.rows),
            ", ".join(v.tags),
            v.run_id,
            v.message,
            ", ".join(_short_ref(i, ds.project) for i in v.inputs),
            v.command,
        ]
        for v in ds.log()
    ]
    _echo_table(
        ["version", "date", "author", "items", "tags", "run", "message", "inputs", "command"],
        rows,
    )


def _ls_lineage(ds: data.Dataset) -> None:
    for node in ds.lineage():
        indent = "  " * node.depth
        ref = _short_ref(node.ref, ds.project)
        if node.info is None:
            typer.echo(f"{indent}{ref}  ({node.status})")
            continue
        info = node.info
        details = [
            info.timestamp.strftime("%Y-%m-%d %H:%M"),
            info.author + (f" [{info.agent}]" if info.agent else ""),
        ]
        details += [f"run {info.run_id}"] if info.run_id else []
        details += [f"code {info.code[:12]}"] if info.code else []
        details += [f"git {_short_git(info.git)}"] if info.git else []
        details += [repr(info.message)] if info.message else []
        details += [f"$ {info.command}"] if info.command else []
        typer.echo(f"{indent}{ref}  {'  '.join(details)}")


def _short_git(state: str) -> str:
    from fieldbook.code import snapshot

    return snapshot.short_git(state)


def _short_ref(ref: str, project: str) -> str:
    """`name@v` for versions in `project`, otherwise the ref as stored."""
    return ref.removeprefix(f"{project}/")


def _ls_items(table, long: bool, limit: int | None) -> None:
    rows = table.slice(0, limit).to_pylist() if limit is not None else table.to_pylist()
    if not long:
        for row in rows:
            typer.echo(row["path"])
        return
    extra = [c for c in table.column_names if c not in data.BASE_COLUMNS]
    _echo_table(
        ["path", "size", "type", *extra],
        [
            [
                r["path"],
                _human_size(r["size"]),
                r["mime_type"],
                *("" if r[c] is None else str(r[c]) for c in extra),
            ]
            for r in rows
        ],
    )


ROWS_SHOWN = 20  # rows of a records dataset listed when --limit is not given


def _ls_rows(table) -> None:
    rows = [["" if v is None else str(v) for v in row.values()] for row in table.to_pylist()]
    _echo_table(table.column_names, rows)


def _ls_columns(ds: data.Dataset, where: str | None) -> None:
    index = ds.index()
    try:
        typer.echo(f"{index.count_rows(filter=where)} rows")
    except (ValueError, OSError) as e:
        raise fail(f"invalid filter {where!r}: {str(e).splitlines()[0]}") from None
    _echo_table(["column", "type"], [[f.name, str(f.type)] for f in index.schema])


def _ls_diff(table, old: str, new: str, limit: int | None) -> None:
    if "_change" in table.column_names:  # records: the rows gone and the rows new
        marks = {"removed": "-", "added": "+"}
        shown = table.slice(0, limit if limit is not None else ROWS_SHOWN)
        columns = [c for c in table.column_names if c != "_change"]
        typer.echo(f"--- {old}\n+++ {new}")
        _echo_table(
            ["", *columns],
            [
                [marks[r["_change"]], *("" if r[c] is None else str(r[c]) for c in columns)]
                for r in shown.to_pylist()
            ],
        )
        changes = table["_change"].to_pylist()
        typer.echo(f"{changes.count('removed')} removed, {changes.count('added')} added")
        if shown.num_rows < table.num_rows:
            typer.echo(f"(first {shown.num_rows} differing rows; --limit N for more)", err=True)
        return
    if limit is not None:
        raise fail("--limit applies to --diff on a dataset of records")
    rows = table.to_pylist()
    _echo_table(
        ["path", "change", "changed"], [[r["path"], r["change"], r["changed"]] for r in rows]
    )
    counts = {c: sum(r["change"] == c for r in rows) for c in ("added", "removed", "modified")}
    typer.echo(", ".join(f"{n} {c}" for c, n in counts.items()))


def _ls_verify(report: data.VerifyReport) -> None:
    for where in report.missing:
        typer.echo(f"missing  {where}")
    for where in report.corrupt:
        typer.echo(f"corrupt  {where}")
    check = "content hashes" if report.deep else "sizes"
    typer.echo(
        f"checked {report.objects} files ({check}) in {len(report.datasets)} datasets: "
        f"{len(report.missing)} missing, {len(report.corrupt)} corrupt"
    )
    if not report.ok:
        raise typer.Exit(1)


def _ls_stats(table) -> None:
    keys = [c for c in table.column_names if c not in ("files", "bytes")]
    total_files = sum(table["files"].to_pylist())
    total_bytes = sum(b or 0 for b in table["bytes"].to_pylist())
    rows = [
        [
            *("" if r[k] is None else str(r[k]) for k in keys),
            str(r["files"]),
            f"{100 * r['files'] / total_files:.1f}%" if total_files else "-",
            _human_size(r["bytes"] or 0),
        ]
        for r in table.to_pylist()
    ]
    if keys:
        rows.append(
            ["total", *[""] * (len(keys) - 1), str(total_files), "100.0%", _human_size(total_bytes)]
        )
    _echo_table([*keys, "items", "share", "size"], rows)


@app.command("add")
def data_add(
    name: str,
    sources: Annotated[
        list[str] | None,
        typer.Argument(metavar="[SRC]...", help="Files, directories, or s3://bucket/prefix."),
    ] = None,
    prefix: Annotated[str, typer.Option(help="Prefix for item paths.")] = "",
    hidden: Annotated[
        bool,
        typer.Option("--hidden", help="Include files and directories starting with '.'."),
    ] = False,
    fields: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="KEY=VALUE metadata for every added item; repeatable. Values are parsed "
            "as JSON (5, 0.9, true, null), otherwise kept as strings.",
        ),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Tag the version this add creates. Must be a new tag."),
    ] = None,
    info: Annotated[
        list[str] | None,
        typer.Option("--info", help="KEY=VALUE dataset info (e.g. license=CC-BY-4.0); repeatable."),
    ] = None,
    copy: Annotated[
        str | None,
        typer.Option(
            "--copy",
            metavar="NAME[@VERSION|TAG]",
            help="Add the items of another dataset version without copying content.",
        ),
    ] = None,
    where: Annotated[
        str | None,
        typer.Option(help="With --copy: SQL filter on the source index, e.g. \"label = 'cat'\"."),
    ] = None,
    delimiter: Annotated[
        str | None,
        typer.Option(help="Read the sources as delimited text with this delimiter ('\\t': tab)."),
    ] = None,
    message: Annotated[str, typer.Option("--message", "-m", help="Commit message.")] = "",
) -> None:
    """Add files, or the rows of tables, to a dataset. The first add creates the dataset.

    What is first added decides what the dataset holds from then on. Tables only
    (Parquet, JSON lines, .csv, .tsv): their rows are imported. Anything else, or a mix:
    every file is stored as a file.

    \b
    fieldbook data add sales 2025.parquet q1.csv         import rows
    fieldbook data add sales export.txt --delimiter '|'  any delimited text
    fieldbook data add cats a.jpg imgs/ b.jpg            add files and folders
    fieldbook data add cats imgs/ --prefix train/        under a path prefix
    fieldbook data add cats a.jpg --set label=tabby      with metadata
    fieldbook data add cats imgs/ --tag batch-3          tag the new version
    fieldbook data add cats imgs/ --info license=MIT     set dataset info
    fieldbook data add cats s3://open-data/cats/         read from an S3 bucket
    fieldbook data add clean --copy raw@3 --where "blurry = false"
                                                         copy items

    Paths that already exist are refused; change them with `fieldbook data update`.
    Tags on versions that already exist are set or moved with `fieldbook data update --tag`.
    """
    sources = sources or []
    if not sources and copy is None:
        raise fail("specify at least one SRC, or --copy")
    if where is not None and copy is None:
        raise fail("--where applies to --copy")
    try:
        ds = data.dataset(name)
        if tag is not None:
            data.validate_tag(tag)
    except data.DatasetError as e:
        raise fail(str(e)) from None
    if tag is not None:
        if ds.exists and tag in ds.tags():
            raise fail(
                f"tag {tag!r} already exists on {name}; move it with `fieldbook data update --tag`"
            )
    from fieldbook import storage
    from fieldbook.data import remote

    buckets = [s for s in sources if remote.is_bucket(s)]
    local = [Path(s) for s in sources if not remote.is_bucket(s)]
    missing = [str(s) for s in local if not s.exists()]
    if missing:
        raise fail(f"not found: {', '.join(missing)}")
    try:
        values = dict(_parse_field(f) for f in fields or [])
        info_values = _parse_info(info)
        items: list[tuple[str, Path]] = []
        for src in local:
            if src.is_dir():
                items.extend(data.dir_items(src, prefix, include_hidden=hidden))
            else:
                items.append((prefix + src.name, src))
        if _adds_records(ds, items, buckets, copy, delimiter):
            if prefix or hidden or values or copy is not None:
                raise fail("--prefix, --hidden, --set, and --copy apply to datasets of files")
            if delimiter is not None:
                delimiter = delimiter.encode().decode("unicode_escape")  # '\t' as typed
            with ds.edit(message) as edit:
                for src in sources:
                    edit.add_records(src, delimiter=delimiter)
                edit.set_info(**info_values)
            if tag is not None:
                ds.tag(tag, edit.version)
            tagged = f", tagged {tag}" if tag is not None else ""
            typer.echo(f"{name}: added {edit.rows_added} rows, v{edit.version}{tagged}")
            return
        if delimiter is not None:
            raise fail(f"--delimiter reads tables into rows; {name} holds files")
        with ds.edit(message) as edit:
            count = 0
            if copy is not None:
                count += edit.copy_from(
                    data.dataset(*_split_ref(copy)), where, prefix=prefix, **values
                )
            for src in buckets:
                typer.echo(_bucket_note(src, ds), err=True)
                count += edit.add_dir(src, prefix, include_hidden=hidden, **values)
            bar = _progress(len(items), f"Adding {len(items)} files")
            if bar is None:
                count += edit.add_many(items, **values)
            else:
                with bar:
                    count += edit.add_many(items, progress=bar.update, **values)
            edit.set_info(**info_values)
    except (data.DatasetError, ValueError, storage.StorageError) as e:
        raise fail(str(e)) from None
    if tag is not None:
        ds.tag(tag, edit.version)
    tagged = f", tagged {tag}" if tag is not None else ""
    typer.echo(f"{name}: added {count} items, v{edit.version}{tagged}")


def _adds_records(
    ds: Any, items: list, buckets: list[str], copy: str | None, delimiter: str | None
) -> bool:
    """Whether an add imports rows: into a records dataset always; into a new dataset
    when every source is a table (data.md §1)."""
    from fieldbook.data import records

    if ds.exists:
        return ds.kind == records.RECORDS
    if copy is not None or not (items or buckets):
        return False
    from fieldbook.data import remote

    names = [file for _, file in items]
    names += [f.key for src in buckets for _, f in remote.files(src, "", ds.cfg)]
    return all(records.is_tabular(name, delimiter) for name in names)


def _bucket_note(src: str, ds: Any) -> str:
    """How a bucket source will be copied: inside S3, or down to this machine; and a
    warning when the two buckets are in different regions (AWS may charge for the move)."""
    from fieldbook import storage
    from fieldbook.clouds import aws
    from fieldbook.data import remote

    if storage.scheme(ds.cfg) != "s3":
        return f"reading {src} into the dataset on this machine"
    note = f"reading {src} into the dataset (its files pass through this machine)"
    if not ds.cfg.storage_endpoint:
        here, there = storage.s3_region(ds.cfg), aws.bucket_region(remote.bucket(src))
        if here and there and here != there:
            note += f"\nnote: the source is in {there}, your bucket in {here}: AWS may charge "
            note += "for moving data between regions"
    return note


@app.command("update")
def data_update(
    name: str,
    path: Annotated[str | None, typer.Argument(metavar="[PATH]")] = None,
    file: Annotated[
        Path | None, typer.Option("--file", help="Replace the item's content with this file.")
    ] = None,
    fields: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="KEY=VALUE; repeatable. Values are parsed as JSON (5, 0.9, true, null), "
            "otherwise kept as strings.",
        ),
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Create or move a tag.")] = None,
    version: Annotated[
        int | None, typer.Option(help="Version for --tag (default: latest).")
    ] = None,
    revert: Annotated[
        str | None,
        typer.Option(
            "--revert", metavar="VERSION", help="Make a new version identical to VERSION or tag."
        ),
    ] = None,
    info: Annotated[
        list[str] | None,
        typer.Option("--info", help="KEY=VALUE dataset info; empty VALUE removes; repeatable."),
    ] = None,
    values_file: Annotated[
        Path | None,
        typer.Option(
            "--from",
            metavar="CSV",
            help="Set values per item from a CSV with a path column; empty cells are skipped.",
        ),
    ] = None,
    drop: Annotated[
        list[str] | None,
        typer.Option("--drop", metavar="COL", help="Remove a column; repeatable."),
    ] = None,
    move: Annotated[
        tuple[str, str] | None,
        typer.Option(
            "--move",
            metavar="OLD NEW",
            help="Rename an item, or a directory if both end with '/'. Content is kept.",
        ),
    ] = None,
    schema_file: Annotated[
        str | None,
        typer.Option(
            "--schema",
            metavar="YAML",
            help="Set the schema checked on every commit (Pandera YAML file); '' removes it.",
        ),
    ] = None,
    where: Annotated[
        str | None,
        typer.Option(help='Records: the rows --set applies to, e.g. "label IS NULL".'),
    ] = None,
    message: Annotated[str, typer.Option("--message", "-m", help="Commit message.")] = "",
) -> None:
    """Change items, info, schema, or tags, or revert to a version.

    \b
    fieldbook data update sales --where "region = 'EU'" --set vat=0.2
                                                     change rows
    fieldbook data update cats a.jpg --file new.jpg  replace content (new version)
    fieldbook data update cats a.jpg --set label=tabby
                                                     change metadata
    fieldbook data update cats --from labels.csv     metadata per item from a CSV
    fieldbook data update cats --drop label          remove a column
    fieldbook data update cats --move unsorted/ train/
                                                     rename items
    fieldbook data update cats --schema schema.yaml  check this and later versions
    fieldbook data update cats --info license=MIT    change dataset info
    fieldbook data update cats --tag best [--version 3]
                                                     create or move a tag
    fieldbook data update cats --revert 3            new version identical to v3

    PATH, --from, --drop, --move, --info, and --schema can be combined in one version.
    Metadata not set is kept.
    """
    edits = (
        path is not None
        or where is not None
        or bool(info)
        or values_file is not None
        or bool(drop)
        or move is not None
        or schema_file is not None
    )
    actions = [edits, tag is not None, revert is not None]
    if sum(actions) != 1:
        raise fail(
            "specify one of: PATH (--file, --set), --from, --drop, --move, --info, --schema; "
            "--tag; --revert"
        )
    if revert is not None:
        if version is not None:
            raise fail("--version applies to --tag")
        try:
            new_version = data.dataset(name).revert(_parse_ref(revert), message or None)
        except data.DatasetError as e:
            raise fail(str(e)) from None
        typer.echo(f"{name}: v{new_version} = {revert}")
        return
    if tag is not None:
        if fields or file is not None:
            raise fail("--file and --set apply to PATH")
        try:
            data.dataset(name).tag(tag, version, move=True)
        except (data.DatasetError, ValueError) as e:
            raise fail(str(e)) from None
        typer.echo(f"{name}: tag {tag} -> v{data.dataset(name, tag).version}")
        return
    if version is not None:
        raise fail("--version applies to --tag")
    if path is not None and not fields and file is None:
        raise fail("specify --file and/or --set KEY=VALUE for PATH")
    if path is None and (file is not None or (fields and where is None)):
        raise fail("--file and --set apply to PATH (or --set to the rows of --where)")
    if where is not None and (path is not None or not fields):
        raise fail("--where goes with --set KEY=VALUE, without PATH")
    schema_path = Path(schema_file) if schema_file else None
    for f in (file, values_file, schema_path):
        if f is not None and not f.is_file():
            raise fail(f"{f} is not a file")
    try:
        values = dict(_parse_field(f) for f in fields or [])
        info_values = _parse_info(info)
        per_item = _read_values(values_file) if values_file is not None else {}
        ds = data.dataset(name)
        if not ds.exists:
            raise data.DatasetNotFoundError(f"dataset {ds.key} not found")
        with ds.edit(message) as edit:
            if path is not None:
                edit.update(path, file, **values)
            if where is not None:
                edit.update_records(where, **values)
            for item, item_values in per_item.items():
                if item_values:
                    edit.update(item, **item_values)
            moved = edit.move(*move) if move is not None else 0
            edit.drop_columns(*(drop or []))
            edit.set_info(**info_values)
            if schema_file is not None:
                edit.set_schema(schema_path.read_text() if schema_path else None)
    except (data.DatasetError, ValueError) as e:
        raise fail(str(e)) from None
    changed = [
        *([path] if path is not None else []),
        *([f"rows where {where}"] if where is not None else []),
        *([f"{len(per_item)} items"] if per_item else []),
        *([f"dropped {', '.join(drop)}"] if drop else []),
        *([f"moved {moved} items"] if move is not None else []),
        *(["info"] if info_values else []),
        *([] if schema_file is None else ["schema" if schema_file else "removed schema"]),
    ]
    typer.echo(f"{name}: updated {'; '.join(changed)}, v{edit.version}")


@app.command("rm")
def data_rm(
    spec: Annotated[
        str | None,
        typer.Argument(metavar="[NAME[@VERSION[,VERSION...]]]", show_default=False),
    ] = None,
    paths: Annotated[list[str] | None, typer.Argument(metavar="[PATH]...")] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="Delete the whole dataset with all versions and tags.")
    ] = False,
    old: Annotated[
        bool,
        typer.Option(
            "--old",
            help="Delete versions not kept by the retention rules (tagged, used by runs, "
            "latest --keep, last --days). Without NAME: every dataset, plus unused files.",
        ),
    ] = False,
    keep: Annotated[
        int | None,
        typer.Option(
            "--keep",
            min=1,
            help=f"With --old: keep the latest N versions (default: {cleanup.KEEP_LATEST}).",
        ),
    ] = None,
    days: Annotated[
        float | None,
        typer.Option(
            "--days",
            min=0,
            help="With --old: keep versions from the last D days "
            f"(default: {cleanup.RETENTION.days}).",
        ),
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Remove a tag.")] = None,
    where: Annotated[
        str | None,
        typer.Option(help="Records: remove the rows matching this SQL filter (new version)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be deleted and freed.")
    ] = False,
    now: Annotated[
        bool,
        typer.Option(
            "--now",
            help="Free files changed in the last 24 hours too. Only safe when no other "
            "edit is running.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
    message: Annotated[str, typer.Option("--message", "-m", help="Commit message.")] = "",
) -> None:
    """Remove items, versions, tags, or whole datasets.

    \b
    fieldbook data rm cats a.jpg b.jpg
                                       remove items (a new version)
    fieldbook data rm sales --where "amount IS NULL"
                                       remove rows (records)
    fieldbook data rm cats@3           delete version 3
    fieldbook data rm cats --tag best  remove a tag
    fieldbook data rm cats --all       delete the dataset
    fieldbook data rm cats --old       delete versions not kept by retention rules
    fieldbook data rm cats --old --keep 3 --days 1
                                       with stricter rules
    fieldbook data rm --old            the same for every dataset

    Stored files are freed when the last version that uses them is deleted.
    """
    paths = paths or []
    name, versions = _split_versions(spec) if spec else (None, [])
    actions = [bool(paths), bool(versions), all_, old, tag is not None, where is not None]
    if sum(actions) != 1:
        raise fail("specify exactly one of: PATH..., --where, @VERSION, --all, --old, --tag")
    if name is None and not old:
        raise fail("NAME is required")
    if (paths or tag is not None or where is not None) and (dry_run or now):
        raise fail("--dry-run and --now apply to versions, --old, and --all")
    if (keep is not None or days is not None) and not old:
        raise fail("--keep and --days apply to --old")
    retention = {
        "keep_latest": keep if keep is not None else cleanup.KEEP_LATEST,
        "retention": timedelta(days=days) if days is not None else cleanup.RETENTION,
    }
    grace = timedelta(0) if now else cleanup.GRACE
    try:
        if paths:
            with data.dataset(name).edit(message) as edit:
                for path in paths:
                    edit.remove(path)
            typer.echo(f"{name}: removed {len(paths)} items, v{edit.version}")
            return
        if where is not None:
            with data.dataset(name).edit(message) as edit:
                edit.remove_records(where)
            typer.echo(f"{name}: removed rows where {where}, v{edit.version}")
            return
        if tag is not None:
            data.dataset(name).untag(tag)
            typer.echo(f"{name}: removed tag {tag}")
            return
        if all_:
            if not (yes or dry_run):
                typer.confirm(f"Delete dataset {name} and all its versions?", abort=True)
            report = data.dataset(name).delete(grace=grace, dry_run=dry_run)
        elif versions:
            report = data.dataset(name).delete_versions(versions, grace=grace, dry_run=dry_run)
        elif name is not None:
            report = data.dataset(name).prune(grace=grace, dry_run=dry_run, **retention)
        else:
            report = data.prune(grace=grace, dry_run=dry_run, **retention)
    except data.DatasetError as e:
        raise fail(str(e)) from None
    typer.echo(_describe(report))


def _split_versions(spec: str) -> tuple[str, list[int]]:
    name, _, ref = spec.partition("@")
    if not ref:
        return name, []
    try:
        return name, [int(v) for v in ref.split(",")]
    except ValueError:
        raise fail(f"expected version numbers after @, got {ref!r}") from None


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:,.0f} B" if unit == "B" else f"{count:,.1f} {unit}"
        count /= 1024
    return ""


def _describe(report: data.CleanupReport) -> str:
    verb = "would delete" if report.dry_run else "deleted"
    free = "would free" if report.dry_run else "freed"
    parts = []
    if report.datasets:
        parts.append(f"{verb} dataset {', '.join(report.datasets)}")
    count = sum(len(v) for v in report.versions.values())
    if count or not report.datasets:
        listed = "; ".join(
            f"{key} v{', v'.join(map(str, vs))}" for key, vs in report.versions.items()
        )
        parts.append(f"{verb} {count} versions" + (f" ({listed})" if listed else ""))
    if report.dry_run:
        parts.append("space is freed with the versions that go")
    else:
        parts.append(f"{free} {_size(report.freed)}")
    if report.compacted:
        parts.append(f"compacted {', '.join(report.compacted)}")
    kept: list[str] = []
    runs_kept = sum(len(v) for v in report.kept_for_runs.values())
    if runs_kept:
        kept.append(f"{runs_kept} versions used by runs")
    if kept:
        parts.append(f"kept {', '.join(kept)}")
    if report.scratch:
        parts.append(f"{verb} {len(report.scratch)} smoke-run scratch areas")
    if report.code_snapshots or report.package_lists:
        parts.append(
            f"{verb} {len(report.code_snapshots)} code snapshots and "
            f"{len(report.package_lists)} package lists no longer used"
        )
    return "; ".join(parts)
