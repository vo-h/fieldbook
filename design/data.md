# Datasets

Status: Built; checked on local disk and a real AWS S3 bucket (§13) · Last updated: 2026-10-12

Versioned datasets: how they are stored, read, changed, and deleted. Code: `fieldbook/data/`. Where they are stored is in [storage.md](storage.md); how runs record the versions they use is in §11 and [runs.md](runs.md).

## 1. Model

**A dataset is one Lance table, and nothing else.** There is no separate store of files. What a row is depends on the dataset's kind:

| Kind | A row is | Columns | Read with |
|---|---|---|---|
| Files | One file (an image, an audio clip, a checkpoint) | The table below | `files()`, `open(path)`, `read_many(paths)`, `checkout(dir)` |
| Records | One record of tabular data | The data's own columns | `to_table()`, `fieldbook data query` |

Columns of a files dataset (its **index** is these columns without `data`; "index" never means a Lance search index):

| Column | Type | Notes |
|---|---|---|
| `path` | string | Logical path within the dataset, e.g. `train/001.jpg`. Unique within a version: the key. |
| `hash` | string | SHA-256 of the content (hex). For verification and for comparing versions. |
| `size` | int64 | Bytes. |
| `mime_type` | string | Detected at add time. |
| `added_at` | timestamp | When the item was added. |
| *user columns* | any | Metadata and labels (`width`, `duration`, `label`, ...). Optionally declared by a schema (§9). |
| `data` | blob | The file's bytes, in Lance's blob encoding (file format 2.2). Lance places each file by size: under about 64 KB in the main data file, up to about 4 MB packed into shared files of up to 1 GB, larger ones in a file of their own. |

**What decides the kind:** what is first added. A dataset keeps its kind.

| First added | The dataset holds |
|---|---|
| A table, or a folder of nothing but tables | Records: the rows are imported |
| Anything else, including a folder that mixes images with a table of labels | Files: every file is stored as a file, the table included |

- **A table** is Parquet (`.parquet`), JSON lines (`.json`, `.jsonl`, `.ndjson`), or delimited text of any delimiter: `.csv` and `.tsv` by extension, the delimiter found from the content (`,` tab `;` `|`). `--delimiter` states it, and then a file of any extension (`.txt`, `.dat`) is read as delimited text.
- The kind is in the table's metadata (`fieldbook.kind`); a table without the mark is told by its columns (files have a `data` column), so that one odd table cannot break the listing of all datasets. File operations on a records dataset, and row operations on a files dataset, are refused with a message naming the right call.
- Labels that arrive as a table next to the images become metadata by a deliberate step (`update --from labels.csv`), not by the import.
- An imported table is its rows: the data is the same after export (`COPY ... TO`), the original file's bytes are not kept.
- Where data came from is not a column: the version's record holds the command that added it and, for a bucket, its address (§11).

Metadata extraction (resolution, page count, ...) is the user's code; the data module stores what it is given. File integrity checks (decodable, not truncated) are out of scope.

## 2. Versions and tags

- Every committed change is a new Lance version of the table. Versions share the data they do not change.
- **Version numbers increase but may skip.** Lance numbers every commit, and some commits are not versions of the dataset: a step of an edit that also adds or drops a column or changes info or the schema, and Lance's own housekeeping. These are hidden: not listed, not opened, tagged, or reverted to, and removed by cleanup. A version of the dataset is a commit that carries fieldbook's record.
- Tags give versions readable names (`raw`, `clean-v2`). Tags are Lance tags. Names use letters, digits, `.`, `_`, `-` and cannot be all digits, so `name@3` always means a version number. `add --tag` tags the version the add creates (the tag must be new); tags on versions that already exist are created or moved with `update --tag`.
- Each version's record is the commit's transaction properties: the message, `author`, `agent`, `inputs`, `command`, and, when saved in a launched run, `run_id`, `code`, `git`, and `env` (§11).
- Branching and merging are not supported.

## 3. Python API

```python
import fieldbook as fb

ds = fb.dataset("cats")                          # opens or creates in the current project
raw = fb.dataset("raw", version=3)

# --- a files dataset ---
with ds.edit(message="add batch 3", inputs=["hf://datasets/x"]) as edit:  # inputs: §11
    edit.add("train/003.jpg", "/data/batch3/003.jpg", label="tabby")
    edit.add_dir("/data/batch3", prefix="train/")    # read in parallel (workers=8)
    edit.add_dir("s3://open-data/cats", prefix="raw/")  # from an S3 bucket (§5)
    edit.set_info(license="CC-BY-4.0", source="https://example.org")  # None removes a key
    edit.remove("train/001.jpg")
    edit.update("train/002.jpg", label="siamese")
    edit.update("train/004.jpg", "/data/fixed/004.jpg")  # replace content; metadata kept
    edit.drop_columns("score")                     # remove user columns; earlier versions keep them
    edit.copy_from(raw, "label = 'cat'", prefix="cats/")  # copy raw@3's items (§5)
    edit.move("unsorted/", "train/")                # rename items or directories
    edit.set_schema(open("schema.yaml").read())    # Pandera schema (YAML or DataFrameSchema); None removes
print(edit.version)                                # new version number

view = fb.dataset("cats", version="clean-v2")    # tag or version number
view.kind                                          # "files" or "records"
view.files(prefix="val/", where="label = 'cat'")   # matching index rows (Arrow table)
view.stats(depth=1)                                # files and bytes per top-level dir (train/val/test)
view.stats(prefix="train/", by=["label"])          # files and bytes per label within train
view.open("train/002.jpg")                         # file-like object: one file
view.read_many(["train/002.jpg", "train/007.jpg"]) # bytes of several, fetched together (§10)
view.checkout("/tmp/cats", prefix="train/")        # write the files out, for tools that need files
view.diff(3)                                       # changes from v3 to this version

# --- a records dataset ---
sales = fb.dataset("sales")
with sales.edit("june") as edit:
    edit.remove_records("amount IS NULL")          # by SQL filter
    edit.update_records("region = 'EU'", vat=0.2)  # set columns in the matching rows
    edit.add_records("june.parquet")               # or .csv/.tsv/.jsonl, a folder, s3://...
    edit.add_records("export.txt", delimiter="|")
    edit.add_records(df)                           # a pandas DataFrame, pyarrow Table, list of dicts
print(edit.rows_added)
sales.to_table("amount > 10", columns=["id", "amount"], limit=100)   # Arrow table

# --- both kinds ---
ds.tag("clean-v2", version=edit.version)
ds.tag("clean-v2", version=7, move=True)        # repoint an existing tag
ds.untag("clean-v2")
ds.revert(3)                                       # new version identical to v3 (or a tag)
view.index()                                       # the Lance table at this version
view.info()                                        # dataset info at this version
view.schema()                                      # dataset schema (YAML) at this version, or None
ds.verify(deep=True)                               # check stored data of every version
ds.log()                                           # version history, with inputs and command
view.lineage()                                     # this version's inputs, recursively (§11)
ds.delete_versions([3, 4])                         # delete specific versions
ds.prune()                                         # delete versions not kept by retention rules
ds.delete()                                        # delete the dataset with all versions and tags
fb.data.prune()                                   # prune every dataset
```

Inside a run, opening a dataset version records it as a run input (§11).

## 4. CLI

`fieldbook data` works on the current project's datasets (the `project` setting). Project names follow the same rule as dataset names (letters, digits, `.`, `_`, `-`; starting with a letter or digit), since both are folder names under the storage root.

```
fieldbook data ls                                                  # datasets: kind, version, items or rows, size
fieldbook data ls <name>[@<v|tag>] [--prefix P] [--where SQL] [-l] [--limit N]   # items; for records, rows (20 unless --limit)
fieldbook data ls <name>[@<v|tag>] --stats [--prefix P] [--where SQL] [--depth N] [--by COL]...   # records: row count and columns
fieldbook data ls <name> --versions                                # version history
fieldbook data ls <name> --tags                                    # tags
fieldbook data ls <name>[@<v|tag>] --diff <v|tag> [--prefix P] [--limit N]   # changes from <v>; records: rows removed (-) and added (+)
fieldbook data ls <name>[@<v|tag>] --info                          # dataset info
fieldbook data ls <name>[@<v|tag>] --schema                        # dataset schema
fieldbook data ls <name>[@<v|tag>] --lineage                       # inputs, recursively
fieldbook data ls [<name>] --verify [--deep]                       # check stored data (exit 1 on problems)
fieldbook data cat <name>[@<v|tag>] <path> > out                   # item content to stdout
fieldbook data checkout <name>[@<v|tag>] <dir> [--prefix P]        # write the files into a folder
fieldbook data query -d [<alias>=]<name>[@<v|tag>]... -q SQL [--limit N]   # SQL over rows, or over a files index
fieldbook data add <name> <src>... [--prefix P] [--set KEY=VALUE]... [--hidden] [-m MSG]   # add items (new paths only)
fieldbook data add <name> <table>... [--delimiter D]                        # import rows (§1 decides which)
fieldbook data add <name> <src>... --tag <tag>                              # tag the version this add creates
fieldbook data add <name> <src>... --info KEY=VALUE...                      # set dataset info
fieldbook data add <name> --copy <src>[@<v|tag>] [--where SQL] [--prefix P]  # copy another version's items
fieldbook data add <name> s3://<bucket>/<prefix> [--prefix P]              # read from an S3 bucket (§5)
fieldbook data rm <name> <path>... [-m MSG]                # remove items (new version; history unchanged)
fieldbook data rm <name> --where SQL [-m MSG]              # remove rows of a records dataset
fieldbook data rm <name>@<v>[,<v>...] [--dry-run] [--now]  # delete versions
fieldbook data rm <name> --tag <tag>                       # remove a tag
fieldbook data rm <name> --all [--yes] [--dry-run] [--now] # delete the dataset
fieldbook data rm [<name>] --old [--keep N] [--days D] [--dry-run] [--now]   # delete versions not kept by retention rules
fieldbook data update <name> <path> [--file F] [--set KEY=VALUE]... [-m MSG]   # content and/or metadata
fieldbook data update <name> --where SQL --set KEY=VALUE...        # set columns in matching rows (records)
fieldbook data update <name> --from <file.csv>                     # metadata per item (path column)
fieldbook data update <name> --drop <col>...                       # remove user columns
fieldbook data update <name> --move <old> <new>                    # rename an item or directory (ends in /)
fieldbook data update <name> --schema <file.yaml>                  # set the schema; '' removes it
fieldbook data update <name> [<path> ...] --info KEY=VALUE...      # dataset info; empty VALUE removes
fieldbook data update <name> --tag <tag> [--version V]            # create or move a tag
fieldbook data update <name> --revert <v|tag> [-m MSG]           # new version identical to <v>
```

## 5. Write semantics

Files datasets:

- **Add creates new items only.** Adding a path that already exists fails; existing items are changed with `update`.
- **Update changes content, metadata, or both, as a new version.** New content is a new row (the old one is marked removed); earlier versions keep the old content. Metadata not set is kept, as is `added_at`.
- **Hidden files are skipped.** When adding a directory, files and directories whose names start with `.` (e.g. `.DS_Store`, `.git/`) are skipped unless `--hidden` (`include_hidden=True`) is given.
- **Metadata is user columns.** Users compute metadata themselves and set it with `add --set` (same value for every added file), `update --set` (one item), or `update --from CSV` (per item: a `path` column plus one column per field; empty cells are skipped, `null` clears a value). Values are parsed as JSON where possible, otherwise kept as strings. `--from`, `--drop`, `--info`, and a single PATH can be combined in one version.
- **Moving keeps the stored file.** `move` renames items or directories by rewriting the `path` column, keeping content, metadata, and `added_at`; items already changed in the same edit cannot be moved. Existing paths are refused.
- **Copying between datasets copies the bytes.** `copy_from` adds items of another dataset version (filtered by `where`, placed under `prefix`) with their metadata and a new `added_at`, reading the source's files when the edit is saved (their hash and size are known, so they are not hashed again). The source version is recorded as an input. A dataset of another project is refused.
- **Content is not shared by hash.** The same file under two paths, or removed and later added again, is stored twice. Unchanged files are shared across versions, and a revert copies nothing.
- **Adding from an S3 bucket.** `add_dir` and `fieldbook data add` take `s3://bucket/prefix` (or one object): its objects are read into the table, hidden names skipped as for directories, and the address is recorded as an input of the version (§11). The bytes pass through the machine doing the add, whatever the storage root. Anonymous if the machine has no AWS login; requester-pays buckets and other providers are not supported. The CLI says which way the bytes go, and warns when the two buckets are in different regions (AWS may charge for the move).
- **Edits stage changes in one map** from item path to its pending state: the new row, or a tombstone if removed (the staging model of lakeFS and the git index). Each operation reads an item's current state (pending, else saved) and writes its new state, so operations compose: an item removed or moved away cannot then be updated, a move carries earlier changes, and remove + add replaces an item. Missing items are reported when the operation is called. Changes of metadata to items the edit does not otherwise touch are kept as the changed fields only, so a relabel of a hundred thousand items reads none of their rows.

Records datasets:

- **Rows are appended.** `add_records` takes files, folders, an `s3://` address (recorded as an input, §11), or data in memory. Columns are reconciled by name across the sources and with the table: a column the table lacks is added (null in earlier rows), a column a source lacks is null, and types are widened where Arrow allows it (int and float to float). A value that does not fit the table's column type is refused, naming the column.
- **Rows are removed and changed by SQL filter** (`remove_records(where)`, `update_records(where, col=value)`), since rows have no key of their own.
- **Operations apply in the order written, each seeing what the earlier ones left,** as statements in a transaction do: a filter reaches rows the edit added and rows it already changed, and the whole edit is still one version. Lance's own `delete` and `update` each make a commit, so an edit is not a sequence of them (§6).
- **No uniqueness is enforced.** Importing the same file twice stores its rows twice; a schema's `unique` check (§9) covers the rows of one save.

Both kinds:

- **Dropping columns.** `drop_columns` / `--drop` removes user columns from the new version; earlier versions keep them. A column cannot be set and dropped in the same edit. An edit that drops columns conflicts with any concurrent commit instead of being redone.
- **Column types are fixed by first use.** Values set later are converted to the column's type when that is lossless (`5` into a text column becomes `"5"`, `1` into a float column `1.0`); otherwise the edit fails with an error naming the column (`1.5` into a whole-number column is refused, not cut down).
- **Edits are single-use.** A committed edit cannot be changed or committed again.
- **Concurrency.** Several processes and people may edit the same dataset at once; edits to different items all land, and edits to the same items fail with a conflict error (§6).

## 6. Saving a version

The rule that keeps edits cheap: **a file's bytes are never rewritten unless the bytes changed.**

| Change | How it is written | Not the obvious way, because |
|---|---|---|
| New files or rows | Appended as new data files (`write_fragments`) | A merge-insert of file data held the incoming files in memory and ran out of it (Lance 12, with an index on the key) |
| Removed | Rows marked deleted (`fragment.delete_rows`, or `fragment.delete(filter)` for records) | Space comes back with cleanup (§7) |
| Metadata, a rename | The changed columns are written anew for the data files that hold those rows (`fragment.update_columns`) | Lance's `update`, and a merge-insert of only the changed columns, store the changed rows' files again (measured on 12 and 13) |
| Replaced content | The row is marked deleted and a new one appended | |

- **One edit, one version.** The pieces above are prepared without committing and committed together as one Lance `Update` operation that carries the edit's record. Checked: 500 relabels, 200 removals, and 300 new files became one version; storage grew by the new files only; the earlier version stayed whole.
- **Steps that take a commit of their own:** adding a column (unless new column values are all the edit changes, the usual first relabel, which is one commit), dropping a column, and changing info or the schema. Lance's high-level calls are used for these, because an operation built from an Arrow schema renumbers the columns and reads then return other columns' data. Such a step is marked (`fieldbook.internal`) or carries no record, and is not a version (§2). It cannot be told which version it follows, so the save checks that nothing was committed since it read the table.
- **The first save** writes every staged row and commits them as version 1.
- **Files are read a batch at a time** when saving (2,000 files or 256 MB, whichever comes first), hashed as they are read, by 8 threads; a million files never sit in memory together.
- **A local file over 64 MB is not read into memory.** It is hashed a piece at a time, and Lance is given its address and copies it in itself (`external_blob_mode="ingest"`); the dataset then has its own copy. A 2 GB file: 0.17 GB of memory at its peak, against about 4 GB read whole. Smaller files stay as they were, since for them this is slower (20,000 files of 4 KB: 2.1 to 2.5 s against 1.4 s). Not yet for a large file from a bucket or copied from another dataset: those are read whole.
- **A records edit whose order matters** (anything but removals, then at most one update, then additions) is worked out in a working Lance table on local disk, so that Lance evaluates its filters there as it does on the dataset: added rows go into it, and a saved row is read into it when an update first matches it and is matched there from then on. What comes out is the same one commit: changed columns for the saved rows that remain, deleted marks, and the rows to append. The rows such an edit adds or changes are therefore written twice, once to local disk; an edit in the simple order streams its additions straight to the dataset.
- **Concurrent saves.** Lance accepts two concurrent `Update` commits unless both name the same data file as updated, and has no unique key, so two adds of the same path would both land. Every save therefore names the table's first data file as updated: Lance refuses the later of two concurrent saves, and fieldbook decides at path level. If no item the edit changes (and no info it sets) was changed meanwhile, it is redone on the latest version, reusing the data files it already wrote; otherwise `ConflictError` is raised. A records edit is always redone, its filters applied to the latest rows. Edits that drop columns or change the schema conflict with any concurrent save. Two processes creating the same dataset at once: the first save is committed so that it is refused if the dataset exists by then, and the second is redone on it. Lance's own `write_dataset` is not used for it: it commits the second create as an overwrite of the first, whose rows are then gone (seen with four processes: 2 rounds of 40; with the refusal, none of 60).
- **No index on `path`.** At a million rows a scan finds paths as fast, and the index is what made merge-insert run out of memory.
- **`files()` sorts by path**, since saves append rows and the stored order is not path order.

## 7. Deleting versions and freeing storage

**Space is freed by Lance when versions are deleted.** Removing items or rows creates a new version and changes no history, so it frees nothing by itself. Versions are deleted in three ways:

| Operation | Deletes |
|---|---|
| Delete versions (`rm <name>@<v>`) | The listed versions |
| Delete the dataset (`rm <name> --all`) | Every version and tag of the dataset: its whole table |
| Prune (`rm [<name>] --old`) | Versions not kept by the retention rules |

A version is **kept by the retention rules** if any of these hold: it has a tag; it is one of the latest N versions of its dataset (`--keep`, default 10); it was created within the retention window (`--days`, default 7); or it is recorded as an input by a run in the tracking module. Cleanup runs only when invoked, so how much history is kept is the user's choice; the settings apply to one command and are not saved.

Explicit version deletion refuses the latest version (make a new edit instead), tagged versions (remove the tag first), and versions used by runs. Deleting a dataset is refused if a run used any of its versions.

**Versions used by runs** are those listed in the `fieldbook.inputs` tag of an active MLflow run (tracking.md §3), read directly from the local SQLite store, or through the MLflow client when `MLFLOW_TRACKING_URI` points elsewhere. Deleting the run in MLflow releases its versions. If run records cannot be read, nothing is deleted. Prune reports the versions it kept only because runs used them.

How the bytes come back:

- **Deleting a version** (`cleanup_old_versions`) removes the data files only that version used, and with it the hidden steps of edits (§2).
- **A data file that still holds kept rows** stays whole, removed rows included. Prune therefore **compacts** a table when at least 20% of its rows are marked removed or it has 100 data files or more: the data files that hold removed rows (and, among many, the small ones) are rewritten without them. For files this is fieldbook's own: the kept rows are read in batches (`read_blobs`), written as new data files, and swapped for the old in one commit that is the new version ("compacted: ..."); a save that lands meanwhile makes the commit fail, and the next cleanup does it. Lance's own compaction fetches a blob column one file a request (670 files of 4 KB in a bucket: 46 to 51 s, against 1.4 s for one batched read of them; its thread, batch, and binary-copy settings made no difference), so it is used only for records, which have no such column, and for data files holding a file over 64 MB; it leaves no record, so the table is then recorded as a version. Either way the versions before it expire by the same rules, and the space is freed when they do. Storage roughly doubles while a compaction runs.
- **Data written by an edit that has not committed** (one still running, or one that failed) is left alone for a week, by Lance's own rule; `--now` frees it too, and is only safe when no other edit is running.
- `--dry-run` reports the versions that would be deleted; the size freed is known only by doing it.

Pruning every dataset also deletes code snapshots and package lists nothing refers to any more (runs.md §4) and stale smoke-run scratch areas (runs.md §9). Cleanup covers the current project only: each project keeps its own files (storage.md §1).

## 8. Info, history, revert, diff, and verify

- **Info:** free-form key-value strings at the dataset level (`description`, `license`, `source`, ...). Stored in the table's schema metadata, so info is versioned with the data and carried forward by every edit. Concurrent edits that change the same info key conflict.
- **History:** `log()` and `fieldbook data ls NAME --versions` read one commit record per Lance version and list those that are versions of the dataset (§2). In a bucket each is a round trip, so they are read by up to 16 threads, each with its own Lance handle (a handle cannot be shared between threads); on local disk, in order from one handle. `fieldbook data ls` with no name reads its datasets in parallel.
- **Revert:** a new version pointing at the old version's data (Lance's `Restore`), including its info and schema; nothing is copied. History is unchanged. The target version must still exist.
- **Diff:** for files, compares two versions by path. An item is `added`, `removed`, or `modified`; for modified items, `changed` lists `content` (hash differs) and/or the changed metadata columns. `added_at` is ignored. For records it reads as `diff` of two files does: `-` rows that are only in the old version, `+` rows only in the new one, a changed row as one of each, then the totals. Rows have no key and no order, so whole rows are compared, each as many times as it occurs, sorted so that a changed row's two lines meet; a column only one version has is null in the other. The first 20 differing rows are printed (`--limit`). Both versions are read in full, by DuckDB (`fieldbook[sql]`).
- **Verify:** reads every version. For files, each stored file's size is checked against its row, and with `--deep` its SHA-256; a row that versions share is checked once. For records, each version is checked to read (`--deep` reads every row). A version whose data files are gone is reported as missing.

## 9. Dataset schema

- **Optional.** A dataset without a schema accepts any user columns; their types are fixed by first use (§5).
- **Format.** A Pandera `DataFrameSchema` (pandas API) stored as YAML in the table's schema metadata, so it is versioned with the data and restored by revert. For a files dataset it covers the index columns (metadata), not file contents; for a records dataset, the rows.
- **Checked on every commit**, before anything is committed. If any check fails, `ValidationError` lists every failure (path or row, column, check, value); the CLI prints the first 20. Concurrent edits are checked against the schema of the version they are redone on.
- **What is checked.** Files: the whole new index. Records: the rows a save adds or changes, a batch at a time as they are written (so a table of tens of millions of rows is not loaded to add a few), and every row when the schema itself is set or changed. Checks across rows (`unique`) therefore cover one save's rows on a records dataset.
- **Changing the schema** is an edit like any other and creates a new version of the same dataset. The new version must pass the new schema, so values can be backfilled in the same edit (`update --schema s.yaml --from labels.csv`). Concurrent edits that both change the schema conflict.
- **Only storable checks.** Built-in Pandera checks (`isin`, `in_range`, `ge`, `str_matches`, `unique`, `nullable`, `required`, ...) are allowed. Schemas with custom Python checks, unknown keys, or another dataframe library's API (e.g. Polars) are rejected when set, since Pandera would drop or misread them. Custom checks run in the user's own code.
- **Optional dependency.** Pandera and pandas are installed with `fieldbook[validate]` and imported only when a dataset has a schema.
- **Not covered:** file integrity (decodable, not truncated).

## 10. Reading with other tools

The platform provides no loaders. A file has no address of its own (it is a row of the table), so other tools get the data in one of these ways:

| Need | Use |
|---|---|
| A training loop over files | `read_many(paths)`: one batch, fetched together. One `open()` a file is one request a file, which is slow from a bucket |
| A tool that needs real files | `checkout(dir)` / `fieldbook data checkout`: writes the files out at their paths |
| SQL | `fieldbook data query` (needs `fieldbook[sql]`) |
| Arrow, pandas, Polars | `to_table()` for records, `files()` for a files index; `index()` is the Lance table itself, for Lance's own readers |

**`fieldbook data query`** runs DuckDB SQL. Each `--dataset NAME[@VERSION|TAG]` is a table named after the dataset (or `ALIAS=`): a records dataset's rows, read from its Lance table as the query needs them (only the columns asked for); a files dataset's index. Up to `--limit` rows are printed (default 100); `COPY ... TO` in the query exports results. Repeating `--dataset` with aliases compares versions row by row. Scans over rows were about five times slower than over the Parquet file they came from (0.57 s against 0.10 s for an aggregate over 20 million rows).

```python
# PyTorch: batches fetched together (run on AWS, below; not in the test suite)
class Items(torch.utils.data.Dataset):
    def __init__(self, ds, prefix=""):
        self.ds, self.rows = ds, ds.files(prefix).to_pylist()

    def __len__(self):
        return len(self.rows)

    def __getitems__(self, indices):                 # the DataLoader passes a batch of indices
        rows = [self.rows[i] for i in indices]
        contents = self.ds.read_many([row["path"] for row in rows])
        return [(decode(content), row["label"]) for row, content in zip(rows, contents)]
        # decode: your own bytes-to-tensor function; collate (below): your own batching

loader = DataLoader(Items(fb.dataset("pets", version=3), "train/"), batch_size=64, shuffle=True,
                    num_workers=8, multiprocessing_context="spawn", persistent_workers=True,
                    collate_fn=collate)
```

- **Start the workers with `spawn`** (or `forkserver`): Lance is not safe to fork, and Linux forks by default. A `Dataset` view pickles, so each worker gets its own.
- **Workers take a while to start** (about 1.5 s each, one after another, most of it importing torch): keep them (`persistent_workers`).
- **A view can be shared by threads** for `read_many`.

Pin a version so reads are consistent. **Measured on AWS** (2026-10-12): one `m6i.xlarge` (4 CPUs) in the bucket's region, shuffled batches of 64, each file checked against its hash in the first batch:

| | 200 KB files (5,000) | 4 KB files (50,000) |
|---|---|---|
| `read_many`, one batch at a time | 217 a second (43 MB a second) | 706 a second |
| `read_many`, 8 threads | 1,511 a second (302 MB a second) | 4,398 a second |
| `DataLoader`, no workers | 213 a second | 764 a second |
| `DataLoader`, 4 workers | 585 a second (117 MB a second) | 1,506 a second |
| `DataLoader`, 8 workers | 1,613 a second (323 MB a second) | 2,498 a second |
| Adding the files, from the machine | 1,357 a second (271 MB a second) | 5,102 a second |

Reads scale with how many batches are in flight, not with CPUs: use more workers than cores. From a laptop to a bucket, 388 a second: train from inside the region. When a loader with persistent workers that return tensors is dropped before its epoch ends, some workers die at shutdown on Linux (`terminate called without an active exception`). That is PyTorch's: a loader with no dataset at all, and no fieldbook or Lance imported, does the same; every batch read is right.

**Point-in-time joins** give leakage-free training data: each example gets the latest feature value from *before* its timestamp, never a later one. With examples and feature values as records datasets:

```
fieldbook data query -d examples@3 -d features@7 -q "
  SELECT e.*, f.purchases_30d
  FROM examples e ASOF LEFT JOIN features f
    ON e.user_id = f.user_id AND e.ts >= f.ts"
```

Results are saved as a new version by writing them to a file (`COPY ... TO 'out.parquet'`) and adding it.

In a smoke run (`fieldbook run --smoke`), reads return a sample (N items per top-level folder for files; the first N × 100 rows for records), versions are saved to a scratch area instead of the dataset, and tags and deletions are skipped (runs.md §9). The scratch area is a zero-copy clone of the table (Lance's `shallow_clone`).

## 11. Lineage

Each version records where it came from, in its transaction properties:

- **`inputs`:** dataset versions read while the edit was open (from `ds.edit()` to the version being saved), through `files`, `stats`, `open`, `read_many`, `checkout`, `to_table`, or `fieldbook data query`. Reads of the dataset being edited are excluded. Unpinned reads record the version actually read. `edit(inputs=[...])` adds others: datasets of this project, `name@version|tag` strings, or sources outside the platform (`hf://...`, URLs), stored as given; a dataset of another project is refused. Stored as `project/name@version`.
- **`command`:** the process's command line, e.g. `fieldbook data add cats ./imgs` or `clean.py --min-size 10`. Not recorded in notebooks or interactive sessions. Values of flags that look like secrets (`--token`, `--api-key`, ...) are masked (N8).
- **`run_id`, `code`, `git`, `env`:** set when the version is saved in a run started with `fieldbook run`: the platform run ID, the code snapshot hash (restore with `fieldbook log code`), the git commit it was taken on, and the environment record ([runs.md](runs.md) §4–5).

`fieldbook data ls clean@5 --lineage` prints the version and its inputs, recursively. Inputs whose dataset or version was deleted are shown as `(deleted)`; deleting them is allowed. Versions reached twice are expanded once.

Limits: data read before the edit opened, or by another process, is not recorded; pass it with `inputs=`. Lineage covers dataset versions and runs; a model's lineage is its MLflow run (tracking.md §6).

MLflow runs get the dataset versions read and written as tags `fieldbook.inputs` and `fieldbook.outputs` (tracking.md §3).

## 12. Scale and implementation notes

- **Scale, measured on a laptop's disk:** a million files of 4 KB (§14) and three million of 1 KB: adding them took 814 s (3,700 a second), relabelling 10% 9.7 s, removing 20% 9.3 s, an edit of one item 2.7 s, a shuffled read of 20,000 1.4 s, verify 31 s. **Memory is the limit on one add:** that add peaked at 6.9 GB, about 2.3 GB per million files staged in one edit, so larger imports go in several edits. Records to tens of millions of rows (measured at 20 million).
- **An open edit of a files dataset loads the dataset's paths** (to check what exists) and fetches full rows only for the items it moves or replaces. It loads the whole index (without the files) when the dataset has a schema to validate (§9). Looking paths up on demand instead is later work (README.md §9).
- **Large files** (checkpoints, videos) go in the table too; Lance gives each a data file of its own, and a local one is streamed in (§6).
- **Reads:** item lookups (`open`, `read_many`) use a path-to-row table cached per version (0.3 s to build at a million items). Pinned views (`fb.dataset(name, version=...)`) load it once; unpinned views check for a newer version on each read. Training loops should pin a version.
- **Lance version: 13**, pinned. Lance releases a major version about every three weeks; the measurements in §6 and §14 are rerun on upgrades. Its blob layout is new (2026) and has no specification page yet.
- **Lance logging:** Lance's creation warnings are silenced by default (`LANCE_LOG=error`).
- **Lance is imported on first use.** Importing it takes over a second, so `fieldbook` commands and scripts that do not open a dataset skip it: `import fieldbook` takes about 0.3 s.
- **Not supported:** branching and merging (parallel work uses separate datasets or tags); referencing external data in place (Lance can reference outside objects from a blob column; not explored).

## 13. Testing

- Unit tests for every API call and CLI command, for both kinds.
- Concurrency: parallel edits from separate processes, conflicting and non-conflicting; the same new path added at once; a save made while another edit is open.
- Deletion safety: kept, tagged, latest, and run-referenced versions and in-progress writes are never deleted. `tests/data/test_cleanup_invariant.py` runs the most aggressive cleanup after every operation and checks that every remaining version is intact and protected versions survive.
- A bucket: `tests/test_storage_bucket.py` runs saves, reads, cleanup with compaction, and queries against an S3-compatible endpoint.
- **Checked on AWS S3** (2026-10-12, from a laptop; all passed): 2,000 files added (7.5 s); an edit that relabels 200, removes 100, adds, and moves, as one version (6.6 s), with the first version whole; `verify --deep`; prune with compaction (8.4 MB to 4.6 MB; 88 s with Lance's compaction; with fieldbook's own, a 1,000-file dataset went from 55 s to 20 to 31 s, now mostly the upload); 100,000 rows imported (1.5 s); a records edit that adds, updates, and removes in that order, as one version (5.1 s); a diff of records; 4 processes creating the same dataset at the same instant, 3 rounds, no save lost; 6 writers saving 5 times each at once, all 31 items there; 8 edits adding the same new path at the same instant, 3 rounds: one landed, 7 refused.

## 14. Why the files are in the table

Until 2026-10-12 a dataset was an index plus one stored object per file, named by its hash. An object per file is what made datasets of many small files slow and costly in a bucket. Lance was built to hold such data inside the table (an index over packed content, as git's packfiles, restic, and Haystack do). Measured with a prototype against the old code, then decided:

| | Object per file | In the table |
|---|---|---|
| **Bucket, from a laptop, 100,000 files of 4 KB** | | |
| Add all the files | 3,130 s (32 a second) | 42 s (2,365 a second) |
| Objects in the bucket | 100,010 | 9 |
| Relabel | 11 items a second | 4,518 a second |
| Shuffled read, best method each has | 70 a second | 388 a second |
| **Local disk, a million files of 4 KB** | | |
| Add all the files | 983 s | 217 s |
| Relabel 10% | 615 s | 3 s |
| Shuffled read of 20,000 | 7.7 s | 1.0 s |
| **Machine in the bucket's region, shuffled, 200 KB files** | 132 a second (26 MB a second) | 2,556 a second (511 MB a second) |

The implementation that replaced the prototype, local disk, 100,000 files of 4 KB: add 19 s; relabel 10% 0.2 s; remove 20% 0.2 s; an edit that relabels, removes, and adds 0.06 s; shuffled read of 20,000 0.4 s.

Decided (2026-10-12): one model, not a choice per dataset; tables are imported as rows, with no option to keep the file; large files go in the table too; no conversion, since there were no datasets to convert.

What was given up:

| Given up | How much it matters |
|---|---|
| Identical content stored once | A file removed and added again, or under two paths, is stored twice. Unchanged files are still shared across versions. Lance has no un-delete, so taking a removed row back would mean writing its deletion files ourselves: possible, not planned |
| Copying between datasets without copying bytes | Now a real copy |
| Copies inside S3 when adding from a bucket | The bytes pass through the machine |
| Other tools reading the files straight from the bucket | They need `checkout` first |
| The exact bytes of an imported table | The rows are the same after export; the file is not |
| A mature storage layer | Lance's blob layout is new, and its documented size thresholds disagreed with its code |

Seen once while measuring, not reproduced, nothing changed for them: an upload from a laptop failed with Lance's default of 10 parts at once and worked with 3 (`LANCE_UPLOAD_CONCURRENCY`), while the connection was dropping; a read with 64 threads, each with its own Lance handle, timed out fetching an EC2 machine's credentials where 8 and 32 had worked.
