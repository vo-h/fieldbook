"""Smoke runs (`fieldbook run --smoke`; runs.md §9): reads return a sample; versions
are saved to a scratch area that is deleted afterwards, not to the dataset; tags and
deletions are skipped. Every check still runs.

A rerun (`fieldbook run --rerun`; runs.md §17) uses the scratch area the same way, without the
sampling: what it saves is compared with the original's versions, never added to them."""

import os
import sys

import pyarrow as pa

from fieldbook.launch import context

SCRATCH_DIR = "scratch"
ROWS_PER_ITEM = 100  # rows of a records dataset read in place of one item of a files one


def scratch_id() -> str | None:
    """The scratch area in use: a smoke run's (`PLATFORM_SCRATCH`), or one kept for
    inspection (`FIELDBOOK_SCRATCH`, read-only)."""
    return context.current().scratch or os.environ.get("FIELDBOOK_SCRATCH") or None


def writes_to_scratch() -> bool:
    """Whether saved versions go to the scratch area (a smoke run with one, or a rerun)."""
    return bool(context.current().scratch)


def scratch_path(scratch: str, name: str) -> str:
    """A dataset's table in a scratch area, relative to the project's folder."""
    return f"{SCRATCH_DIR}/{scratch}/datasets/{name}.lance"


def kind() -> str:
    return "smoke run" if sample_size() is not None else "rerun"


def note(message: str) -> None:
    print(f"{kind()}: {message}", file=sys.stderr, flush=True)


def sample_size() -> int | None:
    """Items per folder to read, or None outside a smoke run."""
    return context.current().smoke


def sample(table: pa.Table, prefix: str, n: int) -> pa.Table:
    """Up to `n` items from each top-level folder under `prefix` (files directly under
    it count as one folder), so small splits such as `val/` are never left empty. The
    items with the lowest hashes are taken: the same items every run, in effect random.
    """
    paths, hashes = table["path"].to_pylist(), table["hash"].to_pylist()
    taken: list[int] = []
    counts: dict[str, int] = {}
    for i in sorted(range(len(paths)), key=lambda i: (hashes[i], paths[i])):
        rest = paths[i][len(prefix) :]
        folder = rest.split("/", 1)[0] if "/" in rest else "."
        if counts.get(folder, 0) < n:
            counts[folder] = counts.get(folder, 0) + 1
            taken.append(i)
    return table.take(pa.array(sorted(taken), pa.int64()))


def skip(action: str) -> bool:
    """In a smoke run or a rerun, say that `action` is not saved and return True."""
    if sample_size() is None and not writes_to_scratch():
        return False
    note(f"{action} (not saved)")
    return True
