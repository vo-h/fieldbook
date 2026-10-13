"""Version lineage: the inputs and command recorded with each version (data.md §11).

An edit's inputs are the dataset versions read while it is open (between `ds.edit()` and
its commit), plus any passed explicitly. They are stored as `project/name@version`; sources
outside the platform (e.g. `hf://...`) are stored as given.
"""

import functools
import json
import os
import re
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fieldbook import credentials
from fieldbook.launch import agents, context
from fieldbook.tracking import costs

if TYPE_CHECKING:
    from fieldbook.data.dataset import Dataset, Edit, VersionInfo

_REF_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)@(\d+)$")

_open_edits: "weakref.WeakSet[Edit]" = weakref.WeakSet()
_lock = threading.Lock()
# Every dataset version read or written by this process, for MLflow run tags
# (tracking.md §3).
_process_inputs: set[str] = set()
_process_outputs: set[str] = set()


@dataclass(frozen=True)
class LineageNode:
    ref: str  # "project/name@version", or a source outside the platform
    depth: int
    status: str  # "ok", "deleted", "external", or "repeated" (expanded earlier in the walk)
    info: "VersionInfo | None" = None


def version_ref(ds: "Dataset", version: int) -> str:
    """`project/name@version`; versions in a smoke run's scratch area are
    `scratch/<id>/project/name@version`, so they are never taken for real versions."""
    ref = f"{ds.project}/{ds.name}@{version}"
    return f"scratch/{ds._scratch}/{ref}" if ds.in_scratch else ref


def open_edit(edit: "Edit") -> None:
    with _lock:
        _open_edits.add(edit)


def close_edit(edit: "Edit") -> None:
    with _lock:
        _open_edits.discard(edit)


def record_read(ds: "Dataset", version: int) -> None:
    """Record `ds@version` as read by this process, and as an input of every open edit
    of another dataset."""
    ref = version_ref(ds, version)
    if ref not in _process_inputs:
        with _lock:
            _process_inputs.add(ref)
        _sync_mlflow("fieldbook.inputs", _process_inputs)
    if not _open_edits:
        return
    with _lock:
        for edit in _open_edits:
            if edit.ds.key != ds.key:
                edit.inputs.add(ref)


def record_output(ds: "Dataset", version: int) -> None:
    with _lock:
        _process_outputs.add(version_ref(ds, version))
    _sync_mlflow("fieldbook.outputs", _process_outputs)


def _author() -> str | None:
    """`FIELDBOOK_AUTHOR` (set by `fieldbook run` for its process), else the `author` setting."""
    if author := os.environ.get("FIELDBOOK_AUTHOR"):
        return author
    from fieldbook.config import ConfigError, settings

    try:
        return settings().author
    except ConfigError:
        return None


def run_tags() -> dict[str, str]:
    """MLflow tags for runs started in this process (tracking.md §3)."""
    run = context.current()
    values = {"fieldbook.run_id": run.run_id, "fieldbook.code": run.code, "fieldbook.env": run.env,
              "fieldbook.git": run.git}  # fmt: skip
    tags = {tag: value for tag, value in values.items() if value}
    if author := _author():
        tags["fieldbook.author"] = author  # who ran it, for costs by author (sharing.md §4)
    if agent := agents.current():
        tags["fieldbook.agent"] = agent  # the coding agent acting for them (runs.md §20)
    if run.smoke:
        tags["fieldbook.smoke"] = str(run.smoke)
    if original := os.environ.get("FIELDBOOK_RERUN_OF"):
        tags["fieldbook.rerun_of"] = original
    tags.update(costs.machine_tags())  # a cloud run: its machine, for its cost
    if run.run_id and (command := current_command()):
        tags["fieldbook.command"] = command  # with config files in the snapshot, the full config
    with _lock:
        if _process_inputs:
            tags["fieldbook.inputs"] = json.dumps(sorted(_process_inputs))
        if _process_outputs:
            tags["fieldbook.outputs"] = json.dumps(sorted(_process_outputs))
    return tags


def _sync_mlflow(tag: str, refs: set[str]) -> None:
    """Update `tag` on the active MLflow run, if MLflow is in use in this process."""
    mlflow = sys.modules.get("mlflow")
    if mlflow is None or mlflow.active_run() is None:
        return
    with _lock:
        value = json.dumps(sorted(refs))
    mlflow.set_tag(tag, value)


def pinned(name: str, variable: str = "FIELDBOOK_PIN_INPUTS") -> int | None:
    """While a run is repeated on the data the original used (`fieldbook run --rerun`, runs.md §17):
    the version of dataset `name` that reads without a version use (`FIELDBOOK_PIN_INPUTS`), or
    the version its first edit starts from, 0 for an empty dataset (`FIELDBOOK_PIN_BASES`).
    None otherwise."""
    pins = os.environ.get(variable)
    return _pins(pins).get(name) if pins else None


@functools.lru_cache(maxsize=4)
def _pins(value: str) -> dict[str, int]:
    return {str(name): int(version) for name, version in json.loads(value).items()}


def current_command() -> str:
    """The command line of this process, or "" in notebooks and interactive sessions.

    Values of flags that look like secrets (`--token`, `--api-key`, ...) and of the run's
    secrets (`.env`, projects.md §4) are masked.
    """
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return ""
    ipython = sys.modules.get("IPython")
    if ipython is not None and ipython.get_ipython() is not None:
        return ""
    argv = list(sys.argv)
    if not argv or argv[0] in ("", "-c"):
        return ""
    if Path(argv[0]).is_absolute():
        argv[0] = Path(argv[0]).name  # e.g. /.../.venv/bin/fieldbook -> fieldbook
    secrets = [os.environ.get(name, "") for name in context.current().secrets]
    return credentials.mask_command(argv, secrets)


def lineage(ds: "Dataset") -> list[LineageNode]:
    """The version `ds` points at, then its inputs, recursively (depth-first)."""
    from fieldbook.data.dataset import Dataset, DatasetError

    nodes: list[LineageNode] = []
    expanded: set[str] = set()

    def walk(ref: str, depth: int) -> None:
        match = _REF_RE.match(ref)
        if match is None:
            nodes.append(LineageNode(ref, depth, "external"))
            return
        if ref in expanded:
            nodes.append(LineageNode(ref, depth, "repeated"))
            return
        project, name, version = match.group(1), match.group(2), int(match.group(3))
        if project != ds.project:  # recorded before projects were kept apart
            nodes.append(LineageNode(ref, depth, "external"))
            return
        try:
            info = Dataset(name, cfg=ds.cfg)._version_info(version)
        except DatasetError:
            nodes.append(LineageNode(ref, depth, "deleted"))
            return
        expanded.add(ref)
        nodes.append(LineageNode(ref, depth, "ok", info))
        for parent in info.inputs:
            walk(parent, depth + 1)

    walk(version_ref(ds, ds.version), 0)
    return nodes
