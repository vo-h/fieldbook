"""`fieldbook run --rerun RUN`: run an earlier run again from its record (runs.md §17): the code of
its snapshot, its command, its seed, and the dataset versions it read and started from, in
the project's current environment. It is a run with a new ID. What it logs to MLflow is
real; what it saves to datasets goes to a scratch area, where it is compared with what the
original saved and then deleted, so a rerun never adds versions to a dataset."""

import json
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fieldbook import storage
from fieldbook.code import environments, snapshot
from fieldbook.config import Settings, settings
from fieldbook.history import RunRecord
from fieldbook.launch import LaunchError, local, note
from fieldbook.launch.tasks import Task

IDENTICAL = "identical"


@dataclass
class Result:
    exit_code: int
    run_id: str
    changed: list[str] = field(default_factory=list)  # arguments added to the command
    same: list[str] = field(default_factory=list)  # metrics that came out identical
    differs: dict[str, dict[str, Any]] = field(default_factory=dict)  # metric -> {a, b}
    outputs: dict[str, str] = field(default_factory=dict)  # original version -> how it compares

    @property
    def verdict(self) -> str:
        if self.exit_code != 0:
            return f"failed (exit code {self.exit_code})"
        changes = [f"{k} {_show(v['a'])} -> {_show(v['b'])}" for k, v in self.differs.items()]
        changes += [f"{ref} {how}" for ref, how in self.outputs.items() if how != IDENTICAL]
        if changes:
            return "differs: " + ", ".join(changes)
        held = [f"{len(self.same)} metrics ({', '.join(self.same[:4])})"] if self.same else []
        held += [ref for ref, how in self.outputs.items() if how == IDENTICAL]
        if not held:
            return "ran; nothing to compare (no metrics, no saved versions)"
        return ("nothing changed: " if self.changed else "reproduced: ") + ", ".join(held)


def rerun(
    original: RunRecord,
    extra: list[str],
    *,
    seed: int | None = None,
    cfg: Settings | None = None,
) -> Result:
    """Run `original` again; `extra` arguments are added to its command (one change at a
    time is the point: `fieldbook run --rerun RUN --lr 0.01`)."""
    cfg = cfg or settings()
    if not original.code or not original.command:
        raise LaunchError(
            f"run {original.run_id} has no code snapshot or command on record (a notebook, "
            "or a run started outside a project)"
        )
    argv = shlex.split(original.command)
    if "***" in " ".join(argv):
        raise LaunchError(
            f"run {original.run_id} was given a secret on its command line, which is not "
            "recorded: run its command yourself, with the value"
        )
    record = json.loads(original.env) if original.env else {}
    project = environments.find(Path.cwd())  # the project's environment, not the checkout's
    _say_package_changes(record, project, cfg)
    read, saved = _versions(original.inputs), _versions(original.outputs)
    checkout = Path(tempfile.mkdtemp(prefix="fieldbook-rerun-"))
    try:
        snapshot.checkout(original.code, checkout, cfg)
        workdir = _workdir(checkout, record.get("cwd"), argv[0])
        command = ["python", *argv, *extra] if argv[0].endswith(".py") else [*argv, *extra]
        os.environ |= {
            "FIELDBOOK_CODE_ROOT": str(checkout),
            "FIELDBOOK_GIT": original.git,
            "FIELDBOOK_RERUN_OF": original.run_id,
            "FIELDBOOK_PIN_INPUTS": json.dumps({name: max(v) for name, v in read.items()}),
            # an edit starts from the version before the first one the original saved
            "FIELDBOOK_PIN_BASES": json.dumps({name: min(v) - 1 for name, v in saved.items()}),
        }
        if project.venv is not None:
            os.environ["VIRTUAL_ENV"] = str(project.venv)
        if read:
            pinned = ", ".join(f"{name}@{max(v)}" for name, v in sorted(read.items()))
            note(f"reading the versions the original read: {pinned}")
        launch = local.prepare(
            Task(run=command, workdir=workdir),
            seed=seed if seed is not None else record.get("seed"),
            deterministic=bool(record.get("deterministic")),
            code=original.code,
            rerun=True,
        )
        scratch = launch.context.scratch
        try:
            result = Result(local.execute(launch), launch.context.run_id, changed=extra)
            if result.exit_code == 0:
                result.outputs = _compare_saved(saved, scratch, cfg)
        finally:
            local.finish_scratch(scratch, cfg, keep=False)
    except snapshot.SnapshotError as e:
        raise LaunchError(str(e)) from None
    finally:
        shutil.rmtree(checkout, ignore_errors=True)
    return _compare(original, result, cfg)


def _versions(refs: list[str]) -> dict[str, list[int]]:
    """`{"raw": [3]}` from `["raw@3"]`; sources outside the project are left out."""
    found: dict[str, list[int]] = {}
    for ref in refs:
        if "@" in ref and "://" not in ref:
            name, version = ref.rsplit("@", 1)
            found.setdefault(name.rsplit("/", 1)[-1], []).append(int(version.lstrip("v")))
    return found


def _workdir(checkout: Path, cwd: str | None, script: str) -> Path:
    """The folder of the checkout the command ran in: recorded since this was written; for
    older runs, the folder where its script is found."""
    if cwd is not None and (checkout / cwd).is_dir():
        return checkout / cwd
    if not script.endswith(".py") or (checkout / script).is_file():
        return checkout
    tail = Path(script).parts
    found = [p for p in checkout.rglob(tail[-1]) if p.parts[-len(tail) :] == tail]
    if len(found) != 1:
        raise LaunchError(
            f"cannot tell which folder `{script}` was run from: start it from a checkout "
            "(`fieldbook log code`) yourself"
        )
    return found[0].parents[len(tail) - 1]


def _say_package_changes(record: dict, project: environments.PythonEnv, cfg: Settings) -> None:
    now = environments.query_interpreter(str(project.python)).get("packages")
    then = record.get("packages")
    if not now or not then:
        return
    current = snapshot.store_packages(now, cfg)
    if current != then:
        from fieldbook.history import _packages

        d = _packages(then, current, cfg)
        count = len(d["added"]) + len(d["removed"]) + len(d["changed"])
        note(
            f"warning: {count} installed packages differ from the original run's "
            "(`fieldbook log diff` of the two runs lists them)"
        )


def _compare_saved(saved: dict[str, list[int]], scratch: str, cfg: Settings) -> dict[str, str]:
    """How each dataset the original saved came out this time: the files (path and content
    hash) of its last saved version against those in the rerun's scratch area."""
    from fieldbook.data import smoke
    from fieldbook.data.dataset import Dataset, _open

    def files(index: Any) -> dict[str, str]:
        table = index.to_table(columns=["path", "hash"])
        return dict(zip(table["path"].to_pylist(), table["hash"].to_pylist(), strict=True))

    found = {}
    for name, versions in sorted(saved.items()):
        ref = f"{name}@{max(versions)}"
        then = files(Dataset(name, max(versions), cfg=cfg).index())
        try:
            again = _open(storage.location(smoke.scratch_path(scratch, name), cfg), cfg)
        except (FileNotFoundError, ValueError, OSError):
            found[ref] = "was not saved this time"
            continue
        now = files(again)
        changed = sum(1 for path in then.keys() & now.keys() if then[path] != now[path])
        added, missing = len(now.keys() - then.keys()), len(then.keys() - now.keys())
        if not (changed or added or missing):
            found[ref] = IDENTICAL
        else:
            found[ref] = f"{changed} files changed, {added} added, {missing} missing"
    return found


def _compare(original: RunRecord, result: Result, cfg: Settings) -> Result:
    """Fill in how the new run's metrics compare, and note the outcome on the original
    (where someone asking "does this result hold?" looks) and on the new run if it left a
    record (runs.md §15)."""
    from fieldbook import history, notes

    try:
        new = history.find(result.run_id, cfg)
    except history.HistoryError:
        new = None  # it logged nothing to MLflow
    if new is not None and result.exit_code == 0:
        for name in sorted(set(original.metrics) & set(new.metrics)):
            if original.metrics[name] == new.metrics[name]:
                result.same.append(name)
            else:
                result.differs[name] = {"a": original.metrics[name], "b": new.metrics[name]}
    change = f" with {shlex.join(result.changed)}" if result.changed else ""
    notes.add(original.run_id, f"rerun as {result.run_id}{change}: {result.verdict}", cfg=cfg)
    if new is not None:
        notes.add(new.run_id, f"rerun of {original.run_id}{change}: {result.verdict}", cfg=cfg)
    return result


def _show(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)
