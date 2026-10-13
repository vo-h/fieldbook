"""Noticing that a run repeats an earlier one (runs.md §16): the same code, environment,
and command, with the data it read unchanged since.

Nothing is compared file by file. Everything a run depends on is already one value: the
code snapshot's hash covers every file of the repository, the package list has a hash,
and a dataset version is a number that never changes meaning. These are hashed into one
fingerprint, and each run leaves an empty file `fingerprints/<fingerprint>/<run ID>` in
the project's folder of the storage root. The check before a run is one listing of that
folder; only when it finds an earlier run is that run's record read.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fieldbook import storage
from fieldbook.config import Settings

FOLDER = "fingerprints"
FAILED = ("FAILED", "KILLED")


@dataclass(frozen=True)
class Repeat:
    """An earlier run that the one about to start would repeat."""

    run_id: str
    summary: str  # one line: when, who, status, metrics, what it saved, its latest note


def fingerprint(
    *,
    code: str,
    environment: str | None,
    command: list[str] | str,
    workdir: Path,
    repo: Path,
    seed: int | None = None,
    envs: dict[str, str] | None = None,
) -> str:
    """One hash of what decides a run's result, apart from its data: the code snapshot, the
    environment (installed packages' hash; a cloud run's image), the command as typed
    (`python train.py --lr 1` and `fieldbook run train.py --lr 1` are the same), the folder it
    runs in, a task file's variables, and the seed if one was asked for. A run with a seed
    of its own choosing repeats an earlier one whatever seed that had."""
    if isinstance(command, list):
        command = [arg for arg in command if arg][command[:1] == ["python"] :]
        if command and command[0].endswith(".py"):  # the script, however it was pointed at
            script = (workdir / command[0]).resolve()
            if script.is_relative_to(repo):
                command = [script.relative_to(repo).as_posix(), *command[1:]]
    try:
        where = workdir.resolve().relative_to(repo).as_posix()
    except ValueError:
        where = str(workdir)
    parts = [code, environment or "", command, where, seed, sorted((envs or {}).items())]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def mark(value: str, run_id: str, cfg: Settings) -> None:
    """Leave this run's mark under its fingerprint."""
    files = storage.fs(cfg)
    storage.makedirs(files, f"{FOLDER}/{value}")
    files.pipe_file(f"{FOLDER}/{value}/{run_id}", b"")


def earlier(value: str, cfg: Settings) -> list[str]:
    """IDs of the runs with this fingerprint, newest first. One listing."""
    files = storage.fs(cfg)
    try:
        names = files.ls(f"{FOLDER}/{value}", detail=False)
    except FileNotFoundError:
        return []
    return sorted((name.rsplit("/", 1)[-1] for name in names), reverse=True)


def find(value: str, cfg: Settings) -> Repeat | None:
    """The latest earlier run with this fingerprint whose result still stands: it left a
    record, did not fail, and every dataset it read is still at the version it read."""
    run_ids = earlier(value, cfg)
    if not run_ids:
        return None
    from fieldbook import history

    records = {r.run_id: r for r in history.runs(cfg)}
    for run_id in run_ids:
        run = records.get(run_id)
        if run is None or run.status in FAILED:
            continue
        if history.with_details([run], cfg)[0].smoke or _data_changed(run.inputs, cfg):
            continue
        return Repeat(
            run_id, _summary(history.with_notes(history.with_details([run], cfg), cfg)[0])
        )
    return None


def _data_changed(inputs: list[str], cfg: Settings) -> bool:
    from fieldbook.data.dataset import Dataset

    for ref in inputs:
        if "://" in ref or "@" not in ref:
            continue  # a source outside the project: not something to compare
        name, version = ref.rsplit("@", 1)
        try:
            if Dataset(name.rsplit("/", 1)[-1], cfg=cfg).version != int(version.lstrip("v")):
                return True
        except Exception:  # deleted, or unreadable: its result cannot be vouched for
            return True
    return False


def _summary(run) -> str:  # noqa: ANN001 (history.RunRecord)
    when = run.started.astimezone().strftime("%Y-%m-%d %H:%M") if run.started else ""
    parts = [f"run {run.run_id}", when, run.author, (run.status or "").lower()]
    parts.append("  ".join(f"{k}={v:.4g}" for k, v in sorted(run.metrics.items())[:4]))
    if run.outputs:
        parts.append("saved " + ", ".join(run.outputs[:4]))
    if run.notes:
        verdict = f"[{run.verdict}] " if run.verdict else ""
        parts.append(f"note: {verdict}{run.notes[-1]['text']}")
    return "; ".join(part for part in parts if part)
