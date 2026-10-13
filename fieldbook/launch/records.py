"""A run's record of having started, being alive, and having ended (runs.md §18), in the
project's folder of the storage root, so that any member's machine can tell how a run is
doing or how it ended without the machine it ran on.

    runs/<run ID>.json          what was started: command, author, code, where, time limit
    runs/<run ID>.alive         rewritten every half minute while the run's process lives
    runs/<run ID>.end.<code>    written once, when it ends; the exit code is in the name
    runs/<run ID>/              its output, launch log, and the files it wrote

The three small files sit side by side so that one listing of `runs/` tells every run's
state: no file is read to list runs. Written by `fieldbook run` (here, and on a cloud machine);
a script started with plain `python` has no process around it to write them.
"""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fieldbook import storage
from fieldbook.config import Settings

FOLDER = "runs"
BEAT = 30  # seconds between signs of life (and between pieces of output)
STALE = 4 * BEAT  # without a sign of life for this long, the process is taken to be gone
LAUNCH_WINDOW = 1800  # seconds a cloud run may take to give its first sign of life

FINISHED, FAILED, RUNNING = "FINISHED", "FAILED", "RUNNING"
LAUNCHING, KILLED = "LAUNCHING", "KILLED"


@dataclass
class State:
    run_id: str
    started: bool = False  # has a start record
    alive: float | None = None  # when it last gave a sign of life
    exit_code: int | None = None
    ended: float | None = None

    def status(self, now: float | None = None) -> str:
        """FINISHED or FAILED once ended. Before that RUNNING while signs of life arrive,
        LAUNCHING for a cloud run that has given none yet, and KILLED when they stopped
        without an end: the process or its machine was removed from outside."""
        now = now or time.time()
        if self.exit_code is not None:
            return FINISHED if self.exit_code == 0 else FAILED
        if self.alive is None:
            return LAUNCHING if now - started_at(self.run_id) < LAUNCH_WINDOW else KILLED
        return RUNNING if now - self.alive < STALE else KILLED


def started_at(run_id: str) -> float:
    """When a run was created: its ID begins with the UTC time."""
    try:
        return datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return 0.0


def start(cfg: Settings, run_id: str, fields: dict[str, Any], *, keep: bool = False) -> None:
    """Write the start record. `keep`: leave one that exists (a cloud machine's side of a
    run, whose record the laptop wrote when it launched)."""
    files = storage.fs(cfg)
    storage.makedirs(files, FOLDER)
    path = f"{FOLDER}/{run_id}.json"
    if not (keep and files.exists(path)):
        files.pipe_file(path, json.dumps({"run_id": run_id, **fields}).encode())


def beat(cfg: Settings, run_id: str) -> None:
    storage.fs(cfg).pipe_file(f"{FOLDER}/{run_id}.alive", b"")


def end(cfg: Settings, run_id: str, exit_code: int, **fields: Any) -> None:
    files = storage.fs(cfg)
    storage.makedirs(files, FOLDER)
    body = json.dumps({"ended": time.time(), "exit_code": exit_code, **fields}).encode()
    files.pipe_file(f"{FOLDER}/{run_id}.end.{exit_code}", body)


def states(cfg: Settings, prefix: str = "") -> dict[str, State]:
    """Every run's state (or those whose ID starts with `prefix`), from one listing, made
    anew each time: this is what `fieldbook log wait` watches."""
    files = storage.fresh(storage.fs(cfg))
    try:
        listed = files.glob(f"{FOLDER}/{prefix}*", detail=True)
    except FileNotFoundError:
        return {}
    found: dict[str, State] = {}
    for path, info in listed.items():
        name = path.rsplit("/", 1)[-1]
        run_id, _, kind = name.partition(".")
        if not kind:
            continue  # the run's folder
        state = found.setdefault(run_id, State(run_id))
        if kind == "json":
            state.started = True
        elif kind == "alive":
            state.alive = _modified(info)
        elif kind.startswith("end."):
            state.exit_code, state.ended = int(kind[4:]), _modified(info)
    return {run_id: state for run_id, state in found.items() if state.started}


def _modified(info: dict) -> float:
    at = info.get("mtime") or info.get("LastModified") or info.get("created") or 0.0
    return at.timestamp() if hasattr(at, "timestamp") else float(at)


def read(cfg: Settings, run_ids: list[str]) -> dict[str, dict[str, Any]]:
    """The start records of `run_ids` (those that have one): one small file each."""
    files = storage.fs(cfg)
    found = {}
    for run_id in run_ids:
        try:
            found[run_id] = json.loads(files.cat_file(f"{FOLDER}/{run_id}.json"))
        except FileNotFoundError:
            continue
    return found


def read_end(cfg: Settings, state: State) -> dict[str, Any]:
    path = f"{FOLDER}/{state.run_id}.end.{state.exit_code}"
    try:
        return json.loads(storage.fs(cfg).cat_file(path))
    except (FileNotFoundError, ValueError):
        return {}
