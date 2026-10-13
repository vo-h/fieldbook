"""Notes on runs: what was learned, kept with the record (runs.md §15). `fieldbook log note`,
`fb.note()`.

A run's metrics say what happened; a note says what it means ("lr 3e-4 diverges after
epoch 2: dead end"), so that the next person, or the next session of a coding agent, reads
the conclusion instead of working it out again.

Each note is a small file, `notes/<run ID>/<time>-<random>.json`, in the project's folder
of the storage root: notes work for runs that log nothing to MLflow, members never write
the same file, and nothing is ever rewritten.
"""

import json
import secrets
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from fieldbook import storage
from fieldbook.config import Settings, settings
from fieldbook.launch import agents

FOLDER = "notes"
VERDICTS = ("good", "bad", "inconclusive")


class NoteError(ValueError):
    pass


@dataclass(frozen=True)
class Note:
    run_id: str
    time: float  # seconds since the epoch
    author: str
    text: str
    verdict: str | None = None  # one of VERDICTS: was the run worth building on?
    agent: str = ""  # the coding agent that wrote it for the author (runs.md §20)


def add(run_id: str, text: str, *, verdict: str | None = None,
        cfg: Settings | None = None) -> Note:  # fmt: skip
    """Add a note to a run. A run can have several; the latest verdict is the run's."""
    cfg = cfg or settings()
    text = text.strip()
    if not text and not verdict:
        raise NoteError("a note needs text or a verdict")
    if verdict is not None and verdict not in VERDICTS:
        raise NoteError(f"verdict {verdict!r}: one of {', '.join(VERDICTS)}")
    note = Note(run_id, time.time(), cfg.author or "", text, verdict, agents.current())
    files = storage.fs(cfg)
    storage.makedirs(files, f"{FOLDER}/{run_id}")
    name = f"{int(note.time * 1000):015d}-{secrets.token_hex(3)}.json"
    files.pipe_file(f"{FOLDER}/{run_id}/{name}", json.dumps(asdict(note)).encode())
    return note


def of(run_ids: Iterable[str], cfg: Settings | None = None) -> dict[str, list[Note]]:
    """The notes of each of `run_ids` that has any, oldest first."""
    files = storage.fs(cfg or settings())
    wanted = set(run_ids)
    try:
        paths = [p for p in files.glob(f"{FOLDER}/*/*.json") if p.split("/")[-2] in wanted]
    except FileNotFoundError:
        return {}
    found: dict[str, list[Note]] = {}
    for path in sorted(paths):  # file names begin with the time
        note = Note(**json.loads(files.cat_file(path)))
        found.setdefault(note.run_id, []).append(note)
    return found


def verdict(notes: list[Note]) -> str | None:
    """A run's verdict: its latest note that gives one."""
    return next((n.verdict for n in reversed(notes) if n.verdict), None)
