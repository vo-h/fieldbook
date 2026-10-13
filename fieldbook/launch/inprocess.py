"""`fb.init()`: what `fieldbook run` does, for notebooks and interactive sessions
(runs.md §10)."""

import json
import os
import secrets
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

from fieldbook import credentials
from fieldbook.code import environments, snapshot
from fieldbook.config import require_project, settings
from fieldbook.launch import autolog, context, new_run_id, runenv, seeding

_started_here = False  # the current run was started by init() in this process


@dataclass(frozen=True)
class Run:
    run_id: str
    code: str  # code snapshot hash
    seed: int

    def __repr__(self) -> str:
        return f"Run({self.run_id}, code {self.code[:12]}, seed {self.seed})"


def init(
    *,
    seed: int | None = None,
    experiment: str | None = None,
    deterministic: bool = False,
    max_snapshot_mb: int = snapshot.MAX_SIZE_MB,
) -> Run:
    """Start a run in this process: run ID, code snapshot of the repository containing the
    working directory, environment record, seed, and MLflow settings. Dataset versions
    saved afterwards record the run, and MLflow runs started afterwards are tagged with it.

    Call it at the top of a notebook, after saving the notebook, since the snapshot holds
    files as they are on disk. Calling it again starts a new run with a new snapshot.
    Inside `fieldbook run`, it returns the launched run instead.
    """
    global _started_here
    launched = context.current()
    if launched.run_id and not _started_here:
        return Run(launched.run_id, launched.code, launched.seed or 0)
    require_project()  # runs belong to a project
    cfg = settings()
    snap = snapshot.take(Path.cwd(), cfg, max_size_mb=max_snapshot_mb)
    _, env_secrets = credentials.project_values(Path.cwd(), snap.repo)
    for name, value in env_secrets.items():
        os.environ.setdefault(name, value)  # variables already set win
    asked_seed = seed
    seed = seed if seed is not None else secrets.randbelow(2**31)
    found = environments.query_interpreter(sys.executable)
    record = environments.record(snap.repo, seed, found, cfg, deterministic=deterministic)
    run = Run(new_run_id(), snap.hash, seed)
    _note_repeat(run, record, snap.repo, asked_seed, cfg)
    run_context = context.RunContext(
        run_id=run.run_id,
        code=run.code,
        git=snapshot.git_state(snap.repo),
        env=json.dumps(record, sort_keys=True),
        seed=seed,
        secrets=credentials.run_secret_names(env_secrets),
        deterministic=deterministic,
    )
    experiment = experiment or cfg.experiment or cfg.project
    os.environ.update(
        runenv.run_env(cfg, run_context, experiment=experiment, found=found, python=sys.executable)
    )
    seeding.apply(seed, deterministic=deterministic)
    autolog.enable()
    _started_here = True
    print(f"run {run.run_id}  code {run.code[:12]}  seed {seed}", file=sys.stderr)
    return run


def _note_repeat(run: Run, record: dict, repo: Path, asked_seed: int | None, cfg) -> None:  # noqa: ANN001
    """A script started with plain `python`: say so if it repeats an earlier run
    (runs.md §16). Not in notebooks and sessions, which have no command to repeat."""
    from fieldbook.launch import local, repeats

    if hasattr(sys, "ps1") or sys.flags.interactive or "ipykernel" in sys.modules:
        return
    if not sys.argv or not sys.argv[0].endswith(".py"):
        return
    mark = repeats.fingerprint(code=run.code, environment=record["packages"], command=sys.argv,
                               workdir=Path.cwd(), repo=repo, seed=asked_seed)  # fmt: skip
    local.note_repeat(mark, run.run_id, cfg)


def seed(value: int, *, key: object = None, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch in this process, as `fieldbook run` does at start-up; for
    processes the launcher does not start, such as the trials of a hyperparameter search.

    With `key` (e.g. a trial's config), the seed is derived from `value` and `key`, so each
    trial gets its own seed and a rerun of the same trial gets the same one. Returns the
    seed applied.
    """
    if key is not None:
        text = json.dumps(key, sort_keys=True, default=str).encode()
        value = (value + zlib.crc32(text)) % 2**31
    seeding.apply(value, deterministic=deterministic)
    return value


def note(text: str, *, verdict: str | None = None) -> None:
    """Note what the current run showed, from its own script (as `fieldbook log note RUN TEXT`):
    `fb.note("val loss plateaus at 0.31 from epoch 4", verdict="inconclusive")`."""
    from fieldbook import notes

    run_id = context.current().run_id
    if not run_id:
        raise RuntimeError("fb.note() needs a run: use fieldbook run, or call fb.init() first")
    notes.add(run_id, text, verdict=verdict)
