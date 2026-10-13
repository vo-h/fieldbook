"""`fieldbook run` on this machine: snapshot the code, record the environment, and run a script
or task (runs.md)."""

import json
import os
import secrets
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from fieldbook import credentials, storage
from fieldbook.code import environments, snapshot
from fieldbook.config import Settings, settings
from fieldbook.launch import LaunchError, agents, context, new_run_id, note, repeats, runenv, runlog
from fieldbook.launch.context import RunContext
from fieldbook.launch.tasks import Task


@dataclass
class Launch:
    """A prepared run: what `execute` starts, and the context it passes on."""

    commands: list[str | list[str]]  # shell commands (setup) or argv
    workdir: Path
    env: dict[str, str]
    context: RunContext
    cfg: Settings
    owns_scratch: bool = False  # delete the scratch area afterwards (unless keep_scratch)
    keep_scratch: bool = False
    record: dict | None = None  # the start record; None: nothing is kept of this run
    repo: Path | None = None
    output_dir: Path | None = None
    on_machine: bool = False  # the cloud machine's side of a run
    secret_values: tuple[str, ...] = ()


def run(task: Task, **options) -> int:
    """Snapshot, then execute `task` (see `prepare`). Returns its exit code."""
    return execute(prepare(task, **options))


def prepare(
    task: Task,
    *,
    seed: int | None = None,
    experiment: str | None = None,
    max_snapshot_mb: int = snapshot.MAX_SIZE_MB,
    smoke: int | None = None,
    deterministic: bool = False,
    keep_scratch: bool = False,
    run_id: str | None = None,
    code: str | None = None,
    again: bool = False,
    rerun: bool = False,
) -> Launch:
    """Run steps 1-4 (runs.md §3): run ID, code snapshot, environment record,
    MLflow setup. Nothing of the task runs yet.

    `smoke` makes this a smoke run reading up to that many items per folder and saving
    versions to a scratch area, deleted afterwards unless `keep_scratch`
    (runs.md §9). A step of a pipeline smoke run (`PLATFORM_SMOKE` and
    `PLATFORM_SCRATCH` already set) is a smoke run using the pipeline's scratch area.
    A run cannot start inside another run (runs.md §3).

    `rerun` (with `code`; `fieldbook run --rerun`): the run repeats an earlier one from a
    checkout of its snapshot;
    versions it saves go to a scratch area, left for the caller to compare and delete
    (runs.md §17).

    On a cloud machine, `run_id` and `code` come from the laptop (cloud-runs.md
    §2): the code is that snapshot, already checked out in the working directory, so no
    new snapshot is taken.
    """
    outer = context.current()
    if outer.run_id:
        raise LaunchError(
            f"already inside run {outer.run_id} (started by fieldbook run or fb.init()); a run "
            "cannot start another. Call the code directly (python script.py, or import it). "
            "A script that launches runs itself is started with FIELDBOOK_RECORD=0, so that "
            "importing fieldbook does not start a run for it."
        )
    cfg = settings()
    run_id = run_id or new_run_id()
    smoke = smoke or outer.smoke
    scratch = outer.scratch or (run_id if smoke or rerun else None)
    if code is not None:  # on a cloud machine: the checkout's root, set by the launch command
        root = Path(os.environ.get("FIELDBOOK_CODE_ROOT") or task.workdir).resolve()
        snap = snapshot.Snapshot(code, root, created=False, size=0)
        git = os.environ.get("FIELDBOOK_GIT", "")  # from the laptop: the machine has no repository
    else:
        try:
            snap = snapshot.take(task.workdir, cfg, max_size_mb=max_snapshot_mb)
        except snapshot.SnapshotError as e:
            raise LaunchError(str(e)) from None
        if snap.outside_links:
            note(f"warning: symlinks outside the repository are not in the snapshot, so this "
                  f"run cannot be reproduced from it: {', '.join(snap.outside_links)}")  # fmt: skip
        git = snapshot.git_state(snap.repo)
    asked_seed = seed
    seed = seed if seed is not None else secrets.randbelow(2**31)
    project = environments.find(task.workdir, snap.repo)
    python = str(project.python)
    found = environments.query_interpreter(python)
    run_command = task.run
    if isinstance(run_command, list) and run_command[0] == "python":
        if found.get("fieldbook"):  # start through the bootstrap, which applies the seed
            run_command = [python, "-m", "fieldbook.launch.bootstrap", *run_command[1:]]
        else:
            run_command = [python, *run_command[1:]]
            note(f"fieldbook is not installed in {python}: the seed is not applied")
    elif isinstance(run_command, list) and Path(run_command[0]).name == "torchrun":
        seeded = _torchrun_through_bootstrap(run_command) if found.get("fieldbook") else None
        if seeded is None:
            note("the seed is not applied in torchrun's workers (needs SCRIPT.py and fieldbook)")
        run_command = seeded or run_command
    secrets_file, env_secrets = credentials.project_values(task.workdir, snap.repo)
    if env_secrets:
        note(f"secrets from {secrets_file.name}: {', '.join(sorted(env_secrets))}")
        if credentials.is_tracked(secrets_file, snap.repo):
            note(f"warning: git tracks {secrets_file}; its secrets may be in the history")
    record = environments.record(
        snap.repo, seed, found, cfg, deterministic=deterministic, workdir=task.workdir
    )
    if code is None and not smoke:  # not a cloud machine's side, a rerun, or a test
        mark = repeats.fingerprint(
            code=snap.hash, environment=record["packages"], command=task.run,
            workdir=task.workdir, repo=snap.repo, seed=asked_seed, envs=task.envs,
        )  # fmt: skip
        note_repeat(mark, run_id, cfg, quiet=again)
    run_context = RunContext(
        run_id=run_id,
        code=snap.hash,
        git=git,
        env=json.dumps(record, sort_keys=True),
        seed=seed,
        secrets=credentials.run_secret_names(env_secrets),
        deterministic=deterministic,
        smoke=smoke,
        scratch=scratch,
    )
    experiment = (experiment or cfg.experiment or cfg.project) + ("-smoke" if smoke else "")
    env = {
        **env_secrets,  # variables already in the environment win
        **os.environ,
        **project.activated(),
        **task.envs,
    }
    env |= runenv.run_env(
        cfg, run_context, experiment=experiment, found=found, python=python,
        uri=os.environ.get("MLFLOW_TRACKING_URI") or env_secrets.get("MLFLOW_TRACKING_URI"),
    )  # fmt: skip
    if isinstance(run_command, list) and not shutil.which(run_command[0], path=env.get("PATH")):
        raise LaunchError(f"{run_command[0]}: command not found")
    output_dir = cfg.state_dir / "outputs" / run_id  # for files that would go elsewhere
    env["PLATFORM_OUTPUT_DIR"] = str(output_dir)
    secret_values = tuple(v for v in env_secrets.values() if v) + tuple(
        os.environ[name] for name in run_context.secrets if os.environ.get(name)
    )
    typed = task.run if isinstance(task.run, str) else task.run[task.run[:1] == ["python"] :]
    on_machine = code is not None and not rerun
    record = {
        "started": time.time(),
        "author": cfg.author or "",
        "agent": agents.current(),
        "command": typed if isinstance(typed, str) else
        credentials.mask_command(list(typed), list(secret_values)),
        "code": snap.hash,
        "git": git,
        "env": run_context.env,
        "smoke": bool(smoke),
        "where": "cloud" if on_machine else "local",
        "rerun_of": os.environ.get("FIELDBOOK_RERUN_OF", "") if rerun else "",
    }  # fmt: skip
    # of a cluster's machines, the first keeps the record: they share one run
    keeps = os.environ.get("SKYPILOT_NODE_RANK", "0") == "0"
    at = f"  git {snapshot.short_git(git)}" if git else ""
    note(f"run {run_id}  code {snap.hash[:12]}{at}  seed {seed}  python {python}")
    if smoke:
        note(f"smoke run: up to {smoke} items per folder; versions go to scratch {scratch}")
    if task.ignored and code is None:  # cloud machines use resources and num_nodes at launch
        note(f"ignored locally: {', '.join(task.ignored)}")
    return Launch(
        commands=([task.setup] if task.setup else []) + [run_command],
        workdir=task.workdir,
        env=env,
        context=run_context,
        cfg=cfg,
        owns_scratch=bool(smoke) and not outer.scratch,  # a rerun's is its caller's
        keep_scratch=keep_scratch,
        record=record if keeps else None,
        repo=snap.repo,
        output_dir=output_dir,
        on_machine=on_machine,
        secret_values=secret_values,
    )


def note_repeat(mark: str, run_id: str, cfg: Settings, *, quiet: bool = False) -> None:
    """Say so if this run repeats an earlier one (runs.md §16), and leave its own mark.
    It runs either way: on this machine a repeat costs time, not money."""
    try:
        if not quiet and (repeat := repeats.find(mark, cfg)):
            note(f"this repeats an earlier run (same code, environment, command, and data): "
                 f"{repeat.summary}")  # fmt: skip
        repeats.mark(mark, run_id, cfg)
    except Exception as e:  # bookkeeping never stops a run
        note(f"warning: could not check for an earlier run: {type(e).__name__}: {e}")


def _torchrun_through_bootstrap(argv: list[str]) -> list[str] | None:
    """`torchrun [OPTIONS] SCRIPT.py ARGS` as `torchrun [OPTIONS] -m fieldbook.launch.bootstrap
    SCRIPT.py ARGS`, so every worker applies the run's seed before the script, as
    `fieldbook run SCRIPT.py` does. None if torchrun does not start a script (`-m`,
    `--no-python`)."""
    for i, arg in enumerate(argv[1:], start=1):
        if arg in ("-m", "--module", "--no-python", "--no_python"):
            return None
        if arg.endswith(".py") and not arg.startswith("-"):
            return [*argv[:i], "-m", "fieldbook.launch.bootstrap", *argv[i:]]
    return None


def execute(launch: Launch) -> int:
    """Run step 5: run the task's commands in order, stopping at the first that fails.
    Returns the exit code. The run's record, output, and files are kept as it goes
    (runs.md §18)."""
    keeper = None
    if launch.record is not None:
        from fieldbook.launch.keeping import Keeper

        launch.output_dir.mkdir(parents=True, exist_ok=True)
        keeper = Keeper(
            launch.context.run_id, launch.cfg, launch.record, repo=launch.repo,
            output_dir=launch.output_dir, secrets=launch.secret_values,
            on_machine=launch.on_machine,
            files=launch.cfg.keep_outputs and not launch.context.smoke,
        )  # fmt: skip
        keeper.start()
    code = 1
    try:
        for command in launch.commands:
            shell = isinstance(command, str)
            copied = runlog.popen(
                command,
                keeper.log if keeper else None,
                cwd=launch.workdir,
                env=launch.env,
                shell=shell,
                executable="/bin/bash" if shell else None,
            )
            code = wait(copied.process)
            copied.drain()
            if code != 0:
                return code
        return 0
    finally:
        if launch.owns_scratch:
            finish_scratch(launch.context.scratch, launch.cfg, keep=launch.keep_scratch)
        if keeper is not None:
            keeper.finish(code)


def finish_scratch(scratch: str, cfg: Settings, *, keep: bool) -> None:
    """Delete a smoke run's scratch area, or say how to inspect it (runs.md §9).
    Files stored only by the smoke run are freed by the next cleanup."""
    from fieldbook.data.smoke import SCRATCH_DIR

    fs = storage.fs(cfg)
    path = f"{SCRATCH_DIR}/{scratch}"
    if not fs.exists(path):
        return
    if keep:
        note(f"scratch kept; inspect with FIELDBOOK_SCRATCH={scratch} fieldbook data ls NAME")
    else:
        fs.rm(path, recursive=True)


def wait(process: subprocess.Popen) -> int:
    """Wait for the script, passing on signals sent to `fieldbook run`, so the script never
    outlives it. Returns the exit code, 128 + N if the script was stopped by signal N.

    Ctrl-C in a terminal already reaches the script (same process group), so SIGINT is
    not forwarded, only ignored here; SIGTERM and SIGHUP are forwarded. SIGKILL cannot
    be caught: killing `fieldbook run` with it still leaves the script running.
    """

    def forward(signum: int, frame: object) -> None:
        process.send_signal(signum)

    handlers = {signal.SIGINT: lambda signum, frame: None, signal.SIGTERM: forward}
    if hasattr(signal, "SIGHUP"):
        handlers[signal.SIGHUP] = forward
    previous = {sig: signal.signal(sig, handler) for sig, handler in handlers.items()}
    try:
        code = process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 128 - code if code < 0 else code
