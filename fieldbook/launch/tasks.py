"""What to run: a script, a command, or a task file (`fieldbook run train.yaml`; SkyPilot's
format, so the same file serves cloud runs)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fieldbook.launch import LaunchError

LOCAL_TASK_FIELDS = {"name", "workdir", "envs", "setup", "run"}


@dataclass
class Task:
    run: str | list[str]  # shell command, or argv for scripts (argv[0] "python": see run)
    setup: str | None = None
    workdir: Path = field(default_factory=Path.cwd)
    envs: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)  # task fields not used locally
    resources: dict | None = None  # task file `resources`, used by cloud runs
    num_nodes: int | None = None


def script_task(script: Path, args: list[str]) -> Task:
    if not script.is_file():
        raise LaunchError(f"{script} not found")
    return Task(run=["python", str(script), *args])


def command_task(argv: list[str]) -> Task:
    """Any command, such as `torchrun --nproc-per-node 4 train.py` (runs.md §2).
    It is found on the project environment's PATH."""
    return Task(run=list(argv))


def load_task(path: Path, args: list[str]) -> Task:
    """A SkyPilot task YAML, for local execution (runs.md §6)."""
    if args:
        raise LaunchError("arguments cannot be passed to a task file; use its envs")
    try:
        spec = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise LaunchError(f"cannot read task {path}: {e}") from None
    if not isinstance(spec, dict) or not isinstance(spec.get("run"), str):
        raise LaunchError(f"{path} needs a 'run' command")
    workdir = (path.parent / spec.get("workdir", ".")).resolve()
    if not workdir.is_dir():
        raise LaunchError(f"workdir {workdir} not found")
    return Task(
        run=spec["run"],
        setup=spec.get("setup"),
        workdir=workdir,
        envs={str(k): str(v) for k, v in (spec.get("envs") or {}).items()},
        ignored=sorted(set(spec) - LOCAL_TASK_FIELDS),
        resources=spec.get("resources"),
        num_nodes=spec.get("num_nodes"),
    )
