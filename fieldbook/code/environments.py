"""The project's Python environment and the record of it (runs.md §3, §6). Runs,
UIs, and pipeline services use the project's environment, not necessarily the one `fieldbook` is
installed in: a global `pipx` or `uv tool` install has none of the project's packages."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fieldbook import config
from fieldbook.code import snapshot
from fieldbook.config import Settings


@dataclass(frozen=True)
class PythonEnv:
    python: Path
    venv: Path | None = None  # None: the interpreter running fieldbook

    @property
    def bin_dir(self) -> Path:
        return self.python.parent

    def activated(self) -> dict[str, str]:
        """Variables that activate the environment for a child process, as `source
        .venv/bin/activate` does: its `bin/` first on PATH, so its tools resolve."""
        if self.venv is None:
            return {}
        return {
            "VIRTUAL_ENV": str(self.venv),
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        }


def find(workdir: Path, repo: Path | None = None) -> PythonEnv:
    """The project's environment: the active one (`$VIRTUAL_ENV`), else the nearest `.venv`
    from `workdir` up to the repository root (`repo`, found from `workdir` if not given),
    else the interpreter running fieldbook."""
    if repo is None:
        try:
            repo = snapshot.repo_root(workdir)
        except snapshot.SnapshotError:
            repo = workdir.resolve()
    candidates = [Path(os.environ["VIRTUAL_ENV"])] if os.environ.get("VIRTUAL_ENV") else []
    directory = workdir.resolve()
    while True:
        candidates.append(directory / ".venv")
        if directory == repo or directory.parent == directory:
            break
        directory = directory.parent
    for venv in candidates:
        if (python := _venv_python(venv)).is_file():
            return PythonEnv(python, venv)
    return PythonEnv(Path(sys.executable))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# Run by the script's interpreter: its Python version and installed packages.
_INTERPRETER_QUERY = """
import importlib.metadata as m, json, platform, re
names = {}
for d in m.distributions():
    name = d.metadata["Name"]
    if name:
        names[re.sub(r"[-_.]+", "-", name).lower()] = d.version
packages = sorted(f"{n}=={v}" for n, v in names.items())
import importlib.util
print(json.dumps({
    "python": platform.python_version(),
    "packages": packages,
    "fieldbook": importlib.util.find_spec("fieldbook") is not None,
    "psutil": importlib.util.find_spec("psutil") is not None,
    "mlflow": importlib.util.find_spec("mlflow") is not None,
    # The local SQLite store needs full MLflow; mlflow-skinny lacks SQLAlchemy and Alembic.
    "mlflow_sql": all(importlib.util.find_spec(m) for m in ("sqlalchemy", "alembic")),
}))
"""


def query_interpreter(python: str) -> dict:
    """Python version, installed packages, and whether fieldbook is importable.

    Asking the interpreter takes about 270 ms, so the answer is kept in this machine's state
    folder, one file per interpreter, until its installed packages change: installing,
    upgrading, or removing a package changes its `site-packages` folder. Not shared: each
    machine records what is actually installed there.
    """
    fingerprint, cache = _fingerprint(Path(python)), _cache_file(python)
    if fingerprint and cache:
        try:
            saved = json.loads(cache.read_text())
            if saved["fingerprint"] == fingerprint:
                return saved["found"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    found = _ask(python)
    if found and fingerprint and cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"fingerprint": fingerprint, "found": found}))
        except OSError:
            pass
    return found


def _ask(python: str) -> dict:
    result = subprocess.run([python, "-c", _INTERPRETER_QUERY], capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def _fingerprint(python: Path) -> list | None:
    """The interpreter and its `site-packages` folders, with their modification times; None
    if they cannot be found (then the interpreter is asked every time)."""
    prefix = python.parent.parent  # the environment, e.g. .venv (not the interpreter it links to)
    folders = sorted(prefix.glob("lib/python3*/site-packages")) + sorted(
        prefix.glob("Lib/site-packages")
    )
    if not folders:
        return None
    try:
        files = [python.resolve(), *folders]
        return [[str(f), f.stat().st_mtime_ns] for f in files]
    except OSError:
        return None


def _cache_file(python: str) -> Path | None:
    try:
        state = config.settings().state_dir
    except config.ConfigError:
        return None
    name = hashlib.sha256(str(Path(python).absolute()).encode()).hexdigest()[:16]
    return state / "environments" / f"{name}.json"


def record(
    repo: Path,
    seed: int,
    found: dict,
    cfg: Settings | None = None,
    *,
    deterministic: bool = False,
    workdir: Path | None = None,
) -> dict[str, str | int | bool | None]:
    """The environment record (runs.md §5), from `query_interpreter` output.

    `packages` is the hash of the packages actually installed (stored under
    `packages/`); `uv_lock` is the hash of the lock file, i.e. what should be installed.
    """
    lock = repo / "uv.lock"
    packages = found.get("packages")
    try:  # where in the repository the command was run, so that it can be run again there
        cwd = (workdir or Path.cwd()).resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        cwd = None
    return {
        "cwd": cwd,
        "seed": seed,
        "deterministic": deterministic,
        "python": found.get("python"),
        "packages": snapshot.store_packages(packages, cfg) if packages else None,
        "platform": platform.platform(),
        "uv_lock": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "cuda_driver": _cuda_driver(),
        "image": os.environ.get("FIELDBOOK_IMAGE"),  # the environment image of a cloud run
    }


def _cuda_driver() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout else None
