"""Where settings come from, highest priority first:

1. environment variables `FIELDBOOK_<NAME>` (command-line options such as
   `fieldbook run --experiment` set them);
2. `.fieldbook/config.toml` in the project: this machine only, not committed (e.g. a
   different storage root on this laptop);
3. the project file, committed: `[tool.fieldbook]` in `pyproject.toml`, or `fieldbook.toml`,
   nearest upward from the working directory. Everything the project's members share;
4. defaults: the project's `.fieldbook/` for local data and state, and git's `user.name`
   as the author.

A project's `.fieldbook/` sits next to its project file; git worktrees of one repository
share the main working tree's, so they see the same runs and data.
"""

import functools
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_settings import EnvSettingsSource

from fieldbook.config.model import (
    LOCAL_DIR,
    LOCAL_FILE,
    PROJECT_FILE,
    PROJECT_ONLY,
    ConfigError,
    Settings,
)

PATH_KEYS = ("storage_root", "state_dir")  # relative paths in a file are relative to the file
KEYS = set(Settings.model_fields)


@dataclass(frozen=True)
class Layer:
    source: str  # where the values come from, for `fieldbook config` (file paths are listed there)
    values: dict[str, Any]


@dataclass(frozen=True)
class Project:
    file: Path | None  # the project file, or None outside a project
    root: Path  # the folder holding it (outside a project: the working directory)
    local: Path  # its .fieldbook folder

    @property
    def local_file(self) -> Path:
        return self.local / LOCAL_FILE


def project(start: Path | None = None) -> Project:
    """The project containing `start` (default: the working directory)."""
    path, _ = find_project_file(start)
    root = path.parent if path is not None else (start or Path.cwd()).resolve()
    return Project(path, root, local_dir(root))


def layers(start: Path | None = None) -> list[Layer]:
    """Each source's values, highest priority first."""
    env = {k: v for k, v in EnvSettingsSource(Settings)().items() if v != ""}
    _project_only(env, "the variable FIELDBOOK_{}")
    result = [Layer("environment (FIELDBOOK_*)", env)]
    found = project(start)
    if found.local_file.is_file():
        values = _check_keys(_read(found.local_file), KEYS, found.local_file)
        _project_only(values, str(found.local_file))
        result.append(Layer(f"{LOCAL_DIR}/{LOCAL_FILE}", _relative(values, found.local)))
    if found.file is not None:
        _, values = find_project_file(start)
        where = "" if found.file.name == PROJECT_FILE else " [tool.fieldbook]"
        result.append(Layer(f"{found.file.name}{where}", _relative(values, found.root)))
    return result


def _project_only(values: dict[str, Any], where: str) -> None:
    for key in PROJECT_ONLY:
        if key in values:
            place = where.format(key.upper())
            raise ConfigError(
                f"{key} is set in the project file only, not in {place}: a change to the "
                "budget is committed and reviewed like code"
            )


def settings(start: Path | None = None) -> Settings:
    found = project(start)
    merged: dict[str, Any] = {
        "storage_root": str(found.local / "data"),
        "state_dir": str(found.local),
    }
    if author := git_author(found.root):
        merged["author"] = author
    for layer in reversed(layers(start)):
        merged.update(layer.values)
    try:
        return Settings(**merged)
    except ValidationError as e:
        error = e.errors()[0]
        field = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        raise ConfigError(f"invalid setting {field}: {message}") from None


def require_project(start: Path | None = None) -> None:
    """Commands that use data or runs work inside a project."""
    if project(start).file is None and not os.environ.get("FIELDBOOK_PROJECT"):
        raise ConfigError(
            "not in a fieldbook project: run `fieldbook init` in the project's folder "
            "(or set FIELDBOOK_PROJECT and FIELDBOOK_STORAGE_ROOT)"
        )


def find_project_file(start: Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    """The nearest project file upward from `start` (default: the working directory) and
    its settings, or (None, {})."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        dedicated = candidate / PROJECT_FILE
        if dedicated.is_file():
            return dedicated, _check_keys(_read(dedicated), KEYS, dedicated)
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            section = _read(pyproject).get("tool", {}).get("fieldbook")
            if section is not None:
                return pyproject, _check_keys(section, KEYS, pyproject, "[tool.fieldbook]")
    return None, {}


@functools.lru_cache(maxsize=64)
def local_dir(root: Path) -> Path:
    """`<root>/.fieldbook`; in a git worktree, the main working tree's, so all worktrees of a
    repository share one."""
    common = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    top = _git(root, "rev-parse", "--show-toplevel")
    if common and top and Path(common).name == ".git":
        main, here = Path(common).parent, Path(top)
        if main != here and root.is_relative_to(here):
            return main / root.relative_to(here) / LOCAL_DIR
    return root / LOCAL_DIR


@functools.lru_cache(maxsize=64)
def git_author(root: Path) -> str | None:
    """git's `user.name` for this repository, the name its commits carry."""
    return _git(root, "config", "user.name") or None


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                timeout=10)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _read(path: Path) -> dict[str, Any]:
    """Parsed TOML, cached until the file changes (settings are read on every dataset open)."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _parsed:
        try:
            _parsed[key] = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from None
    return _parsed[key]


_parsed: dict[tuple[str, int, int], dict[str, Any]] = {}


def _check_keys(data: Any, allowed: set[str], path: Path, where: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{path} {where}: expected a table".replace("  ", " "))
    unknown = sorted(set(data) - allowed)
    if unknown:
        location = f"{path} {where}".strip()
        raise ConfigError(f"{location}: unknown settings {unknown} (allowed: {sorted(allowed)})")
    return data


def _relative(values: dict[str, Any], base: Path) -> dict[str, Any]:
    """Plain relative paths in a settings file are relative to that file's folder."""
    return {k: _relative_to(v, base) if k in PATH_KEYS else v for k, v in values.items()}


def _relative_to(value: str, base: Path) -> str:
    if "://" in value or Path(value).expanduser().is_absolute():
        return value
    return str(base / value)
