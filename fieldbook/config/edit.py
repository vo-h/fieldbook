"""Changing settings: `fieldbook config set`, `unset`, and `get`, as `git config` does.

A setting goes to the project file (`[tool.fieldbook]` in `pyproject.toml`, else
`fieldbook.toml`), committed and shared by everyone on the project; with `local=True`, to the
project's `.fieldbook/config.toml`, for this machine only. Values are checked with the
settings model before they are written, and the whole configuration is read back
afterwards; if it no longer loads, the file is restored. Files are edited with tomlkit,
which keeps comments and layout.
"""

import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import ValidationError

from fieldbook.config.display import effective
from fieldbook.config.files import PATH_KEYS, project, settings
from fieldbook.config.model import LOCAL_DIR, LOCAL_FILE, PROJECT_FILE, ConfigError, Settings


@dataclass(frozen=True)
class Place:
    path: Path
    keys: tuple[str, ...]  # the table the setting goes in, e.g. ("tool", "fieldbook")
    label: str  # for messages


def set_value(key: str, raw: str, *, local: bool = False, start: Path | None = None) -> str:
    """Write `key = raw` (converted to the setting's type) to its file; returns a message."""
    _check_key(key)
    value = _convert(key, raw, start or Path.cwd())
    _check_value(key, value)
    place = _place(local, start)
    with _rollback(place.path, start):
        doc = _load(place.path)
        table = _table(doc, place.keys, create=True)
        table[key] = value
        _save(place.path, doc)
    return f"{key} = {_shown(key, value)} in {place.label}" + _overridden(key)


def unset_value(key: str, *, local: bool = False, start: Path | None = None) -> str:
    """Remove `key` from its file; returns a message."""
    _check_key(key)
    place = _place(local, start)
    if not place.path.is_file():
        return f"{key} is not set in {place.label}"
    with _rollback(place.path, start):
        doc = _load(place.path)
        table = _table(doc, place.keys, create=False)
        if table is None or key not in table:
            return f"{key} is not set in {place.label}"
        del table[key]
        _save(place.path, doc)
    return f"{key} removed from {place.label}" + _overridden(key)


def get_value(key: str, start: Path | None = None) -> str | None:
    """The effective value of `key` (as `fieldbook config` shows it), or None if unset."""
    _check_key(key)
    if typing.get_origin(Settings.model_fields[key].annotation) is list:
        return ",".join(getattr(settings(start), key)) or None  # as `set` takes it
    return next(value for name, value, _ in effective(start) if name == key)


# --- where and what -------------------------------------------------------------------


def _check_key(key: str) -> None:
    if key not in Settings.model_fields:
        known = ", ".join(Settings.model_fields)
        raise ConfigError(f"unknown setting {key!r}; settings: {known}")


def _convert(key: str, raw: str, cwd: Path) -> Any:
    """`raw` as the setting's type: lists from `a,b,c`, numbers as numbers, and relative
    paths made absolute (in a file they would be relative to the file)."""
    annotation = Settings.model_fields[key].annotation
    if typing.get_origin(annotation) is list:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if annotation is int:
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"{key} must be a whole number, not {raw!r}") from None
    if key in PATH_KEYS and "://" not in raw and not raw.startswith("~"):
        return str((cwd / raw).resolve())
    return raw


def _check_value(key: str, value: Any) -> None:
    if key == "tracking_uri":
        from fieldbook import credentials

        if credentials.has_secret(value):
            raise ConfigError(
                "tracking_uri must not contain a password: put the address without it here, "
                "and the password in PGPASSWORD in the project's .env"
            )
    try:
        Settings(**{key: value})
    except ValidationError as e:
        message = e.errors()[0]["msg"].removeprefix("Value error, ")
        raise ConfigError(f"invalid setting {key}: {message}") from None


def _place(local: bool, start: Path | None) -> Place:
    found = project(start)
    if local:
        if found.file is None:
            raise ConfigError(
                "--local needs a project: run `fieldbook init` in the project's folder"
            )
        return Place(found.local_file, (), f"{LOCAL_DIR}/{LOCAL_FILE}")
    if found.file is not None and found.file.name == PROJECT_FILE:
        return Place(found.file, (), str(found.file))
    if found.file is not None:
        return Place(found.file, ("tool", "fieldbook"), f"{found.file} [tool.fieldbook]")
    root = _repo_root(
        start or Path.cwd()
    )  # no project file yet: create it, as `fieldbook init` does
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        return Place(pyproject, ("tool", "fieldbook"), f"{pyproject} [tool.fieldbook]")
    return Place(root / PROJECT_FILE, (), str(root / PROJECT_FILE))


def _repo_root(start: Path) -> Path:
    from fieldbook.code import snapshot

    try:
        return snapshot.repo_root(start)
    except snapshot.SnapshotError:
        return start.resolve()


# --- files ----------------------------------------------------------------------------


def _load(path: Path) -> tomlkit.TOMLDocument:
    return tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()


def _save(path: Path, doc: tomlkit.TOMLDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc))


def _table(doc: Any, keys: tuple[str, ...], *, create: bool) -> Any:
    table = doc
    for name in keys:
        if name not in table:
            if not create:
                return None
            table[name] = tomlkit.table(is_super_table=name == "tool")
        table = table[name]
    return table


class _rollback:
    """Restore `path` if the configuration no longer loads after the change."""

    def __init__(self, path: Path, start: Path | None):
        self.path, self.start = path, start
        self.before = path.read_text() if path.is_file() else None

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind, error, traceback) -> bool:  # noqa: ANN001
        if kind is not None:
            return False
        try:
            settings(self.start)
        except ConfigError as e:
            if self.before is None:
                self.path.unlink(missing_ok=True)
            else:
                self.path.write_text(self.before)
            raise ConfigError(f"not changed: {e}") from None
        return False


def _shown(key: str, value: Any) -> str:
    return ",".join(value) if isinstance(value, list) else str(value)


def _overridden(key: str) -> str:
    variable = f"FIELDBOOK_{key.upper()}"
    return f" (note: {variable} is set and overrides it)" if os.environ.get(variable) else ""
