"""Lightweight toolkit for machine-learning experimentation.

Everything is loaded on first use (`fb.dataset`, `fb.init`, ...), so `import fieldbook`
is cheap: the bootstrap that starts every run and every `torchrun` rank imports it.

Importing it from a project's script that was started on its own (`python clean.py`)
starts a run for that process, as `fieldbook run` would have (runs.md §13).
"""

import importlib
from importlib.metadata import version
from typing import Any

__version__ = version("fieldbook")

# name -> (module, attribute or None for the module itself)
_EXPORTS = {
    "dataset": ("fieldbook.data", "dataset"),
    "init": ("fieldbook.launch.inprocess", "init"),
    "note": ("fieldbook.launch.inprocess", "note"),
    "runs": ("fieldbook.history", "table"),
    "seed": ("fieldbook.launch.inprocess", "seed"),
    "storage": ("fieldbook.storage", None),
}

__all__ = ["dataset", "init", "note", "runs", "seed", "storage"]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        loaded = importlib.import_module(module)
        return loaded if attribute is None else getattr(loaded, attribute)
    if name in ("data", "tracking", "launch", "code", "config", "images", "clouds", "credentials"):
        return importlib.import_module(f"fieldbook.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


def _record_this_run() -> None:
    from fieldbook.launch import autorecord

    autorecord.start()


_record_this_run()
