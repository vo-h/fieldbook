"""Recording a run when a project's script imports fieldbook (runs.md §13), so that
`python clean.py` is recorded like `fieldbook run clean.py` without anyone remembering to.

Called at the end of `import fieldbook`. Everything before the decision uses the standard
library only, so the import stays cheap for the processes that must not be recorded: tools
(`pytest`, `fieldbook` itself), interactive sessions, runs already started by `fieldbook run`, and
their worker processes.
"""

import os
import sys
from pathlib import Path

OFF = ("0", "false", "no", "off")
PROJECT_FILES = ("fieldbook.toml", "pyproject.toml")


def start() -> None:
    """Start a run for this process if it is a project's script run on its own."""
    script = _project_script()
    if script is None:
        return
    from fieldbook.launch import inprocess

    try:
        inprocess.init()
    except Exception as e:  # never stop someone's own `python script.py`
        reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        print(f"fieldbook: this run is not recorded: {reason}", file=sys.stderr)


def _project_script() -> Path | None:
    """The script being run, if a record should be started for it: no run is in progress,
    recording is not switched off (`FIELDBOOK_RECORD=0`), and the program is a `.py` file of the
    project the working directory is in (not a tool installed in the environment)."""
    if os.environ.get("PLATFORM_RUN_ID"):  # fieldbook run, fb.init(), or a parent process
        return None
    if os.environ.get("FIELDBOOK_RECORD", "1").lower() in OFF:
        return None
    file = getattr(sys.modules.get("__main__"), "__file__", None)
    if not file or not file.endswith(".py") or sys.flags.interactive:
        return None  # a notebook, a shell, `python -c`, or a tool's entry point
    script = Path(file).resolve()
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        if script.is_relative_to(Path(prefix).resolve()):
            return None  # e.g. pytest, pip: installed in the environment
    root = _project_root(Path.cwd().resolve())
    if root is None or not script.is_relative_to(root):
        return None
    return script


def _project_root(start: Path) -> Path | None:
    """The folder of the nearest project file at or above `start`, read as text: the
    settings model is not imported until a run is known to start."""
    for folder in (start, *start.parents):
        if (folder / "fieldbook.toml").is_file():
            return folder
        pyproject = folder / "pyproject.toml"
        try:
            if pyproject.is_file() and "[tool.fieldbook]" in pyproject.read_text():
                return folder
        except OSError:
            pass
    return None
