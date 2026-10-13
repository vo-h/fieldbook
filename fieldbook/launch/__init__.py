"""Starting runs (runs.md), and what happens inside a run's process.

Starting: task files (`tasks`), `fieldbook run` on this machine (`local`), `fb.init()` in the
current process (`inprocess`), `fieldbook run --on remote` (`cloud`), and SkyPilot (`skypilot`).
Each prepares the run the same way: a code snapshot, an environment record, and the
variables from `runenv`.

Inside the run: the run context its processes read (`context`), the bootstrap that starts
Python scripts (`bootstrap`), and seeding (`seeding`). These three use only the standard
library and must stay that way: the bootstrap starts every run and every `torchrun` rank.
This module imports nothing else from fieldbook.
"""

import secrets
import sys
from datetime import UTC, datetime


class LaunchError(Exception):
    pass


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def note(message: str) -> None:
    """A status line for the person launching, on stderr."""
    print(message, file=sys.stderr, flush=True)
