"""`python -m fieldbook.launch.bootstrap SCRIPT [ARGS...]`: runs SCRIPT as `python SCRIPT ARGS`
would, after applying the run's seed and arranging MLflow autologging. `fieldbook run` starts
Python scripts this way (runs.md §8, tracking.md §8)."""

import runpy
import sys
from pathlib import Path

from fieldbook.launch import autolog, context, seeding


def main() -> None:
    script, *args = sys.argv[1:]
    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)  # as `python SCRIPT` sets it
    run = context.current()
    if run.seed is not None:
        seeding.apply(run.seed, deterministic=run.deterministic)
    autolog.enable()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
