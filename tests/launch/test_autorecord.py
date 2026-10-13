"""A project's script started on its own is recorded when it imports fieldbook
(fieldbook/launch/autorecord.py, runs.md §13)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import fieldbook as fb
from fieldbook.code import snapshot

SCRIPT = """import random
import fieldbook as fb

first = random.random()
with fb.dataset("cleaned").edit(message="plain python") as edit:
    edit.add("a.txt", "raw/a.txt")
print("FIRST", first)
"""


@pytest.fixture
def project(repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The repository as a project (it has a project file), with a script and a data file."""
    (repo / "fieldbook.toml").write_text('project = "test"\n')
    (repo / "raw").mkdir()
    (repo / "raw" / "a.txt").write_text("one")
    (repo / "clean2.py").write_text(SCRIPT)
    (repo / ".gitignore").write_text("data/\nargs.txt\nraw/\n")
    # A real process: the MLflow stub does not apply, so name a server (nothing is set up).
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    return repo


def python(*args: str, cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **env})  # fmt: skip


def test_plain_python_is_recorded_like_fieldbook_run(project: Path) -> None:
    done = python("clean2.py", cwd=project)
    assert done.returncode == 0, done.stderr
    info = fb.dataset("cleaned").log()[0]
    assert info.run_id and info.run_id in done.stderr  # "run <id>  code ...  seed ..."
    assert info.code == snapshot.tree_hash(project)
    assert '"seed"' in info.env and info.command == "clean2.py"


def test_the_seed_is_applied_before_the_scripts_own_code(project: Path) -> None:
    (project / "seeded.py").write_text(
        "import os, random\nimport fieldbook\nprint(os.environ['PLATFORM_SEED'], random.random())\n"
    )
    seed, value = python("seeded.py", cwd=project).stdout.split()
    import random

    random.seed(int(seed))
    assert float(value) == random.random()


def test_recording_can_be_switched_off(project: Path) -> None:
    done = python("clean2.py", cwd=project, FIELDBOOK_RECORD="0")
    assert done.returncode == 0, done.stderr
    info = fb.dataset("cleaned").log()[0]
    assert info.run_id == "" and info.code == "" and "run " not in done.stderr


def test_fieldbook_run_and_its_workers_keep_one_run(project: Path) -> None:
    """Whoever starts first owns the run: importing fieldbook inside it joins it."""
    (project / "worker.py").write_text(
        "import os, subprocess, sys\nimport fieldbook\n"
        "print('RUN', os.environ['PLATFORM_RUN_ID'])\n"
        "if len(sys.argv) == 1:\n"
        "    subprocess.run([sys.executable, __file__, 'child'], check=True)\n"
    )
    done = python("worker.py", cwd=project)
    runs = [line.split()[1] for line in done.stdout.splitlines() if line.startswith("RUN")]
    assert len(runs) == 2 and runs[0] == runs[1], done.stdout + done.stderr
    assert done.stderr.count("run ") == 1  # one record, announced once


def test_tools_and_scripts_outside_a_project_are_left_alone(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.py").write_text(
        "import os\nimport fieldbook\nprint(os.environ.get('PLATFORM_RUN_ID'))"
    )
    assert python("x.py", cwd=outside).stdout.strip() == "None"  # no project there
    code = "import os, fieldbook; print(os.environ.get('PLATFORM_RUN_ID'))"
    assert python("-c", code, cwd=project).stdout.strip() == "None"  # not a script
    done = python("-m", "pytest", "--version", cwd=project)  # a tool in the environment
    assert done.returncode == 0 and "run " not in done.stderr


def test_a_run_that_cannot_be_recorded_still_runs(project: Path) -> None:
    """E.g. gitleaks missing: say so, and let the person's own command go ahead."""
    path = "/usr/bin:/bin"  # git, but no gitleaks
    if any((Path(p) / "gitleaks").exists() for p in path.split(":")):
        pytest.skip("gitleaks is installed in a system folder")
    done = python("clean2.py", cwd=project, PATH=path)
    assert done.returncode == 0, done.stderr
    assert "fieldbook: this run is not recorded: gitleaks is required" in done.stderr
    assert fb.dataset("cleaned").log()[0].run_id == ""


def test_a_script_that_launches_runs_is_told_how(project: Path) -> None:
    (project / "driver.py").write_text(
        "import subprocess, sys\nimport fieldbook\n"
        "sys.exit(subprocess.run([sys.argv[1], 'run', 'clean2.py']).returncode)\n"
    )
    fieldbook = str(Path(sys.executable).parent / "fieldbook")
    refused = python("driver.py", fieldbook, cwd=project)
    assert refused.returncode != 0 and "FIELDBOOK_RECORD=0" in refused.stderr
    assert python("driver.py", fieldbook, cwd=project, FIELDBOOK_RECORD="0").returncode == 0
    assert fb.dataset("cleaned").log()[0].run_id  # the step's own run
