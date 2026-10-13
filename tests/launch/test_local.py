"""`fieldbook run` on this machine (fieldbook/launch/local.py): scripts, tasks, signals, seeds."""

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
from helpers import (
    foreground_signals,
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook.code import environments, snapshot
from fieldbook.launch import local
from fieldbook.tracking import setup


def test_run_records_lineage_on_versions(repo: Path, gitleaks: Path) -> None:
    code, out = run_cli("run", "--seed", "7", "clean.py", "--min-size", "10")
    assert code == 0, out
    assert (repo / "args.txt").read_text() == "--min-size 10"
    info = fb.dataset("clean").log()[0]
    assert info.inputs == ("test/raw@1",)
    assert info.command == "clean.py --min-size 10"
    assert info.run_id and info.code == snapshot.tree_hash(repo)
    env = json.loads(info.env)
    assert env["python"] and env["seed"] == 7
    import pyarrow

    code, listed = run_cli("log", "packages", env["packages"][:10])
    assert code == 0 and f"pyarrow=={pyarrow.__version__}\n" in listed  # installed, not intended
    run_cli("run", "clean.py")  # same environment: the package list is stored once
    assert len(list(fb.storage.local_path("packages").iterdir())) == 1
    assert "no package list" in run_cli("log", "packages", "0000000")[1]
    with tarfile.open(fb.storage.local_path(f"code/{info.code}.tar.gz")) as tar:
        assert sorted(tar.getnames()) == [".gitignore", "clean.py"]  # ignored files excluded


def test_exit_code_is_passed_through(repo: Path, gitleaks: Path) -> None:
    assert run_cli("run", "clean.py", "3")[0] == 3


def test_task_file(repo: Path, gitleaks: Path) -> None:
    (repo / "task.yaml").write_text(
        "name: clean\nresources:\n  accelerators: A100\nenvs:\n  LIMIT: '5'\n"
        "setup: echo setup > setup.txt\nrun: python clean.py $LIMIT\n"
    )
    (repo / ".gitignore").write_text("data/\nargs.txt\nsetup.txt\n")
    code, out = run_cli("run", "task.yaml")
    assert code == 5 and "ignored locally: resources" in out
    assert (repo / "setup.txt").exists() and (repo / "args.txt").read_text() == "5"
    assert "arguments cannot be passed" in run_cli("run", "task.yaml", "x")[1]


WAITER = """import os, signal, sys, time
open("pid", "w").write(str(os.getpid()))
try:
    time.sleep(60)
except KeyboardInterrupt:
    open("interrupted", "w").write("1")
    sys.exit(130)
"""


def _start_waiter(repo: Path) -> tuple[subprocess.Popen, int]:
    import sys
    import time

    (repo / ".gitignore").write_text("data/\nargs.txt\npid\ninterrupted\n")
    (repo / "wait.py").write_text(WAITER)
    fieldbook_bin = Path(sys.executable).parent / "fieldbook"
    # A real `fieldbook` process: the MLflow stub does not apply, so name an MLflow server, for
    # which the launcher sets nothing up.
    env = {**os.environ, "MLFLOW_TRACKING_URI": "http://127.0.0.1:9"}
    proc = subprocess.Popen(
        [str(fieldbook_bin), "run", "wait.py"], cwd=repo, start_new_session=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        preexec_fn=foreground_signals,
    )  # fmt: skip
    for _ in range(300):
        if (repo / "pid").exists() and (repo / "pid").read_text():
            return proc, int((repo / "pid").read_text())
        time.sleep(0.1)
    proc.kill()
    raise AssertionError(f"script did not start: {proc.communicate()}")


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_terminating_fieldbook_run_stops_the_script(repo: Path, gitleaks: Path) -> None:
    import signal

    proc, child = _start_waiter(repo)
    proc.send_signal(signal.SIGTERM)  # sent to fieldbook only, as a scheduler would
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM, err
    assert _gone(child)


def test_ctrl_c_reaches_the_script_once(repo: Path, gitleaks: Path) -> None:
    import signal

    proc, child = _start_waiter(repo)
    os.killpg(proc.pid, signal.SIGINT)  # a terminal sends Ctrl-C to the process group
    try:
        _, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:  # say what was still running
        tree = subprocess.run(["ps", "-A", "-o", "pid,ppid,pgid,stat,etime,command"],
                              capture_output=True, text=True).stdout  # fmt: skip
        group = [line for line in tree.splitlines() if f" {proc.pid} " in line]
        proc.kill()
        raise AssertionError(
            f"Ctrl-C did not end the run; interrupted={(repo / 'interrupted').exists()}; "
            f"script {child}; processes:\n" + "\n".join(group)
        ) from None
    assert proc.returncode == 130 and (repo / "interrupted").exists()
    assert "Traceback" not in err and _gone(child)


def _fake_venv(path: Path, marker: Path) -> Path:
    """A virtual environment whose python records that it ran, then runs the real one."""
    import sys

    bin_dir = path / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text(
        f'#!/bin/sh\necho "$0" >> {marker}\nexec {sys.executable} "$@"\n'
    )
    (bin_dir / "python").chmod(0o755)
    return path


def test_scripts_run_in_the_project_venv(
    repo: Path, gitleaks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "used.txt"
    _fake_venv(repo / ".venv", marker)
    (repo / ".gitignore").write_text("data/\nargs.txt\n.venv/\n")
    code, out = run_cli("run", "clean.py")
    assert code == 0, out
    assert str(repo / ".venv" / "bin" / "python") in marker.read_text()
    assert json.loads(fb.dataset("clean").log()[0].env)["python"]

    active = _fake_venv(tmp_path / "active", marker)  # an activated venv takes precedence
    monkeypatch.setenv("VIRTUAL_ENV", str(active))
    marker.unlink()
    (repo / "task.yaml").write_text("run: python -c 'print(1)'\n")  # tasks get it on PATH
    assert run_cli("run", "task.yaml")[0] == 0
    assert str(active / "bin" / "python") in marker.read_text()


SEEDED = """import json, os, random, sys
import numpy  # imported after start: seeded on import
import helper  # a module next to the script, as with `python script.py`
from multiprocessing import get_context
def square(x):
    return x * x
if __name__ == "__main__":
    with get_context("spawn").Pool(2) as pool:  # spawn re-imports this script
        squares = pool.map(square, [1, 2, 3])
    print(json.dumps({"random": random.random(), "numpy": float(numpy.random.rand()),
                      "squares": squares, "argv": sys.argv, "helper": helper.VALUE,
                      "metrics": os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING")}),
          file=open("out.json", "w"))
"""


def _run_seeded(repo: Path, *options: str) -> dict:
    (repo / "jobs").mkdir(exist_ok=True)
    (repo / "jobs" / "seeded.py").write_text(SEEDED)
    (repo / "jobs" / "helper.py").write_text("VALUE = 42\n")
    (repo / ".gitignore").write_text("data/\nargs.txt\nout.json\n")
    code, out = run_cli("run", *options, "jobs/seeded.py", "--flag")
    assert code == 0, out
    return json.loads((repo / "out.json").read_text())


def test_seed_is_applied(repo: Path, gitleaks: Path) -> None:
    first = _run_seeded(repo, "--seed", "5")
    assert first == _run_seeded(repo, "--seed", "5")
    assert first["random"] != _run_seeded(repo, "--seed", "6")["random"]
    assert first["squares"] == [1, 4, 9] and first["helper"] == 42
    assert first["argv"] == ["jobs/seeded.py", "--flag"] and first["metrics"] == "true"


def test_deterministic_is_recorded(repo: Path, gitleaks: Path) -> None:
    assert run_cli("run", "--deterministic", "clean.py")[0] == 0
    assert json.loads(fb.dataset("clean").log()[0].env)["deterministic"] is True


def test_smoke_run(repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiments = []
    monkeypatch.setattr(
        setup, "mlflow_env", lambda cfg, experiment, *_: experiments.append(experiment) or {}
    )
    code, out = run_cli("run", "--smoke", "--sample", "1", "clean.py")
    assert code == 0, out
    assert "up to 1 items per folder" in out and experiments == ["test-smoke"]
    assert not fb.dataset("clean").exists  # nothing saved
    assert "--sample and --keep apply to --smoke" in run_cli("run", "--sample", "5", "clean.py")[1]


def test_run_inside_a_run_is_refused(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "outer.py").write_text(
        "import subprocess, sys\n"
        "fieldbook = __import__('pathlib').Path(sys.executable).parent / 'fieldbook'\n"
        "inner = subprocess.run([str(fieldbook), 'run', 'clean.py'],"
        " capture_output=True, text=True)\n"
        "open('inner.txt', 'w').write(inner.stderr)\n"
        "sys.exit(inner.returncode)\n"
    )
    code, out = run_cli("run", "outer.py")
    assert code == 1 and "a run cannot start another" in (repo / "inner.txt").read_text()
    assert not fb.dataset("clean").exists


# Stands in for torchrun: starts the module given with -m once per rank, as torchrun does.
FAKE_TORCHRUN = """#!{python}
import os, subprocess, sys
args = sys.argv[1:]
open("torchrun_args.txt", "w").write(" ".join(args))
module = args[args.index("-m") + 1:]
for rank in range(2):
    env = {{**os.environ, "RANK": str(rank)}}
    subprocess.run([sys.executable, "-u", "-m", *module], env=env, check=True)
"""


RANK_SCRIPT = """import os, random, sys
open(f"rank{os.environ['RANK']}.txt", "w").write(f"{random.random()} {sys.argv[1:]}")
"""


def test_torchrun_workers_get_the_seed(
    repo: Path, gitleaks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "torchrun").write_text(FAKE_TORCHRUN.format(python=sys.executable))
    (bin_dir / "torchrun").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    (repo / "ddp.py").write_text(RANK_SCRIPT)
    (repo / ".gitignore").write_text("data/\nargs.txt\n*.txt\n")
    code, out = run_cli(
        "run", "--seed", "7", "torchrun", "--nproc-per-node", "2", "ddp.py", "--lr", "1"
    )
    assert code == 0, out
    args = (repo / "torchrun_args.txt").read_text()
    assert args == "--nproc-per-node 2 -m fieldbook.launch.bootstrap ddp.py --lr 1"
    random.seed(7)
    expected = f"{random.random()} ['--lr', '1']"
    assert (repo / "rank0.txt").read_text() == (repo / "rank1.txt").read_text() == expected


def test_any_command_runs_and_unknown_ones_are_reported(repo: Path, gitleaks: Path) -> None:
    code, out = run_cli("run", "sh", "-c", 'echo "$PLATFORM_RUN_ID" > args.txt')
    assert code == 0, out
    assert (repo / "args.txt").read_text().strip().startswith("20")
    code, out = run_cli("run", "no-such-tool", "x")
    assert code == 1 and "no-such-tool: command not found" in out


@pytest.mark.parametrize(
    ("argv", "seeded"),
    [
        (
            ["torchrun", "--standalone", "t.py", "-x"],
            ["torchrun", "--standalone", "-m", "fieldbook.launch.bootstrap", "t.py", "-x"],
        ),
        (["torchrun", "-m", "pkg.train"], None),
        (["torchrun", "--no-python", "script.sh"], None),
    ],
)
def test_torchrun_rewrite(argv: list[str], seeded: list[str] | None) -> None:
    assert local._torchrun_through_bootstrap(argv) == seeded


def test_seed_needs_fieldbook_in_the_project_env(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = environments.query_interpreter
    monkeypatch.setattr(
        environments, "query_interpreter", lambda py: {**real(py), "fieldbook": False}
    )
    code, out = run_cli("run", "clean.py")
    assert code == 0 and "the seed is not applied" in out


LIGHTNING_ENV = """import json, os
json.dump({k: os.environ.get(k) for k in ("PL_GLOBAL_SEED", "PL_TRAINER_FAST_DEV_RUN")},
          open("pl_env.json", "w"))
"""


def test_lightning_reads_our_seed_and_smoke_limit(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments = []
    monkeypatch.setattr(
        setup, "mlflow_env", lambda cfg, experiment, *_: experiments.append(experiment) or {}
    )
    (repo / "pl.py").write_text(LIGHTNING_ENV)
    (repo / "fieldbook.toml").write_text('project = "test"\nexperiment = "from-file"\n')
    (repo / ".gitignore").write_text("data/\nargs.txt\npl_env.json\n")
    assert run_cli("run", "--seed", "3", "pl.py")[0] == 0
    assert json.loads((repo / "pl_env.json").read_text()) == {
        "PL_GLOBAL_SEED": "3",
        "PL_TRAINER_FAST_DEV_RUN": None,
    }
    assert run_cli("run", "--smoke", "pl.py")[0] == 0
    assert json.loads((repo / "pl_env.json").read_text())["PL_TRAINER_FAST_DEV_RUN"] == "5"
    assert experiments == ["from-file", "from-file-smoke"]  # experiment from the project file


WRITE_THEN_READ = """import fieldbook as fb
with fb.dataset("clean").edit() as e:
    e.copy_from(fb.dataset("raw"))
print("read back:", fb.dataset("clean").files().num_rows)
"""


def test_smoke_run_reads_its_own_output(repo: Path, gitleaks: Path) -> None:
    (repo / "wr.py").write_text(WRITE_THEN_READ)
    code, out = run_cli("run", "--smoke", "--sample", "1", "wr.py")
    assert code == 0, out
    assert not fb.dataset("clean").exists  # the real dataset is untouched
    assert not fb.storage.local_path("scratch").exists() or not any(
        fb.storage.local_path("scratch").iterdir()
    )  # scratch deleted
    code, out = run_cli("run", "--smoke", "--keep", "wr.py")
    assert code == 0 and "FIELDBOOK_SCRATCH=" in out
    assert len(list(fb.storage.local_path("scratch").iterdir())) == 1
    assert "--keep apply to --smoke" in run_cli("run", "--keep", "wr.py")[1]


def test_runs_record_where_the_code_sits_in_git(repo: Path, gitleaks: Path) -> None:
    """The snapshot reproduces the run; the commit, branch, and clean/dirty state say where
    it sits in git's history (runs.md §4)."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              cwd=repo, check=True, capture_output=True, text=True)  # fmt: skip
        return done.stdout.strip()

    assert snapshot.git_state(repo) == ""  # no commits yet
    git("add", "."), git("commit", "-qm", "first")
    commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    assert snapshot.git_state(repo) == f"{commit} {branch} clean"
    (repo / "new_config.yaml").write_text("lr: 0.1\n")  # untracked: not in any commit or diff
    code, out = run_cli("run", "clean.py")
    assert code == 0, out
    info = fb.dataset("clean").log()[0]
    assert info.git == f"{commit} {branch} dirty"
    assert snapshot.short_git(info.git) == f"{commit[:7]} {branch}+changes"
    with tarfile.open(fb.storage.local_path(f"code/{info.code}.tar.gz")) as tar:
        assert "new_config.yaml" in tar.getnames()  # the snapshot has what the commit lacks
    code, out = run_cli("data", "ls", "clean", "--lineage")
    assert f"git {commit[:7]} {branch}+changes" in out
