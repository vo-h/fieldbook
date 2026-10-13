"""Helpers shared by tests in several folders (fixtures shared that way live in conftest.py)."""

import os
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fieldbook.cli import app
from fieldbook.tracking import setup

runner = CliRunner()


def foreground_signals() -> None:
    """Ctrl-C as a terminal's foreground job gets it. A test run started in the background
    of a non-interactive shell (`pytest &`) has SIGINT ignored, and every process it starts
    inherits that, Python included, so a test that sends SIGINT would wait forever."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_foreground_ui(*args: str, path: str = "") -> str:
    """Run `fieldbook <args>` as in a terminal, wait for the address it prints, check `path` on it,
    then press Ctrl-C (SIGINT to the process group). Returns the address once nothing is left
    running and the command exited cleanly."""
    fieldbook = str(Path(sys.executable).parent / "fieldbook")
    process = subprocess.Popen(
        [fieldbook, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,  # its own process group, like a terminal's foreground job
        preexec_fn=foreground_signals,
    )  # fmt: skip
    output, url = "", None
    try:
        deadline = time.monotonic() + 120
        while url is None and time.monotonic() < deadline and process.poll() is None:
            if select.select([process.stdout], [], [], 1)[0]:
                line = process.stdout.readline()
                output += line
                if "(Ctrl-C to stop)" in line:
                    url = line.split(": ", 1)[1].split()[0]
        output += "" if url else process.stdout.read() if process.poll() is not None else ""
        assert url, f"no address printed (exit code {process.returncode}):\n{output}"
        with urllib.request.urlopen(url + path, timeout=10) as response:
            assert response.status == 200
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        code = process.wait(timeout=60)
    assert code in (0, 128 + signal.SIGINT), (code, output + process.stdout.read())
    _wait_until_closed(url)
    deadline = time.monotonic() + 15  # child processes may exit just after
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return url
        time.sleep(0.2)
    os.killpg(process.pid, signal.SIGKILL)
    raise AssertionError(f"processes left running after Ctrl-C:\n{output}")


@pytest.fixture(autouse=True)
def no_mlflow(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip MLflow setup (slow on a fresh store) except in tests marked `mlflow`."""
    if request.node.get_closest_marker("mlflow") is None:
        monkeypatch.setattr(setup, "mlflow_env", lambda cfg, experiment, *_: {})


SCRIPT = """import sys
import fieldbook as fb
raw = fb.dataset("raw", version=1)
with fb.dataset("clean").edit("clean") as e:
    e.copy_from(raw)
open("args.txt", "w").write(" ".join(sys.argv[1:]))
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0)
"""


def run_cli(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


SECRET = "hf_secret_value_123"


def _wait_until_closed(url: str, timeout: float = 15) -> None:
    """Wait until nothing listens at `url` any more (the UI's server processes are gone),
    instead of sleeping a fixed time (D5)."""
    host, port = url.removeprefix("http://").split("/", 1)[0].split(":")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex((host, int(port))) != 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"still listening at {url} after {timeout} s")


def worker_port(slot: int) -> int:
    """A port no other parallel test worker uses, for servers that cannot bind port 0 and
    report it (D5): each pytest-xdist worker gets its own block of 200 ports."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw") or "0"
    return 20000 + 200 * int(worker) + slot
