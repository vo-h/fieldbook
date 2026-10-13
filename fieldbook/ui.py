"""Web UIs run in the foreground, as `mlflow ui` does: the command prints the
address, opens the browser, and runs until Ctrl-C (tracking.md §5)."""

import socket
import subprocess
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path

STARTUP_TIMEOUT = 120  # seconds; MLflow's first start creates its database


class UIError(Exception):
    pass


def serve(
    name: str,
    command: Callable[[int], list[str]],
    *,
    port: int,
    health_path: str,
    open_browser: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run the UI `command(port)` on the first free port from `port` until it exits or
    Ctrl-C. Returns its exit code."""
    from fieldbook.launch.local import wait

    port = _free_port(port)
    url = f"http://127.0.0.1:{port}"
    # Same process group as this command, so Ctrl-C in the terminal reaches the UI.
    process = subprocess.Popen(command(port), cwd=cwd, env=env)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not _healthy(url + health_path):
        if process.poll() is not None:
            return process.returncode
        if time.monotonic() > deadline:
            process.terminate()
            raise UIError(f"{name} did not start within {STARTUP_TIMEOUT} s")
        time.sleep(0.25)
    print(f"{name}: {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    return wait(process)


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise UIError(f"no free port in {start}-{start + 49}")
