"""The MLflow UI (tracking.md §5): `fieldbook log ui` runs MLflow's server in the
foreground over the local store that runs log to; Ctrl-C stops it. Runs do not need it."""

import os
from pathlib import Path

from fieldbook import credentials, ui
from fieldbook.config import Settings
from fieldbook.tracking import store


def serve(cfg: Settings, python: str, *, open_browser: bool) -> int:
    """Run the MLflow UI with `python` (which needs full MLflow) until Ctrl-C."""
    uri = store.tracking_uri(cfg)  # this machine's database, or a shared one (tracking.md §1)
    if not store.is_database(uri):
        raise ui.UIError(
            f"MLFLOW_TRACKING_URI names an MLflow server ({credentials.display_uri(uri)}); "
            "open it in a browser"
        )
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    def command(port: int) -> list[str]:
        return [
            python, "-m", "mlflow", "server",
            "--default-artifact-root", store.artifact_root(cfg),
            "--no-serve-artifacts",  # artifacts are read from the storage root directly
            "--host", "127.0.0.1", "--port", str(port),
            "--workers", "1",  # one user; the default of 4 uses about 2 GB of memory
        ]  # fmt: skip

    env = {
        # Background jobs serve MLflow's GenAI features (LLM scorers) and start 8
        # processes (about 750 MB); viewing runs does not need them.
        "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
        **os.environ,  # an explicit setting wins
        "MLFLOW_BACKEND_STORE_URI": uri,  # not an argument: it may hold a password
    }
    return ui.serve(
        "MLflow",
        command,
        port=cfg.mlflow_port,  # MLflow's default, 5000, is taken by AirPlay on macOS
        health_path="/health",
        open_browser=open_browser,
        env=env,
    )
