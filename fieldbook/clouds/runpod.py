"""RunPod: machine IDs and cost (cloud-runs.md §10)."""

import json
import os
import time
import tomllib
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

BILLING = "https://api.runpod.io/v2/billing/pods"
SETTLED = 2 * 3600  # seconds after a run ends: RunPod bills in hourly buckets


class RunPodError(Exception):
    pass


def machine() -> dict[str, str]:
    """MLflow tags naming this pod (`fieldbook.machine`)."""
    return {"fieldbook.machine": os.environ.get("RUNPOD_POD_ID", "")}


def cost(tags: dict[str, str], start: float, end: float) -> tuple[float, str] | None:
    """The amount billed for the pod (GPU, CPU, and disk); None until billing settles."""
    if time.time() - end < SETTLED:
        return None
    total = billed(tags["fieldbook.machine"], start)
    return (total, "runpod-billing") if total is not None else None


def billed(pod: str, start: float) -> float | None:
    """The amount billed for a pod, in USD; None if RunPod lists nothing for it yet."""
    since = datetime.fromtimestamp(start, UTC) - timedelta(days=1)
    until = datetime.now(UTC) + timedelta(days=1)
    query = urllib.parse.urlencode({
        "podId": pod, "bucketSize": "day",  # startTime needs endTime
        "startTime": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })  # fmt: skip
    # Cloudflare in front of the API refuses Python's default User-Agent (error 1010).
    headers = {"Authorization": f"Bearer {api_key()}", "User-Agent": "fieldbook"}
    request = urllib.request.Request(f"{BILLING}?{query}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    if not data.get("records"):
        return None
    return float(data["metadata"]["totals"]["totalAmount"])


def api_key() -> str:
    """The API key: RUNPOD_API_KEY, else ~/.runpod/config.toml (SkyPilot's `[default]
    api_key`, or runpodctl's top-level `apikey`)."""
    if key := os.environ.get("RUNPOD_API_KEY"):
        return key
    path = Path.home() / ".runpod" / "config.toml"
    try:
        config = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        config = {}
    if key := config.get("default", {}).get("api_key") or config.get("apikey"):
        return key
    raise RunPodError("no RunPod API key (RUNPOD_API_KEY or ~/.runpod/config.toml)")
