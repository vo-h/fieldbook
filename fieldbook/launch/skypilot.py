"""SkyPilot, through its CLI (`sky`): launching the machines of image builds and runs, with
their IAM role on AWS (cloud-runs.md §7, §12)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from fieldbook.clouds import aws
from fieldbook.config import Settings
from fieldbook.launch import LaunchError, note, runlog


class SkyPilotError(LaunchError):
    pass


def cloud_name(infra: str | None) -> str | None:
    """`aws` from `aws`, `aws/us-west-2`, or `AWS`; None if not given."""
    return (str(infra).split("/", 1)[0].strip().lower() or None) if infra else None


def machine_role(cfg: Settings, cloud: str, *, build: bool = False) -> "aws.Role | None":
    """The machine's IAM role on AWS (None where roles do not apply); it must exist."""
    if cloud != "aws" or not aws.roles_apply(cfg):
        return None
    run_role, build_role = aws.roles(cfg)
    role = build_role if build else run_role
    found = aws.role_state(role)
    if found == "missing":
        raise SkyPilotError(
            f"IAM role {role.name} does not exist: run `fieldbook init` to create it"
        )
    if found == "outdated":
        # Not a warning: the role's policy names what a machine may reach and which
        # machines it may delete, so with an old one a paid machine can outlive its run.
        raise SkyPilotError(
            f"IAM role {role.name} no longer matches the settings, so a machine might not "
            "reach the storage or delete itself: run `fieldbook init` to update it"
        )
    if found == "unknown":
        note(f"warning: cannot read IAM to check role {role.name}; launching with it anyway")
    return role


def candidates(resources: dict, nodes: int) -> list[tuple[str, float]] | None:
    """The machines SkyPilot may launch for `resources`, as (GPU or instance type, USD an
    hour for all nodes), cheapest first, from its catalog (no cloud credentials needed).
    One per option: a memory size (`24GB+`) or a list of GPUs gives several, and SkyPilot
    launches the cheapest that has capacity. Empty when nothing fits; None when the catalog
    cannot say (an unknown name, no catalog), which `sky launch` then reports.

    Prices are estimates: within an option, a machine without capacity may be replaced by
    the next size, and regions differ slightly."""
    import contextlib
    import io

    found = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # catalog updates print progress
            import sky

            for option in sky.Resources.from_yaml_config(dict(resources)):
                fits = option.cloud.get_feasible_launchable_resources(option, num_nodes=nodes)
                priced = [(r.get_cost(3600) * nodes, r) for r in fits.resources_list]
                if priced:
                    price, cheapest = min(priced, key=lambda item: item[0])
                    name = next(iter(cheapest.accelerators or {}), cheapest.instance_type)
                    found.append((str(name), price))
    except Exception:
        return None
    return sorted(found, key=lambda item: item[1])


PROJECT_CONFIG = ".sky.yaml"  # a project's own SkyPilot settings, in the working directory


def project_config(ours: dict) -> dict:
    """`ours` merged into the project's `.sky.yaml`: SkyPilot reads only one project file,
    and pointing it at ours would otherwise drop the project's settings."""
    path = Path.cwd() / PROJECT_CONFIG
    theirs = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    return _merge(theirs, ours)


def _merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = value + merged[key]  # remote_identity: first match wins, so ours first
        else:
            merged[key] = value
    return merged


def executable() -> str:
    candidates = [Path(sys.executable).parent / "sky", shutil.which("sky")]
    found = next((str(c) for c in candidates if c and Path(c).exists()), None)
    if found is None:
        raise SkyPilotError("cloud runs need SkyPilot: pip install 'fieldbook[aws]' (or "
                            "'fieldbook[runpod]'), then `sky check`")  # fmt: skip
    return found


def launch(spec: dict, cluster: str, *, secrets: dict, envs: dict, detach: bool,
           cfg: Settings, role: "aws.Role | None" = None,
           log: "runlog.RunLog | None" = None) -> None:  # fmt: skip
    """`sky launch` with values passed by name only (never written to files or command
    lines), deleted when the job ends, and torn down if it has not finished launching within
    `launch_timeout` (design §12)."""
    sky = executable()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="fieldbook-task-", delete=False
    ) as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    command = [sky, "launch", f.name, "-y", "-c", cluster, "--down"]
    if detach:
        command.append("-d")
    for name in envs:
        command += ["--env", name]
    for name in secrets:
        command += ["--secret", name]
    env = {**os.environ, **envs, **secrets}
    if role is not None:  # the machine's IAM role instead of copied credentials
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix="fieldbook-sky-", delete=False
        ) as c:
            yaml.safe_dump(project_config(aws.skypilot_config(role, cluster)), c)
        env["SKYPILOT_PROJECT_CONFIG"] = c.name
    limit = cfg.launch_timeout * 60
    try:
        copied = runlog.popen(command, log, env=env)  # `log`: a run's launch log (runs.md §18)
        process = copied.process
        try:
            # A build waits for its job too: allow it 30 minutes after launching.
            code = process.wait(timeout=limit if detach else limit + 1800)
        except subprocess.TimeoutExpired:
            process.terminate()
            note(f"launch of {cluster} did not finish within {cfg.launch_timeout} minutes "
                  "(launch_timeout); tearing it down")  # fmt: skip
            subprocess.run([sky, "down", "-y", cluster])
            raise SkyPilotError(f"launch of {cluster} timed out and was torn down") from None
    finally:
        Path(f.name).unlink(missing_ok=True)
        if role is not None:
            Path(c.name).unlink(missing_ok=True)
    copied.drain()
    if code != 0:
        subprocess.run([sky, "down", "-y", cluster], capture_output=True)
        raise SkyPilotError(f"sky launch failed for {cluster} (output above); cluster torn down")
