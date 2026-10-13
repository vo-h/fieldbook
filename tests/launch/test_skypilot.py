"""SkyPilot launches (fieldbook/launch/skypilot.py), against a fake `sky`."""

import json
import sys
from pathlib import Path

import pytest

from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.launch import skypilot

FAKE_SKY = """#!{python}
import json, os, sys, time
log = os.environ["FAKE_SKY_LOG"]
entry = {{"args": sys.argv[1:], "secret": os.environ.get("TOKEN")}}
if sys.argv[1] == "launch":
    entry["task"] = open(sys.argv[2]).read()
open(log, "a").write(json.dumps(entry) + "\\n")
if sys.argv[1] == "launch" and os.environ.get("FAKE_SKY_HANG"):
    time.sleep(60)
"""


def test_launch_passes_values_by_name_and_tears_down_stuck_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    sky = tmp_path / "sky"
    sky.write_text(FAKE_SKY.format(python=sys.executable))
    sky.chmod(0o755)
    log = tmp_path / "sky.log"
    monkeypatch.setattr(skypilot, "executable", lambda: str(sky))
    monkeypatch.setenv("FAKE_SKY_LOG", str(log))
    spec = {"name": "c", "run": "echo hi"}
    skypilot.launch(spec, "c", secrets={"TOKEN": "s3cret"}, envs={}, detach=True, cfg=settings())
    (entry,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert entry["args"][2:] == ["-y", "-c", "c", "--down", "-d", "--secret", "TOKEN"]
    assert entry["secret"] == "s3cret" and "s3cret" not in entry["task"]  # never in the file
    assert not Path(entry["args"][1]).exists()  # task file removed
    log.unlink()
    monkeypatch.setenv("FAKE_SKY_HANG", "1")
    cfg = settings().model_copy(update={"launch_timeout": 0})  # 0 minutes: times out at once
    with pytest.raises(skypilot.SkyPilotError, match="timed out and was torn down"):
        skypilot.launch(spec, "c", secrets={}, envs={}, detach=True, cfg=cfg)
    calls = [json.loads(line)["args"] for line in log.read_text().splitlines()]
    assert calls[-1] == ["down", "-y", "c"]


FAKE_SKY_CONFIG = """#!{python}
import os, sys
open(os.environ["FAKE_SKY_LOG"], "w").write(open(os.environ["SKYPILOT_PROJECT_CONFIG"]).read())
"""


def test_launch_gives_the_cluster_its_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    sky = tmp_path / "sky"
    sky.write_text(FAKE_SKY_CONFIG.format(python=sys.executable))
    sky.chmod(0o755)
    monkeypatch.setattr(skypilot, "executable", lambda: str(sky))
    monkeypatch.setenv("FAKE_SKY_LOG", str(tmp_path / "config.yaml"))
    role = aws.Role("fieldbook-run-abc", "run machines", {})
    skypilot.launch({"run": "true"}, "fieldbook-r1", secrets={}, envs={}, detach=True,
                       cfg=settings(), role=role)  # fmt: skip
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config == {"aws": {"remote_identity": [{"fieldbook-r1": "fieldbook-run-abc"}]}}


def test_a_projects_skypilot_settings_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B8: the launch's own project config is merged into the project's `.sky.yaml`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".sky.yaml").write_text(
        "aws:\n  remote_identity:\n    - '*': my-role\n  vpc_name: lab\nallowed_clouds: [aws]\n"
    )
    ours = aws.skypilot_config(aws.Role("fieldbook-run-abc", "run machines", {}), "fieldbook-r1")
    assert skypilot.project_config(ours) == {
        "aws": {"remote_identity": [{"fieldbook-r1": "fieldbook-run-abc"}, {"*": "my-role"}],
                "vpc_name": "lab"},
        "allowed_clouds": ["aws"],
    }  # fmt: skip
