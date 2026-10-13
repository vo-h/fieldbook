"""Credentials (fieldbook/credentials.py): `.env`, a run's secrets, and masking."""

import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from helpers import (
    SECRET,
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook import credentials

SECRETS_SCRIPT = """import json, os, sys
import fieldbook as fb
json.dump({k: os.environ.get(k) for k in ("HF_TOKEN", "SHARED")}, open("secrets.json", "w"))
with fb.dataset("clean").edit() as e:
    e.copy_from(fb.dataset("raw"))
"""


def test_env_file_secrets(repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / ".env").write_text(f"HF_TOKEN={SECRET}\nSHARED=from-file\n")
    (repo / ".env.local").write_text("OTHER=1\n")
    (repo / ".env.example").write_text("HF_TOKEN=\n")  # a committed template
    (repo / "sec.py").write_text(SECRETS_SCRIPT)
    (repo / ".gitignore").write_text("data/\nargs.txt\nsecrets.json\n")  # .env not ignored
    monkeypatch.setenv("SHARED", "from-shell")  # the environment wins over .env
    code, out = run_cli("run", "sec.py", "--model", SECRET)
    assert code == 0, out
    assert "secrets from .env: HF_TOKEN, SHARED" in out and SECRET not in out
    assert json.loads((repo / "secrets.json").read_text()) == {
        "HF_TOKEN": SECRET,
        "SHARED": "from-shell",
    }
    info = fb.dataset("clean").log()[0]
    assert info.command == "sec.py --model '***'"  # the value is masked in the record
    with tarfile.open(fb.storage.local_path(f"code/{info.code}.tar.gz")) as tar:
        names = tar.getnames()
    assert ".env.example" in names and ".env" not in names and ".env.local" not in names

    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, check=True)
    assert "git tracks" in run_cli("run", "sec.py")[1]


def test_passwords_are_hidden_everywhere_in_addresses() -> None:
    uri = "postgresql://bob:pw1@db.example.com/mlflow?sslmode=require&password=pw2&token=t3"
    shown = credentials.display_uri(uri)
    assert "pw1" not in shown and "pw2" not in shown and "t3" not in shown
    assert "sslmode=require" in shown and "password=***" in shown
    assert credentials.redact(f"could not connect to {uri}", uri) == f"could not connect to {shown}"


def test_secrets_passed_by_name_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    from fieldbook import credentials

    monkeypatch.setenv("FIELDBOOK_SECRETS", "HF_TOKEN,PGPASSWORD")  # set by a cloud launch
    assert credentials.run_secret_names({"WANDB_API_KEY": "x"}) == (
        "HF_TOKEN", "PGPASSWORD", "WANDB_API_KEY",
    )  # fmt: skip
    command = credentials.mask_command(["train.py", "--hub", "hf_secretvalue"], ["hf_secretvalue"])
    assert command == "train.py --hub '***'"
