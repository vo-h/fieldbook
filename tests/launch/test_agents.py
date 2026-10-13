"""Runs, versions, and notes say when a coding agent made them (fieldbook/launch/agents.py)."""

import json
from pathlib import Path

import pytest
from helpers import (
    no_mlflow,  # noqa: F401 (autouse: skips MLflow setup unless marked mlflow)
    run_cli,
)

import fieldbook as fb
from fieldbook import history, notes
from fieldbook.config import settings
from fieldbook.data import lineage
from fieldbook.launch import agents, runenv


def test_an_agent_is_known_by_its_shells_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert agents.current() == ""
    monkeypatch.setenv("CLAUDECODE", "1")
    assert agents.current() == "claude-code"
    monkeypatch.setenv("FIELDBOOK_AGENT", "my-agent")  # named: any agent, or one's own label
    assert agents.current() == "my-agent"
    monkeypatch.setenv("FIELDBOOK_AGENT", "0")  # a person working in the agent's terminal
    assert agents.current() == ""
    assert agents.who("alice", "") == "alice" and agents.who("alice", "x") == "alice [x]"


@pytest.mark.parametrize(
    ("variable", "agent"),
    [("CODEX_SESSION_ID", "codex"), ("CODEX_THREAD_ID", "codex"), ("CURSOR_AGENT", "cursor"),
     ("GEMINI_CLI", "gemini-cli")],
)  # fmt: skip
def test_other_agents_are_known_by_theirs(
    monkeypatch: pytest.MonkeyPatch, variable: str, agent: str
) -> None:
    monkeypatch.setenv(variable, "1")
    assert agents.current() == agent


def test_an_agents_runs_versions_and_notes_carry_its_name(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_cli("run", "clean.py")[0] == 0  # by hand
    monkeypatch.setenv("CLAUDECODE", "1")
    assert runenv.settings_env(settings())["FIELDBOOK_AGENT"] == "claude-code"  # reaches the run
    assert lineage.run_tags()["fieldbook.agent"] == "claude-code"
    (repo / "more.py").write_text(
        "import fieldbook as fb\nwith fb.dataset('extra').edit() as e:\n"
        "    e.copy_from(fb.dataset('raw'))\n"
    )
    assert run_cli("run", "more.py")[0] == 0
    by_agent, by_hand = history.runs(settings())
    assert history.with_details([by_agent, by_hand])
    assert (by_agent.agent, by_hand.agent) == ("claude-code", "")
    assert by_agent.author == by_hand.author  # the author stays the person
    (version,) = fb.dataset("extra").log()
    assert version.agent == "claude-code" and fb.dataset("clean").log()[0].agent == ""
    notes.add(by_agent.run_id, "the agent's conclusion", verdict="good")
    out = run_cli("log")[1].splitlines()
    assert f"{by_agent.author} [claude-code]" in out[1] and "[claude-code]" not in out[2]
    shown = json.loads(run_cli("log", "--json")[1])
    assert shown[0]["agent"] == "claude-code" and shown[0]["notes"][0]["agent"] == "claude-code"
    assert (
        "[claude-code] [good]  the agent's conclusion" in run_cli("log", "note", by_agent.run_id)[1]
    )
    only_agent = run_cli("log", "--agent")[1]
    assert by_agent.run_id in only_agent and by_hand.run_id not in only_agent
    assert by_hand.run_id in run_cli("log", "--no-agent")[1]
    assert "[claude-code]" in run_cli("data", "ls", "extra", "--versions")[1]
