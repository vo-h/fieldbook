"""Whether a coding agent is the one running this (runs.md §20), so that runs, dataset
versions, and notes say so next to their author. The author stays the person: the agent
acts with their credentials and spends their money.

Bookkeeping, not a control: it reads variables the agent's shell sets, which anything can
set or unset. Standard library only: the MLflow plugin and the data module read it.
"""

import os

# variable an agent sets in the commands it runs -> the agent's name. Checked for each:
# Claude Code in its own shell; Codex in its source (exec_env.rs sets the session ID,
# shell_environment.rs the thread ID); Cursor on its forum, by its staff; Gemini CLI in its
# documentation of the shell tool.
KNOWN = {
    "CLAUDECODE": "claude-code",
    "CODEX_SESSION_ID": "codex",
    "CODEX_THREAD_ID": "codex",
    "CURSOR_AGENT": "cursor",
    "GEMINI_CLI": "gemini-cli",
}
OFF = ("", "0", "none", "false", "no")


def current() -> str:
    """The agent running this process, or "". `FIELDBOOK_AGENT` names one (or, set to `0`, says
    there is none); else a known agent's own variable. A run passes it on as `FIELDBOOK_AGENT`,
    so a cloud machine's side of the run knows too."""
    named = os.environ.get("FIELDBOOK_AGENT")
    if named is not None:
        return "" if named.strip().lower() in OFF else named.strip()
    return next((name for variable, name in KNOWN.items() if os.environ.get(variable)), "")


def who(author: str, agent: str) -> str:
    """`ada [claude-code]`: an author with the agent that acted for them."""
    return f"{author} [{agent}]" if agent else author
