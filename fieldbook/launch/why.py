"""`fieldbook log why RUN`: how a run ended, and if it failed, the cause (runs.md §19), from
its record and its kept output (§18). The causes named are the ones that can be told
reliably; anything else is shown as the last lines of the output."""

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from fieldbook.config import Settings
from fieldbook.launch import budget, records, runlog

SHOWN = 15  # lines of output given when no cause is recognised
SIGNALS = {130: "interrupted (Ctrl-C)", 143: "terminated (SIGTERM)", 137: "killed (SIGKILL)"}


@dataclass
class Verdict:
    run_id: str
    status: str
    cause: str  # one line
    exit_code: int | None = None
    seconds: float | None = None  # how long it ran
    detail: list[str] = field(default_factory=list)  # the lines of output that show it

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def why(run_id: str, cfg: Settings) -> Verdict:
    state = records.states(cfg, run_id).get(run_id)
    if state is None:
        cause = (
            "no record of how it ran: it was not started with `fieldbook run` "
            "(a script started with plain python keeps none)"
        )
        return Verdict(run_id, "", cause)
    status = state.status()
    start = records.read(cfg, [run_id]).get(run_id, {})
    began = start.get("started") or records.started_at(run_id)
    end = records.read_end(cfg, state) if state.exit_code is not None else {}
    # how long its process ran, as its own machine measured it; else since it was started,
    # which for a cloud run includes the launch
    seconds = end.get("seconds", round(state.ended - began, 1) if state.ended else None)
    verdict = Verdict(run_id, status, "", state.exit_code, seconds)
    output = runlog.read(run_id, cfg) or []
    if status == records.FINISHED:
        verdict.cause = "it succeeded"
    elif status == records.RUNNING:
        verdict.cause, verdict.detail = "it is still running", output[-5:]
    elif status == records.LAUNCHING:
        verdict.cause = "its machine is still being launched"
        verdict.detail = (runlog.read(run_id, cfg, name="launch") or [])[-5:]
    elif status == records.KILLED:
        verdict.cause = ("it stopped without a record of ending: the process or its machine "
                         "was removed from outside (a reclaimed or deleted machine, kill -9, "
                         "power)")  # fmt: skip
        verdict.detail = output[-SHOWN:]
    elif end.get("phase") == "launch":
        launch = runlog.read(run_id, cfg, name="launch") or []
        verdict.cause, verdict.detail = _launch_cause(launch)
    else:
        verdict.cause, verdict.detail = _cause(state.exit_code, verdict.seconds,
                                               start.get("limit"), output)  # fmt: skip
    return verdict


def _launch_cause(lines: list[str]) -> tuple[str, list[str]]:
    text = "\n".join(lines)
    if re.search(r"ResourcesUnavailableError|Failed to (provision|acquire)|no capacity|"
                 r"InsufficientInstanceCapacity", text, re.I):  # fmt: skip
        cause = "the launch failed: no machine with the requested resources was available"
    elif re.search(r"did not finish within|timed out", text, re.I):
        cause = "the launch did not finish in time (launch_timeout) and was torn down"
    elif re.search(r"pull access denied|manifest unknown|ImagePull|docker", text, re.I):
        cause = "the launch failed while fetching the environment image"
    else:
        cause = "the launch failed before the run started"
    return cause, lines[-SHOWN:]


def _cause(exit_code: int | None, seconds: float | None, limit: int | None,
           lines: list[str]) -> tuple[str, list[str]]:  # fmt: skip
    text = "\n".join(lines[-2000:])
    if limit and seconds and seconds >= limit - 60 and exit_code in (124, 137, 143):
        return f"it reached its time limit ({budget.span(limit)}) and was stopped", lines[-5:]
    if found := _last(r"(torch\.)?OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED",
                      lines):  # fmt: skip
        return "it ran out of GPU memory", _around(lines, found)
    if found := _last(r"^MemoryError|Cannot allocate memory|std::bad_alloc", lines):
        return "it ran out of memory", _around(lines, found)
    if match := re.search(r"ModuleNotFoundError: No module named '([^']+)'", text):
        return (f"the package {match.group(1)!r} is not installed in the run's environment",
                _traceback(lines))  # fmt: skip
    if found := _last(r"^FileNotFoundError|No such file or directory", lines):
        return "a file it needs was not found", _traceback(lines) or _around(lines, found)
    if trace := _traceback(lines):
        return f"it raised {trace[-1].strip()}", trace
    if exit_code == 137:
        return ("it was killed (SIGKILL), usually by the system when memory runs out",
                lines[-SHOWN:])  # fmt: skip
    if exit_code in SIGNALS:
        return f"it was {SIGNALS[exit_code]}", lines[-5:]
    return f"it exited with code {exit_code}", lines[-SHOWN:]


def _last(pattern: str, lines: list[str]) -> int | None:
    """The index of the last line matching `pattern`, or None (0 is a valid index)."""
    matcher = re.compile(pattern)
    for index in range(len(lines) - 1, max(len(lines) - 2000, 0) - 1, -1):
        if matcher.search(lines[index]):
            return index + 1  # so that a match on the first line is truthy
    return None


def _around(lines: list[str], found: int) -> list[str]:
    return lines[max(found - 4, 0) : found + 1]


def _traceback(lines: list[str]) -> list[str]:
    """The last Python traceback: its header, its last two frames, and the error."""
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback (most recent call")]
    if not starts:
        return []
    block = lines[starts[-1] :]
    end = next((i for i, line in enumerate(block[1:], 1) if line and not line.startswith(" ")),
               len(block) - 1)  # fmt: skip
    block = block[: end + 1]
    frames = [i for i, line in enumerate(block) if line.lstrip().startswith("File ")]
    body = block[frames[-2] : -1] if len(frames) >= 2 else block[1:-1]
    return [block[0], *body, block[-1]]
