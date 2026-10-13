"""The run context: what a run passes to its processes as `PLATFORM_*` variables
(runs.md §7). `fieldbook run` and `fb.init()` write it; the data module, the MLflow
plugin, and the bootstrap read it with `current()`. Standard library only: the bootstrap
imports it before the script runs."""

import os
from dataclasses import dataclass

SMOKE_BATCHES = 5  # Lightning stops after this many batches in smoke runs


@dataclass(frozen=True)
class RunContext:
    run_id: str = ""  # empty outside a run
    code: str = ""  # code snapshot hash
    git: str = ""  # `<commit> <branch> clean|dirty` when the snapshot was taken (runs.md §4)
    env: str = ""  # environment record, JSON (runs.md §5)
    seed: int | None = None
    secrets: tuple[str, ...] = ()  # names of the run's .env secrets, for masking
    deterministic: bool = False
    smoke: int | None = None  # items per folder in a smoke run
    scratch: str | None = None  # a smoke run's scratch area

    def to_env(self) -> dict[str, str]:
        """The `PLATFORM_*` variables, plus the framework variables that carry the same
        settings (Lightning reads `PL_GLOBAL_SEED` and `PL_TRAINER_FAST_DEV_RUN`)."""
        env = {
            "PLATFORM_RUN_ID": self.run_id,
            "PLATFORM_CODE": self.code,
            "PLATFORM_GIT": self.git,
            "PLATFORM_ENV": self.env,
            "PLATFORM_SECRETS": ",".join(sorted(self.secrets)),
        }
        if self.seed is not None:
            env |= {"PLATFORM_SEED": str(self.seed), "PL_GLOBAL_SEED": str(self.seed)}
        if self.deterministic:
            env["PLATFORM_DETERMINISTIC"] = "1"
        if self.smoke:
            env["PLATFORM_SMOKE"] = str(self.smoke)
            env["PL_TRAINER_FAST_DEV_RUN"] = str(SMOKE_BATCHES)
        if self.scratch:
            env["PLATFORM_SCRATCH"] = self.scratch
        return {k: v for k, v in env.items() if v}


def current() -> RunContext:
    """The context of the run this process belongs to; empty fields outside a run. A step of
    a pipeline smoke run has `smoke` and `scratch` but no run ID until `fieldbook run` starts it."""
    get = os.environ.get
    seed = get("PLATFORM_SEED")
    return RunContext(
        run_id=get("PLATFORM_RUN_ID", ""),
        code=get("PLATFORM_CODE", ""),
        git=get("PLATFORM_GIT", ""),
        env=get("PLATFORM_ENV", ""),
        seed=int(seed) if seed else None,
        secrets=tuple(filter(None, get("PLATFORM_SECRETS", "").split(","))),
        deterministic=get("PLATFORM_DETERMINISTIC") == "1",
        smoke=int(get("PLATFORM_SMOKE") or 0) or None,
        scratch=get("PLATFORM_SCRATCH") or None,
    )
