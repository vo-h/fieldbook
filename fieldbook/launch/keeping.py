"""What `fieldbook run` keeps of a run while it goes and when it ends (runs.md §18): its record
(started, alive, ended), its output, and the files it wrote. One thread, one tick every
half minute. Keeping never stops or fails a run: an error here is reported once."""

import threading
import time
from pathlib import Path
from typing import Any

from fieldbook.config import Settings
from fieldbook.launch import note, outputs, records, runlog

FILES_EVERY = 10  # ticks between copies of the run's files while it goes (cloud machines)


class Keeper:
    def __init__(self, run_id: str, cfg: Settings, record: dict[str, Any], *, repo: Path,
                 output_dir: Path, secrets: tuple[str, ...] = (), on_machine: bool = False,
                 files: bool = True) -> None:  # fmt: skip
        self.run_id, self.cfg, self.record = run_id, cfg, record
        self.on_machine = on_machine  # a cloud machine: the laptop wrote the start record,
        # and files are copied while the run goes, since the machine may vanish
        self.log = runlog.RunLog(run_id, cfg, secrets=secrets)
        roots = {"project": repo, "output": output_dir}
        self.files = outputs.Collector(run_id, cfg, roots, time.time()) if files else None
        self._began = time.time()  # of the run's own process, after any launch
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fieldbook-keep", daemon=True)
        self._warned = False

    def start(self) -> None:
        self._safely(records.start, self.cfg, self.run_id, self.record, keep=self.on_machine)
        self._safely(records.beat, self.cfg, self.run_id)
        if self.on_machine:  # which machine this is: what its cost is looked up by
            from fieldbook.launch import budget
            from fieldbook.tracking import costs

            self._safely(lambda: budget.machine(self.run_id, costs.machine_tags(), self.cfg))
        self._thread.start()

    def _loop(self) -> None:
        tick = 0
        while not self._stop.wait(records.BEAT):
            tick += 1
            self._safely(self.log.flush)
            self._safely(records.beat, self.cfg, self.run_id)
            if self.files and self.on_machine and tick % FILES_EVERY == 0:
                self._safely(self.files.sync)

    def finish(self, exit_code: int) -> None:
        self._stop.set()
        self._thread.join(timeout=records.BEAT)
        self._safely(self.log.close)
        kept = self._safely(self.files.sync) if self.files else None
        self._safely(records.end, self.cfg, self.run_id, exit_code,
                     seconds=round(time.time() - self._began, 1))  # fmt: skip
        if kept and kept["left_out"]:
            names = ", ".join(f"{f['path']} ({f['size'] / outputs.MB:,.0f} MB, {f['why']})"
                              for f in kept["left_out"][:5])  # fmt: skip
            more = f", and {len(kept['left_out']) - 5} more" if len(kept["left_out"]) > 5 else ""
            note(f"files not kept: {names}{more} (`fieldbook log files {self.run_id}` lists them; "
                 "output_max_file_mb and output_max_total_mb set the limits)")  # fmt: skip

    def _safely(self, action, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        try:
            return action(*args, **kwargs)
        except Exception as e:
            if not self._warned:
                self._warned = True
                note(f"warning: this run's record or output is not being kept: "
                     f"{type(e).__name__}: {e}")  # fmt: skip
            return None
