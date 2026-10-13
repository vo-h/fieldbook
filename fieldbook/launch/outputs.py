"""The files a run wrote, kept (runs.md §18): `train.log`, TensorBoard files, a config
tool's output folder, plots. On a cloud machine they would otherwise go with the machine.

Kept: every file under the repository that is new or changed since the run started,
whether or not git or `.fieldbookignore` ignores it (ignored folders are where outputs go), and
everything in the run's output folder (`PLATFORM_OUTPUT_DIR`), which is the place for
files a script would otherwise write elsewhere on the machine: those cannot be found.

    runs/<run ID>/files/project/<path in the repository>
    runs/<run ID>/files/output/<path in the output folder>
    runs/<run ID>/files.json      what was kept, and what was left out and why

Nothing is left out without being named: a file over `output_max_file_mb`, or past the
run's total (`output_max_total_mb`, MAX_FILES), is listed with its size.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from fieldbook import storage
from fieldbook.config import LOCAL_DIR, Settings
from fieldbook.launch import records

# Never a run's outputs: version control, fieldbook's own state, environments, caches.
SKIPPED_FOLDERS = {".git", LOCAL_DIR, ".venv", "venv", "__pycache__", "node_modules",
                   ".mypy_cache", ".ruff_cache", ".pytest_cache", ".ipynb_checkpoints"}  # fmt: skip
MAX_FILES = 10_000
MB = 1024 * 1024


@dataclass
class Collector:
    """Finds the files a run wrote and copies them to storage; `sync` can be called while
    the run goes (a file that changed again is copied again) and once more at its end."""

    run_id: str
    cfg: Settings
    roots: dict[str, Path]  # "project" -> the repository, "output" -> the run's output folder
    since: float  # the run's start: older files were there before
    _sent: dict[str, tuple[int, int]] = field(default_factory=dict)  # key -> (mtime, size)

    def sync(self) -> dict[str, list[dict]]:
        """Copy what is new or changed; returns what is kept and what was left out."""
        files = storage.fs(self.cfg)
        folder = f"{records.FOLDER}/{self.run_id}"
        max_file = self.cfg.output_max_file_mb * MB
        max_total = self.cfg.output_max_total_mb * MB
        kept, left_out, total = [], [], 0
        for key, path, stat in sorted(self._written()):
            entry = {"path": key, "size": stat.st_size}
            if stat.st_size > max_file:
                left_out.append(entry | {"why": f"over {self.cfg.output_max_file_mb} MB"})
            elif len(kept) >= MAX_FILES or total + stat.st_size > max_total:
                left_out.append(entry | {"why": "past the run's total"})
            else:
                kept.append(entry)
                total += stat.st_size
                state = (stat.st_mtime_ns, stat.st_size)
                if self._sent.get(key) != state:
                    try:
                        storage.makedirs(files, os.path.dirname(f"{folder}/files/{key}"))
                        files.put_file(str(path), f"{folder}/files/{key}")
                        self._sent[key] = state
                    except OSError:  # gone or unreadable by now: the next sync, or never
                        kept.pop()
        manifest = {"kept": kept, "left_out": left_out}
        if kept or left_out:
            storage.makedirs(files, folder)
            files.pipe_file(f"{folder}/files.json", json.dumps(manifest).encode())
        return manifest

    def _written(self):  # noqa: ANN202
        """(key, path, stat) of each regular file changed since the run started."""
        for label, root in self.roots.items():
            if not root.is_dir():
                continue
            for folder, subfolders, names in os.walk(root):
                if label == "project":
                    subfolders[:] = [d for d in subfolders if d not in SKIPPED_FOLDERS]
                for name in names:
                    path = Path(folder, name)
                    try:
                        stat = path.lstat()
                    except OSError:
                        continue
                    if not os.path.isfile(path) or path.is_symlink():
                        continue
                    # whole seconds: some filesystems keep times no finer than that
                    if label == "output" or stat.st_mtime >= int(self.since):
                        yield f"{label}/{path.relative_to(root).as_posix()}", path, stat


def manifest(run_id: str, cfg: Settings) -> dict[str, list[dict]]:
    """What was kept of a run's files, and what was left out."""
    try:
        return json.loads(storage.fs(cfg).cat_file(f"{records.FOLDER}/{run_id}/files.json"))
    except FileNotFoundError:
        return {"kept": [], "left_out": []}


def fetch(run_id: str, key: str, target: Path, cfg: Settings) -> Path:
    """Copy one kept file (`project/train.log`) to `target` (a folder: under its name)."""
    target = target / Path(key).name if target.is_dir() else target
    storage.fs(cfg).get_file(f"{records.FOLDER}/{run_id}/files/{key}", str(target))
    return target
