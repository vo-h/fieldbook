"""A run's output, kept (runs.md §18): what the run's process prints (stdout and stderr)
passes through `fieldbook run` to the terminal unchanged, and a copy goes to the project's folder
of the storage root in pieces, every half minute, so that a machine that disappears leaves
its output up to then.

    runs/<run ID>/log/000001.gz ...   pieces, while the run goes
    runs/<run ID>/output.log.gz       the pieces joined, once it ends
    runs/<run ID>/launch.log.gz       `sky launch`'s output on the laptop (cloud runs)

S3 cannot append to an object, hence pieces. Each piece is a gzip member, and gzip members
laid end to end are a gzip file, so joining is concatenation.

The copy is text for reading, not a replay: a line a progress bar rewrote keeps its last
state, colour codes are dropped, and the values of the run's secrets are masked. Anything
else a script prints is kept as printed.
"""

import contextlib
import gzip
import os
import re
import subprocess
import threading
from collections import deque

from fieldbook import storage
from fieldbook.config import Settings
from fieldbook.launch import records

MAX_BYTES = 50 * 1024 * 1024  # compressed, per run: past it, only the end is still kept
TAIL_LINES = 5000  # kept from the end of an output that passed MAX_BYTES
PIECE_BYTES = 1024 * 1024  # a piece is sent early once this much has gathered
ANSI = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")


class RunLog:
    """Collects output and writes it in pieces. `name`: `output` or `launch`."""

    def __init__(self, run_id: str, cfg: Settings, *, name: str = "output",
                 secrets: tuple[str, ...] = ()) -> None:  # fmt: skip
        self.cfg, self.name = cfg, name
        self.folder = f"{records.FOLDER}/{run_id}"
        self._secrets = [s.encode() for s in secrets if len(s) >= 4]
        self._lock = threading.Lock()
        self._partial = b""  # the line being written
        self._lines: list[bytes] = []  # complete lines not yet sent
        self._pending = 0
        self._pieces = 0
        self._sent = 0  # compressed bytes
        self._tail: deque[bytes] = deque(maxlen=TAIL_LINES)
        self._dropped = 0

    def feed(self, data: bytes) -> None:
        """Take what the process printed. Safe from several threads."""
        with self._lock:
            *complete, self._partial = (self._partial + data).split(b"\n")
            # a line still being rewritten (a progress bar): only its latest state
            again = self._partial.rfind(b"\r", 0, max(len(self._partial) - 1, 0))
            if again >= 0:
                self._partial = self._partial[again + 1 :]
            for line in complete:
                self._lines.append(self._clean(line))
                self._pending += len(line)
        if self._pending >= PIECE_BYTES:
            self.flush()

    def _clean(self, line: bytes) -> bytes:
        line = ANSI.sub(b"", line.rstrip(b"\r")).rsplit(b"\r", 1)[-1]
        for secret in self._secrets:
            line = line.replace(secret, b"***")
        return line

    def flush(self) -> None:
        """Send the complete lines gathered so far as the next piece."""
        with self._lock:
            lines, self._lines, self._pending = self._lines, [], 0
            if not lines:
                return
            if self._sent >= MAX_BYTES:  # too long: from here on only the end is kept
                self._dropped += max(len(self._tail) + len(lines) - TAIL_LINES, 0)
                self._tail.extend(lines)
                return
            self._write(lines)

    def _write(self, lines: list[bytes]) -> None:
        piece = gzip.compress(b"\n".join(lines) + b"\n")
        self._pieces += 1
        self._sent += len(piece)
        files = storage.fs(self.cfg)
        storage.makedirs(files, f"{self.folder}/log-{self.name}")
        files.pipe_file(f"{self.folder}/log-{self.name}/{self._pieces:06d}.gz", piece)

    def close(self) -> None:
        """Send what is left and join the pieces into one file."""
        with self._lock:
            if self._partial:
                self._lines.append(self._clean(self._partial))
                self._partial = b""
        self.flush()
        with self._lock:
            if self._tail:
                note = f"... {self._dropped} lines left out (output over the size limit) ..."
                self._write([note.encode(), *self._tail])
                self._tail.clear()
        files = storage.fs(self.cfg)
        pieces = _pieces(files, self.folder, self.name)
        if pieces:
            files.pipe_file(f"{self.folder}/{self.name}.log.gz",
                            b"".join(files.cat_file(p) for p in pieces))  # fmt: skip
            files.rm(f"{self.folder}/log-{self.name}", recursive=True)


def _pieces(files, folder: str, name: str) -> list[str]:  # noqa: ANN001
    try:
        return sorted(files.glob(f"{folder}/log-{name}/*.gz"))
    except FileNotFoundError:
        return []


def read(run_id: str, cfg: Settings, *, name: str = "output") -> list[str] | None:
    """A run's kept output as lines: the joined file, or the pieces so far while it runs
    (or if it never got to join them). None if nothing was kept."""
    files = storage.fresh(storage.fs(cfg))  # `--follow` reads again and again
    folder = f"{records.FOLDER}/{run_id}"
    try:
        data = files.cat_file(f"{folder}/{name}.log.gz")
    except FileNotFoundError:
        pieces = _pieces(files, folder, name)
        if not pieces:
            return None
        data = b"".join(files.cat_file(p) for p in pieces)
    return gzip.decompress(data).decode("utf-8", "replace").splitlines()


# --- running a command with its output copied ---------------------------------------------


class Copied:
    """A started process whose stdout and stderr go where they went before (file
    descriptors 1 and 2 of this process) and, as a copy, to `log`."""

    def __init__(self, process: subprocess.Popen, readers: list[threading.Thread]) -> None:
        self.process, self._readers = process, readers

    def drain(self) -> None:
        """After the process ended: wait for the last of its output. Not forever: a
        process it left behind may still hold the output open."""
        for reader in self._readers:
            reader.join(timeout=5)


def popen(command: str | list[str], log: RunLog | None, **options) -> Copied:  # noqa: ANN003
    """`subprocess.Popen(command, **options)`, with the output copied to `log` (None: as
    plain Popen). Where this process writes to a terminal the child gets a terminal too (a
    pseudo-terminal), so it prints as it would have: unbuffered, with its progress bars."""
    if log is None:
        return Copied(subprocess.Popen(command, **options), [])
    ends, readers = {}, []
    for stream, fd in (("stdout", 1), ("stderr", 2)):
        read_end, write_end = _channel(fd)
        ends[stream] = write_end
        readers.append(threading.Thread(target=_copy, args=(read_end, fd, log), daemon=True))
    if not os.isatty(1):  # through a pipe, Python would hold its output back in blocks
        options["env"] = {"PYTHONUNBUFFERED": "1", **(options.get("env") or os.environ)}
    try:
        process = subprocess.Popen(command, stdout=ends["stdout"], stderr=ends["stderr"], **options)
    finally:
        for write_end in ends.values():
            os.close(write_end)  # the child has its own; ours would keep the readers waiting
    for reader in readers:
        reader.start()
    return Copied(process, readers)


def _channel(fd: int) -> tuple[int, int]:
    """(read end, write end) for the child's stream: a pseudo-terminal if ours is one."""
    if os.isatty(fd):
        with contextlib.suppress(ImportError, OSError):
            import fcntl
            import pty
            import termios

            master, slave = pty.openpty()
            with contextlib.suppress(OSError):  # the same width, for progress bars
                size = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, size)
            return master, slave
    return os.pipe()


def _copy(source: int, target: int, log: RunLog) -> None:
    try:
        while True:
            try:
                data = os.read(source, 65536)
            except OSError:  # a pseudo-terminal's end of output
                break
            if not data:
                break
            with contextlib.suppress(OSError):
                os.write(target, data)
            log.feed(data)
    finally:
        os.close(source)
