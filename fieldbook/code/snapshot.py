"""Code snapshots: the working tree of a git repository, archived by tree hash
(runs.md §4)."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from fieldbook import storage
from fieldbook.config import LOCAL_DIR, Settings

CODE_DIR = "code"
PACKAGES_DIR = "packages"
IGNORE_FILE = ".fieldbookignore"
MAX_SIZE_MB = 100
GITLEAKS_HINT = (
    "gitleaks is required to scan code snapshots for secrets: "
    "brew install gitleaks, or https://github.com/gitleaks/gitleaks/releases"
)


class SnapshotError(Exception):
    pass


@dataclass(frozen=True)
class Snapshot:
    hash: str  # git tree hash
    repo: Path
    created: bool  # False if an identical snapshot already existed
    size: int  # archive bytes
    outside_links: tuple[str, ...] = ()  # see `outside_links`


def repo_root(path: Path) -> Path:
    result = _git(path, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise SnapshotError(
            f"{path} is not in a git repository; code snapshots need one (run `git init`)"
        )
    return Path(result.stdout.strip())


def files(repo: Path) -> list[str]:
    """Files in the snapshot, relative to the repository root: tracked and untracked files,
    minus those matched by `.gitignore` (untracked only, as in git) or `.fieldbookignore`
    (tracked and untracked), minus `.env` files, which hold secrets, and minus the project's
    `.fieldbook/` (local data and state)."""
    extra = ["--exclude-from", IGNORE_FILE] if (repo / IGNORE_FILE).is_file() else []
    tracked = _git_list(repo, "-c")
    if extra:
        tracked -= _git_list(repo, "-c", "-i", *extra)
    untracked = _git_list(repo, "-o", "--exclude-standard", *extra)
    from fieldbook.credentials import is_secret_file

    local = f"{LOCAL_DIR}/"
    return sorted(p for p in tracked | untracked
                  if (repo / p).is_file() and not is_secret_file(p) and not p.startswith(local)
                  and f"/{local}" not in p)  # fmt: skip


def git_state(repo: Path) -> str:
    """`<commit> <branch> clean|dirty`: where the snapshot sits in git's history, for people
    (a link to the commit on GitHub; "dirty" says the snapshot differs from it). The
    snapshot, not the commit, is what reproduces the run (runs.md §4). Empty without commits
    or git."""
    try:
        commit = _git(repo, "rev-parse", "HEAD", check=False)
        if commit.returncode != 0:
            return ""  # no commits yet
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
        changed = _git(repo, "status", "--porcelain", check=False).stdout.strip()
    except SnapshotError:
        return ""
    branch = "detached" if branch in ("", "HEAD") else branch
    return f"{commit.stdout.strip()} {branch} {'dirty' if changed else 'clean'}"


def short_git(state: str) -> str:
    """`a1b2c3d main` or `a1b2c3d main+changes`, for one-line displays; empty if unknown."""
    commit, _, rest = state.partition(" ")
    branch, _, dirty = rest.partition(" ")
    return f"{commit[:7]} {branch}{'+changes' if dirty == 'dirty' else ''}" if commit else ""


def tree_hash(repo: Path, names: list[str] | None = None) -> str:
    """Git tree hash of `names` (default: `files(repo)`) as they are on disk, via a
    temporary index, so the user's index is untouched."""
    names = files(repo) if names is None else names
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        subprocess.run(
            ["git", "update-index", "--add", "-z", "--stdin"],
            cwd=repo, env=env, input="".join(f"{n}\0" for n in names),
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return _git(repo, "write-tree", env=env).stdout.strip()


def outside_links(repo: Path, names: list[str]) -> tuple[str, ...]:
    """Files in `names` that are symlinks to a path outside `repo` or to an absolute path
    (as `name -> target`). A snapshot keeps the link, not the file, so elsewhere it points
    to nothing, and checkouts refuse it."""
    root = repo.resolve()
    found = []
    for name in names:
        path = repo / name
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(root):
                found.append(f"{name} -> {target}")
    return tuple(found)


def take(path: Path, cfg: Settings | None = None, *, max_size_mb: int = MAX_SIZE_MB,
         refuse_outside_links: bool = False) -> Snapshot:  # fmt: skip
    """Snapshot the repository containing `path`, unless an identical one exists.

    Fails before reading any content if the files total more than `max_size_mb`, or, with
    `refuse_outside_links`, if symlinks point outside the repository.
    """
    repo = repo_root(path)
    names = files(repo)
    _check_size(repo, names, max_size_mb)
    links = outside_links(repo, names)
    if links and refuse_outside_links:
        raise SnapshotError(
            f"symlinks to files outside the repository, which a cloud machine does not have: "
            f"{', '.join(links)}; copy the files into the repository (or data into a dataset)"
        )
    digest = tree_hash(repo, names)
    fs = storage.fs(cfg)
    key = archive_key(digest)
    if fs.exists(key):
        storage.refresh(fs, key)  # in use again: restart its retention period
        return Snapshot(digest, repo, False, fs.size(key), links)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "files"
        for name in names:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, target, follow_symlinks=False)
        staging.mkdir(exist_ok=True)
        _scan(staging, Path(tmp) / "report.json")
        archive = Path(tmp) / "code.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(staging.rglob("*")):
                if not item.is_dir():
                    tar.add(item, arcname=item.relative_to(staging).as_posix())
        staged = f"tmp/{uuid.uuid4().hex}"
        fs.put_file(str(archive), staged)
        storage.move(fs, staged, key)
        return Snapshot(digest, repo, True, archive.stat().st_size, links)


def checkout(prefix: str, target: Path, cfg: Settings | None = None) -> str:
    """Extract the snapshot whose hash starts with `prefix` into `target`. Returns the hash."""
    digest = resolve(prefix, cfg)
    if target.exists() and any(target.iterdir()):
        raise SnapshotError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    with storage.fs(cfg).open(archive_key(digest), "rb") as f, tarfile.open(fileobj=f) as tar:
        tar.extractall(target, filter="data")
    return digest


def resolve(
    prefix: str, cfg: Settings | None = None, *, folder: str = CODE_DIR, kind: str = "code snapshot"
) -> str:
    """The full hash of the stored file in `folder` whose hash starts with `prefix`."""
    if len(prefix) < 7:
        raise SnapshotError(f"use at least 7 characters of the {kind} hash")
    fs = storage.fs(cfg)
    names = fs.ls(folder, detail=False) if fs.exists(folder) else []
    hashes = [n.rsplit("/", 1)[-1].split(".", 1)[0] for n in names]
    matches = [h for h in hashes if h.startswith(prefix)]
    if not matches:
        raise SnapshotError(f"no {kind} {prefix!r}")
    if len(matches) > 1:
        raise SnapshotError(f"{prefix!r} matches several {kind}s; use more characters")
    return matches[0]


def store_packages(packages: list[str], cfg: Settings | None = None) -> str:
    """Store a package list (`name==version` lines) once, by content hash. Returns the hash."""
    text = "".join(f"{p}\n" for p in packages)
    digest = hashlib.sha256(text.encode()).hexdigest()
    fs = storage.fs(cfg)
    key = f"{PACKAGES_DIR}/{digest}.txt"
    if fs.exists(key):
        storage.refresh(fs, key)  # in use again: restart its retention period
    else:
        staged = f"tmp/{uuid.uuid4().hex}"
        with fs.open(staged, "w") as f:
            f.write(text)
        storage.move(fs, staged, key)
    return digest


def read_packages(prefix: str, cfg: Settings | None = None) -> tuple[str, str]:
    """(hash, package list) of the stored list whose hash starts with `prefix`."""
    digest = resolve(prefix, cfg, folder=PACKAGES_DIR, kind="package list")
    with storage.fs(cfg).open(f"{PACKAGES_DIR}/{digest}.txt", "r") as f:
        return digest, f.read()


def archive_key(digest: str) -> str:
    return f"{CODE_DIR}/{digest}.tar.gz"


def _check_size(repo: Path, names: list[str], max_size_mb: int) -> None:
    sizes = {n: (repo / n).lstat().st_size for n in names}
    total = sum(sizes.values())
    if total <= max_size_mb * 2**20:
        return
    largest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:5]
    listed = "\n".join(f"  {name}  {size / 2**20:.1f} MB" for name, size in largest)
    raise SnapshotError(
        f"the code snapshot would be {total / 2**20:.0f} MB (limit {max_size_mb} MB); "
        f"the run was not started. Largest files:\n{listed}\n"
        f"Exclude outputs and data with {IGNORE_FILE} or .gitignore, "
        "or raise the limit with --max-snapshot-size."
    )


def _git_list(repo: Path, *args: str) -> set[str]:
    out = _git(repo, "ls-files", "-z", *args).stdout
    return {p for p in out.split("\0") if p}


def _scan(directory: Path, report: Path) -> None:
    gitleaks = shutil.which("gitleaks")
    if gitleaks is None:
        raise SnapshotError(GITLEAKS_HINT)
    result = subprocess.run(
        [gitleaks, "dir", ".", "--no-banner", "--redact", "--log-level", "error",
         "--report-format", "json", "--report-path", str(report)],
        cwd=directory,  # report paths relative to the repository
        capture_output=True,
        text=True,
    )  # fmt: skip
    if result.returncode == 0:
        return
    findings = json.loads(report.read_text()) if report.exists() else []
    if not findings:
        raise SnapshotError(f"gitleaks failed: {result.stderr.strip() or result.returncode}")
    listed = "\n".join(
        f"  {f.get('File', '?')}:{f.get('StartLine', '?')}  {f.get('RuleID', '?')}"
        for f in findings
    )
    raise SnapshotError(
        f"possible secrets found; the run was not started:\n{listed}\n"
        "Remove them, or add the files to .gitignore."
    )


def _git(repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True):
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=env)
    except FileNotFoundError:  # e.g. a cloud run's image, which needs no git
        raise SnapshotError("git is not installed; code snapshots need it") from None
    if check and result.returncode != 0:
        raise SnapshotError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result
