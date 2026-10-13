"""Credentials encrypted to one person, committed to the repository (sharing.md §2).

`.fieldbook-members/<login>/` holds `postgres.age` and `runpod.age` (each a few `NAME=value`
lines for `.env`), encrypted with age to the SSH keys on the person's GitHub profile, and
to nothing else: every member needs an `ssh-ed25519` or `ssh-rsa` key that is on GitHub
and is a file in `~/.ssh/` (age cannot use ssh-agent, password-manager agents, or security
keys). `recipients.txt` lists the keys used, so a change of keys is noticed. The age
command-line tool does the encryption; it reads passphrase-protected SSH keys, asking for
the passphrase.
"""

import shutil
import subprocess
from pathlib import Path

from fieldbook.members import MembersError

MEMBERS_DIR = ".fieldbook-members"
RECIPIENTS = "recipients.txt"
AGE_HINT = "install age (`brew install age`, or `apt install age`)"
USABLE = ("ssh-ed25519 ", "ssh-rsa ")  # the SSH key types age encrypts to
KEY_HINT = (
    "an ssh-ed25519 or ssh-rsa key that is on their GitHub profile and a file in ~/.ssh/ "
    "(`ssh-keygen -t ed25519`, then `gh ssh-key add ~/.ssh/id_ed25519.pub`)"
)


def folder(repo: Path, login: str) -> Path:
    return repo / MEMBERS_DIR / login


def recipients(github_keys: list[str]) -> list[str]:
    """The keys to encrypt to: the person's usable GitHub SSH keys, normalized (type and key
    only, without comments)."""
    normalized = {" ".join(k.split()[:2]) for k in github_keys}
    return sorted(k for k in normalized if k.startswith(USABLE))


def seal(member: Path, name: str, values: dict[str, str], keys: list[str]) -> Path:
    """Encrypt `values` to `keys` as `<member>/<name>.age`; record the keys."""
    if not keys:
        raise MembersError(f"no usable keys for {member.name}")
    member.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{k}={v}\n" for k, v in values.items())
    args = [arg for key in keys for arg in ("-r", key)]
    path = member / f"{name}.age"
    _age("--encrypt", "--armor", *args, "--output", str(path), input=text)
    record_recipients(member, keys)
    return path


def record_recipients(member: Path, keys: list[str]) -> None:
    """Record the keys a member's files are (or will be) encrypted to."""
    member.mkdir(parents=True, exist_ok=True)
    (member / RECIPIENTS).write_text("".join(f"{k}\n" for k in keys))


def sealed_to(member: Path) -> list[str]:
    """The keys the member's files were encrypted to."""
    path = member / RECIPIENTS
    return path.read_text().split("\n")[:-1] if path.is_file() else []


def own_keys() -> dict[str, Path]:
    """This machine's public SSH keys that age can use, each with its private key file
    (`~/.ssh/*.pub` next to their private keys)."""
    found: dict[str, Path] = {}
    for pub in sorted(Path("~/.ssh").expanduser().glob("*.pub")):
        private = pub.with_suffix("")
        key = " ".join(pub.read_text().split()[:2])
        if key.startswith(USABLE) and private.is_file():
            found[key] = private
    return found


def find_own(repo: Path) -> tuple[Path, Path] | None:
    """(this person's member folder, the private key that opens it), found by matching this
    machine's public keys against each folder's recipients; None if there is none."""
    mine = own_keys()
    root = repo / MEMBERS_DIR
    if not mine or not root.is_dir():
        return None
    for member in sorted(p for p in root.iterdir() if p.is_dir()):
        for key in sealed_to(member):
            if key in mine:
                return member, mine[key]
    return None


def unseal(path: Path, identity: Path) -> dict[str, str]:
    """Decrypt a sealed file with a private key (age asks for its passphrase if any)."""
    text = _age("--decrypt", "--identity", str(identity), str(path))
    values = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            values[name.strip()] = value
    return values


def _age(*args: str, input: str | None = None) -> str:
    age = shutil.which("age")
    if age is None:
        raise MembersError(f"age is required for project credentials: {AGE_HINT}")
    # A key's passphrase is asked on the terminal: age reads it from /dev/tty, not stdin.
    result = subprocess.run([age, *args], input=input, capture_output=True, text=True)
    if result.returncode != 0:
        raise MembersError(f"age: {result.stderr.strip()}")
    return result.stdout
