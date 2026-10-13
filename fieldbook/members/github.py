"""The project's GitHub repository: its collaborators and their public keys, through the
GitHub CLI (`gh`), which holds the owner's GitHub login."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fieldbook.members import MembersError

GH_HINT = "install the GitHub CLI (`brew install gh`) and run `gh auth login`"
WRITE_PERMISSIONS = ("ADMIN", "MAINTAIN", "WRITE")  # the others: TRIAGE, READ

# Everyone with access and their public SSH keys, in one request per 100 people.
COLLABORATORS = """
query($owner: String!, $name: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    collaborators(first: 100, after: $endCursor) {
      edges { permission node { login publicKeys(first: 100) { nodes { key } } } }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""
ROWS = (
    ".data.repository.collaborators.edges[] | "
    "{login: .node.login, permission, keys: [.node.publicKeys.nodes[].key]}"
)


@dataclass(frozen=True)
class Collaborator:
    login: str
    writes: bool  # Write or above: a member; Read or Triage: a viewer
    keys: tuple[str, ...]  # the public SSH keys on their GitHub profile


def collaborators(repo: Path) -> list[Collaborator]:
    """Everyone with access to the repository (directly, through a team, or as an
    organization owner) with their public SSH keys, sorted by login."""
    rows = _gh(repo, "api", "graphql", "--paginate", "-F", "owner={owner}", "-F", "name={repo}",
               "-f", f"query={COLLABORATORS}", "--jq", ROWS)  # fmt: skip
    people = []
    for line in rows.splitlines():
        row = json.loads(line)
        writes = row["permission"] in WRITE_PERMISSIONS
        people.append(Collaborator(row["login"].lower(), writes, tuple(row["keys"])))
    return sorted(people, key=lambda c: c.login)


def current_login(repo: Path) -> str:
    """The GitHub login `gh` is using."""
    return _gh(repo, "api", "user", "--jq", ".login").strip().lower()


def _gh(repo: Path, *args: str) -> str:
    gh = shutil.which("gh")
    if gh is None:
        raise MembersError(f"the GitHub CLI is required: {GH_HINT}")
    result = subprocess.run([gh, *args], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise MembersError(f"gh {' '.join(args[:2])}: {detail}")
    return result.stdout
