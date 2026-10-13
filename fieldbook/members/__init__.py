"""A project's members (sharing.md §2): who has access to the project's GitHub
repository decides who may use its run database, AWS resources, and RunPod.

- `github`: the repository's collaborators and their public SSH keys (`gh`);
- `sealed`: credentials encrypted to a person's keys with age, committed to the repository;
- `database`: a Postgres login per person in the project's groups;
- `sync`: `fieldbook access sync`, which brings all of these in line with GitHub, and the
  member's side, which `fieldbook init` runs.
"""


class MembersError(Exception):
    pass
