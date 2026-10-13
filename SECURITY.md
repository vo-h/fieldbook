# Security

fieldbook handles credentials (cloud accounts, a tracking database, members' keys) and runs code on machines you pay for, so security reports are taken seriously.

## Reporting a problem

Please do not open a public issue. Report it privately instead:

- on GitHub: the repository's **Security** tab, then **Report a vulnerability**; or
- by email: hvo.getz@gmail.com, with "fieldbook security" in the subject.

Say what you found, how to reproduce it, and what it lets someone do. You will get a reply within a week. Once a fix is released, the report is credited to you in the changelog unless you would rather it was not.

## What counts

- A way to read or change another project's data, runs, or credentials.
- Credentials written where they should not be: logs, run records, code snapshots, images, or the storage root.
- A way for a member with read access to write, or for anyone to launch cloud machines on someone else's account.
- Code execution from reading a dataset, a run record, or a settings file.

## Supported versions

fieldbook is in alpha. Fixes go into the latest release only.

## How fieldbook treats secrets

- Code snapshots are scanned with [gitleaks](https://github.com/gitleaks/gitleaks) before they are stored; a snapshot with a finding is refused.
- Secrets live in a project's `.env`, which `fieldbook init` adds to `.gitignore`; they are passed to cloud machines by name and masked in recorded commands and output.
- Members' credentials are encrypted to their GitHub SSH keys before they are committed (`design/sharing.md`).
