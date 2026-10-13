# Shared projects

Status: Built, except a check with two real members (README.md §9); central services (§6) are designed, not built · Last updated: 2026-10-09

Several people work on one project ([projects.md](projects.md) §1). The project is the unit of sharing: its settings are in its repository, and its data, runs, cloud resources, and members belong to it. Sharing adds no servers. Code: `fieldbook/members/`, the members' groups in `fieldbook/clouds/aws.py`, `fieldbook access`.

## 1. Requirements

| # | Requirement |
|---|---|
| R1 | Joining a project is cloning its repository and running `fieldbook init`; nothing else to fetch or configure. |
| R2 | A project's shared settings travel with its code: versioned, reviewed in pull requests, and recorded in each run's code snapshot (projects.md §2). |
| R3 | Each person's access is individual and can be revoked, per project. |
| R4 | Runs, dataset versions, and costs are attributed to a person and kept separate per project. |
| R5 | Several members can save to the same dataset at once. |
| R6 | Free, widely used components; no new servers. |

## 2. Members: the GitHub repository decides

Who is on a project is who has access to its GitHub repository. Access to everything else (the run database, the AWS group, RunPod) is copied from it by one command, so adding or removing a person is done in one place.

| GitHub permission | On the project |
|---|---|
| Admin, Maintain, Write | **Member:** reads and changes data and runs, launches cloud runs |
| Triage, Read | **Viewer:** reads data and runs |

**`fieldbook access sync`** (the owner, after changing access on GitHub; `--dry-run` shows what it would do):

1. Reads the repository's collaborators, their permissions, and their public SSH keys in one request (GitHub's GraphQL API through `gh api graphql`; the owner is logged in with `gh auth login`).
2. **Database:** with the owner's database login (in the owner's `.env`), creates a login per person, `<project>/<github-login>`, with a random password (sent to Postgres as its SCRAM-SHA-256 verifier, so a server that logs statements never holds the password), in the project's group `<project>:writers` or `<project>:readers`; drops the logins of people no longer on the repository. MLflow's tables must already exist and belong to the owner (the owner's `fieldbook init` creates them); `sync` stops with what to do otherwise. The groups' rights are set on the project's database: connecting (taken away from everyone else; Postgres lets every login connect to a database until then, so a new project's database is open to other projects' logins until its first `sync`), reading or writing MLflow's tables, and the same on tables a later MLflow upgrade adds (default privileges of the owner, who runs upgrades).
3. **Delivering the password:** encrypts each new login (`PGUSER`, `PGPASSWORD`) with [age](https://github.com/FiloSottile/age) to the person's public SSH keys, which GitHub publishes, and writes it to `.fieldbook-members/<login>/postgres.age` in the repository, with the keys used in `recipients.txt`. Only that person can read it, with the SSH key they use for GitHub; the owner commits the folder. If a person's keys change, their password is changed and encrypted again; until they `git pull` and run `fieldbook init`, the database refuses their old one, and the error says so. `fieldbook access sync --renew <login>` does the same on request (e.g. a password that leaked); passwords are never changed by hand.
4. **AWS:** keeps the IAM groups `fieldbook-<project>-write` and `-read` (policies as §3) holding exactly the IAM users named after the collaborators' GitHub logins; people without such a user are listed. With IAM Identity Center instead of IAM users, `fieldbook access aws-policy` prints the policies to assign as permission sets.
5. **RunPod:** RunPod has no API for keys, so the owner creates a named key per member in the RunPod console and runs `fieldbook access runpod-key alice`, which encrypts it to `.fieldbook-members/alice/runpod.age`. `sync` lists members without a key, and keys to delete in the console for people who left. When a member's SSH keys change, their RunPod key must be replaced, not entered again: the old encrypted copy stays in the repository's history, readable with the old SSH key.

The person running `sync` is skipped: they already have the owner's access.

**Viewers** are recognized by what their logins allow, so nothing has to be told to their machines: `fieldbook run` stops before starting the script, saying a viewer cannot record runs (the run's set-up asks the database whether this login may insert runs); cloud costs are recorded only from members' machines; and `fieldbook init` reports the run database and the bucket as read-only (its bucket check lists instead of writing a test file).

**A member** clones the repository and runs `fieldbook init`, which finds their folder in `.fieldbook-members/` by their public keys (`~/.ssh/*.pub`), decrypts it with the matching private key (asking for its passphrase if it has one), and writes the credentials to `.env` (again later only when the owner renewed them, or to put back a value `.env` lost; values changed by hand are left). Needs: `age` installed, and an `ssh-ed25519` or `ssh-rsa` key file in `~/.ssh/` that is on their GitHub profile (age does not use ssh-agent, password-manager agents, or security keys). This is required of every member: credentials are encrypted only to keys GitHub lists for the person, never to a key someone adds to the repository. A person without a usable key gets no credentials, and `sync` says what they need.

**Leaving:** remove the person on GitHub and run `fieldbook access sync`: their database login, AWS group membership, and member folder are removed, and their RunPod key is listed for deletion. Cloud runs and datasets they made stay, attributed to them.

## 3. Each project's resources

| Resource | Per project | Notes |
|---|---|---|
| Storage | A folder in the team's bucket: every project sets `storage_root = "s3://acme-data"`, and fieldbook keeps everything of the project (datasets, files, code snapshots, artifacts, checkpoints) under `s3://acme-data/vision/` (storage.md §1) | Members' AWS policy and the machine roles cover only the project's folder. A project needing separate billing can use a bucket of its own |
| Run database | One Postgres server for the team, a database per project (`postgresql://db.acme.com/vision`) | MLflow's tables have fixed names, so projects cannot share tables; Postgres grants connections per database, so members of one project do not see another's runs |
| Image registry | An ECR repository per project | Images differ per project anyway (one per `uv.lock`) |
| Machine roles | A pair per project, limited to its folder (`fieldbook-run-<hash>`, `fieldbook-build-<hash>`; cloud-runs.md §7) | Created by the owner's `fieldbook init` |
| Members' AWS access | Two IAM groups per project, kept by `fieldbook access sync` (§2). Writers: the project's folder, its image repository (create, push, and pull), launching machines (SkyPilot's minimal policy, [SkyPilot: AWS permissions](https://docs.skypilot.ai/en/latest/cloud-setup/cloud-permissions/aws.html)), passing the project's machine roles, and reading CloudTrail (costs). Readers: reading the project's folder | Members use their own AWS identities: IAM users named after their GitHub logins, or IAM Identity Center with the policies `fieldbook access aws-policy` prints. SkyPilot's launch rights are account-wide: a writer can start machines in the account, and terminate only fieldbook's (`fieldbook-*`) |
| RunPod | One RunPod team account; a named key per member, created by the owner | Deleting a member's key revokes it; pods belong to the team account, so attribution is by `fieldbook.author` (§4) |

## 4. Working together

- **Attribution:** runs get `fieldbook.author`, like dataset versions: the `author` setting, which `fieldbook init` sets once to the person's GitHub login (from their member folder, else the GitHub CLI), kept in the project's `.fieldbook/config.toml` so no command asks GitHub; else git's `user.name`. `fieldbook log costs --total` sums `fieldbook.cost_usd` per month and author (deleted runs included: their machines were paid for); costs are per project, since each project has its own database.
- **Costs** keep being recorded from the cloud's records by whichever member opens the UI or launches next (§3 gives every member CloudTrail read access).
- **Several members saving to one dataset:** a save writes only what changed, as one Lance merge-insert, instead of rewriting the whole index (data.md §6). Concurrent saves to different paths are redone on the latest version, writing only their changes again; saves to the same paths conflict, as before.
- **No reading across projects:** a project's datasets are used only by that project; `fb.dataset` has no project argument, `fieldbook data` no `--project` option, and inputs and `copy_from` naming another project are refused. Cross-project use would make cleanup consult every project's run database and blur who owns what. Data needed by two projects is added to each (files are stored per project anyway).

## 5. Security model

| Boundary | Enforced by |
|---|---|
| Between projects | AWS permissions per project (storage, registry, roles), a database per project, repository access |
| Within a project | Trust: every member can read and change the project's data and runs. Enforced permissions within a project need central services (§6) |
| Per person | Individual AWS identities, database logins, and RunPod keys; each revocable (§2) |

Shared settings in a repository are addresses, not secrets; a public repository reveals bucket and database names, but access still needs credentials.

Decisions behind this design: a database per project on one server rather than one shared database (MLflow's tables have fixed names, and Postgres grants connections per database); a login per member rather than a shared password (revocable, attributable); a folder per project in one bucket rather than a bucket each (one bucket to manage; AWS policies cover a folder); an image repository per project; a RunPod key per member; every member needs a GitHub SSH key that is a file in `~/.ssh/`, with no other way to add a key.

## 6. Later: central services

Not built. When an organization needs enforced permissions within a project, or single sign-on, central services can be added on one VM. Each maps onto projects:

| Service | Per project | Facts checked 2026-10-09 |
|---|---|---|
| MLflow server with built-in authentication (`--app-name basic-auth`) | The project's experiments and models, readable or editable by its members (READ, USE, EDIT, MANAGE); artifacts served through the server, so members need no bucket access for them | Not experimental; users in Postgres ([MLflow: basic auth](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/)). SSO is not built in: community plugin `mlflow-oidc-auth`, or a proxy ([MLflow: SSO](https://mlflow.org/docs/latest/self-hosting/security/sso/)) |
| SkyPilot API server holding the cloud credentials | A private workspace per project, limited to its members (`allowed_users`) | Roles and workspaces need SSO through an auth proxy; Kubernetes recommended, VM/Docker without failover ([deploying](https://docs.skypilot.ai/en/latest/reference/api-server/api-server-admin-deploy.html), [RBAC](https://docs.skypilot.ai/en/latest/reference/auth.html), [workspaces](https://docs.skypilot.co/en/latest/admin/workspaces.html)) |
| Single sign-on (Caddy and OAuth2 Proxy) | One identity-provider group per project drives the above | — |
| Secrets manager | Platform credentials move into the services; personal tokens stay in `.env` | Infisical's free self-hosted edition has no SSO or role-based access ([pricing](https://envmanager.com/blog/infisical-pricing)) |
| Backups, `fieldbook upgrade` | Nightly `pg_dump` to the bucket; migrations with a backup first | — |

## 7. Risks

- **Direct database access:** a member with a database login can change any of the project's runs; acceptable within a project (§5).
- **Settings changes reach everyone on their next pull:** intended (reviewed in pull requests), but a bad change affects every member until reverted.
- **Account-wide launch rights:** SkyPilot's minimal AWS policy lets a member start machines anywhere in the account (§3).
- **A project's database is open to the server's other logins until its first `sync`** (§2).

## 8. Testing

`tests/members/`: a fake `gh`, real `age` and SSH keys (including a passphrase-protected key through a pseudo-terminal), a Postgres server that checks passwords with an owner who is not a superuser (as on Neon and RDS), MLflow's own tables with a member and a viewer, moto for the IAM groups, and `fieldbook access sync` as a command. The real GitHub query was run against a real repository.
