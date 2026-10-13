"""Members follow the GitHub repository (fieldbook/members/, sharing.md §2)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import runner, worker_port

from fieldbook.cli import app
from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.members import MembersError, sealed
from fieldbook.members import sync as members
from fieldbook.tracking import costs, store

pytestmark = pytest.mark.skipif(shutil.which("age") is None, reason="age is not installed")

FAKE_GH = """#!{python}
import json, os, sys
world = json.load(open(os.environ["FAKE_GH"]))
if sys.argv[1:3] == ["api", "user"]:
    print(world["user"])
elif sys.argv[1:3] == ["api", "graphql"]:  # one request: everyone's permission and keys
    assert "publicKeys" in " ".join(sys.argv), sys.argv
    for login, permission in world["collaborators"].items():
        keys = world["keys"].get(login, [])
        print(json.dumps({{"login": login, "permission": permission.upper(), "keys": keys}}))
else:
    sys.exit("unexpected: " + " ".join(sys.argv))
"""


class GitHub:
    """A fake `gh`: who is logged in, the repository's collaborators, and their keys."""

    def __init__(self, path: Path):
        self.path = path
        self.world = {"user": "owner", "collaborators": {"owner": "admin"}, "keys": {}}
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.world))

    def add(self, login: str, permission: str, keys: list[str]) -> None:
        self.world["collaborators"][login] = permission
        self.world["keys"][login] = keys
        self.save()

    def remove(self, login: str) -> None:
        del self.world["collaborators"][login]
        self.save()


@pytest.fixture
def github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitHub:
    bin_dir = tmp_path / "gh-bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH.format(python=sys.executable))
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH", str(tmp_path / "github.json"))
    return GitHub(tmp_path / "github.json")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "vision"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "fieldbook.toml").write_text('project = "vision"\n')
    monkeypatch.chdir(repo)
    monkeypatch.setenv("FIELDBOOK_PROJECT", "vision")
    return repo


def machine(home: Path, monkeypatch: pytest.MonkeyPatch, kind: str = "ed25519") -> str:
    """Switch to a person's machine (their home folder) with an SSH key; its public key."""
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    key = home / ".ssh" / f"id_{kind}"
    if not key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", kind, "-N", "", "-C", "me@laptop", "-f", key],
                       check=True)  # fmt: skip
    monkeypatch.setenv("HOME", str(home))
    return key.with_suffix(".pub").read_text().strip()


def test_credentials_open_only_with_the_persons_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = machine(tmp_path / "alice", monkeypatch)
    rsa = machine(tmp_path / "alice", monkeypatch, "rsa")
    member = tmp_path / "repo" / sealed.MEMBERS_DIR / "alice"
    keys = sealed.recipients([alice, rsa, "ecdsa-sha2-nistp256 AAAA unusable"])
    assert keys == sorted(" ".join(k.split()[:2]) for k in (alice, rsa))  # no comments
    sealed.seal(member, "postgres", {"PGUSER": "vision/alice", "PGPASSWORD": "pw"}, keys)
    assert "-----BEGIN AGE ENCRYPTED FILE-----" in (member / "postgres.age").read_text()
    found = sealed.find_own(tmp_path / "repo")
    assert found is not None and found[0] == member
    assert sealed.unseal(member / "postgres.age", found[1])["PGPASSWORD"] == "pw"
    machine(tmp_path / "bob", monkeypatch)
    assert sealed.find_own(tmp_path / "repo") is None  # another person's machine
    with pytest.raises(MembersError, match="no identity matched"):
        sealed.unseal(member / "postgres.age", tmp_path / "bob" / ".ssh" / "id_ed25519")


def test_only_github_ssh_key_files_are_used(
    project: Path, github: GitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keys age cannot use (security keys, ECDSA) get no credentials, and say what to do."""
    github.add(
        "alice", "write", ["sk-ssh-ed25519@openssh.com AAAA key", "ecdsa-sha2-nistp256 AAAA"]
    )
    lines = members.sync(project, settings())
    assert any(
        line.startswith("alice: no credentials yet: they need an ssh-ed25519") for line in lines
    )
    assert not sealed.folder(project, "alice").exists()
    with pytest.raises(MembersError, match="ssh-keygen -t ed25519"):
        members.runpod_key(project, "alice", "rp")


@pytest.fixture
def postgres(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Postgres server that checks passwords, with the project's databases `vision` and
    `audio`. The owner can create roles and databases but is not a superuser, as on Neon
    and RDS; their login is in the environment, as from the owner's .env."""
    psycopg = pytest.importorskip("psycopg")
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not (initdb and pg_ctl):
        pytest.skip("Postgres is not installed")
    data, password = tmp_path / "pg", tmp_path / "pw"
    password.write_text("admin-pw\n")
    subprocess.run([initdb, "-D", data, "-U", "admin", f"--pwfile={password}",
                    "--auth=scram-sha-256"], check=True, capture_output=True)  # fmt: skip
    port = worker_port(160)
    options = f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=''"
    options += " -c log_statement=all"  # B5: what a server set to log statements would keep
    subprocess.run([pg_ctl, "-D", data, "-o", options, "-l", tmp_path / "pg.log", "-w", "start"],
                   check=True, capture_output=True)  # fmt: skip
    base = f"postgresql://127.0.0.1:{port}"
    with psycopg.connect(f"{base}/postgres", user="admin", password="admin-pw",
                         autocommit=True) as conn:  # fmt: skip
        conn.execute("CREATE ROLE owner LOGIN CREATEROLE CREATEDB PASSWORD 'owner-pw'")
        conn.execute("CREATE DATABASE vision OWNER owner")
        conn.execute("CREATE DATABASE audio OWNER owner")
        conn.execute("REVOKE CONNECT ON DATABASE audio FROM PUBLIC")  # as audio's sync does
    with psycopg.connect(f"{base}/vision", user="owner", password="owner-pw",
                         autocommit=True) as conn:  # fmt: skip
        conn.execute("CREATE TABLE experiments (id serial PRIMARY KEY, name text)")
        conn.execute("CREATE TABLE runs (id serial PRIMARY KEY)")  # as MLflow has
    monkeypatch.setenv("FIELDBOOK_TRACKING_URI", f"{base}/vision")
    monkeypatch.setenv("PGUSER", "owner")
    monkeypatch.setenv("PGPASSWORD", "owner-pw")
    yield base
    subprocess.run([pg_ctl, "-D", data, "-m", "immediate", "-w", "stop"], capture_output=True)


def as_member(base: str, env: Path, database: str = "vision"):
    import psycopg

    values = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line)
    return psycopg.connect(f"{base}/{database}", user=values["PGUSER"],
                           password=values["PGPASSWORD"], autocommit=True)  # fmt: skip


def test_database_logins_follow_github(
    project: Path, github: GitHub, postgres: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    owner_home = Path(os.environ["HOME"])
    alice = machine(tmp_path / "alice", monkeypatch)
    bob = machine(tmp_path / "bob", monkeypatch)
    monkeypatch.setenv("HOME", str(owner_home))
    github.add("alice", "write", [alice])
    github.add("bob", "read", [bob])

    preview = members.sync(project, settings(), dry_run=True)
    assert "alice: database login created (member)" in preview
    assert not (project / sealed.MEMBERS_DIR).exists()
    lines = members.sync(project, settings())
    assert lines[0] == "GitHub: 2 members, 1 viewers (you, owner, are skipped)"
    assert "bob: database login created (viewer)" in lines and lines[-1].startswith("commit")
    assert members.sync(project, settings())[1:] == []  # nothing left to do

    # Alice's machine: `fieldbook init` finds her credentials and writes them to .env.
    machine(tmp_path / "alice", monkeypatch)
    monkeypatch.delenv("PGUSER")
    monkeypatch.delenv("PGPASSWORD")
    result = runner.invoke(app, ["init"])
    assert "PGUSER, PGPASSWORD written to .env (alice)" in result.output, result.output
    assert "project credentials: up to date" in runner.invoke(app, ["init"]).output
    # B7: a value lost from .env is put back; one changed by hand is left alone.
    env = project / ".env"
    kept = [line for line in env.read_text().splitlines() if not line.startswith("PGPASSWORD")]
    env.write_text("\n".join(kept).replace("PGUSER=vision/alice", "PGUSER=by-hand") + "\n")
    result = runner.invoke(app, ["init"])
    assert "project credentials: PGPASSWORD written to .env (alice)" in result.output
    assert "PGUSER=by-hand" in env.read_text()
    env.write_text(env.read_text().replace("PGUSER=by-hand", "PGUSER=vision/alice"))
    with as_member(postgres, project / ".env") as conn:
        conn.execute("INSERT INTO experiments (name) VALUES ('cats')")  # a member writes
    with pytest.raises(psycopg.OperationalError, match="permission denied"):
        as_member(postgres, project / ".env", "audio")  # another project's database

    monkeypatch.setenv("HOME", str(owner_home))  # the owner again
    monkeypatch.setenv("PGUSER", "owner")
    monkeypatch.setenv("PGPASSWORD", "owner-pw")
    with psycopg.connect(f"{postgres}/vision", autocommit=True) as conn:  # an MLflow upgrade
        conn.execute("CREATE TABLE added_later (id serial PRIMARY KEY, x int)")
    bob_env = tmp_path / "bob.env"
    member = sealed.folder(project, "bob")
    values = sealed.unseal(member / "postgres.age", tmp_path / "bob" / ".ssh" / "id_ed25519")
    bob_env.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    log = (tmp_path / "pg.log").read_text()  # B5: the server never saw the password itself
    assert values["PGPASSWORD"] not in log and "PASSWORD 'SCRAM-SHA-256$4096:" in log
    with as_member(postgres, bob_env) as conn:
        assert conn.execute("SELECT name FROM experiments").fetchall() == [("cats",)]
        assert conn.execute("SELECT * FROM added_later").fetchall() == []
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("INSERT INTO experiments (name) VALUES ('dogs')")  # a viewer
    # B1: a viewer is told so up front, and their machine does not try to record costs.
    address = f"{postgres}/vision"
    assert store.read_only(address) is False  # the owner
    monkeypatch.setenv("PGUSER", values["PGUSER"])
    monkeypatch.setenv("PGPASSWORD", values["PGPASSWORD"])
    assert store.read_only(address) is True
    monkeypatch.setattr(costs, "MlflowClient", None, raising=False)
    assert costs.record(settings()) == 0  # skipped before anything is read or priced
    monkeypatch.setenv("PGUSER", "owner")
    monkeypatch.setenv("PGPASSWORD", "owner-pw")
    with as_member(postgres, project / ".env") as conn:
        conn.execute("INSERT INTO added_later (x) VALUES (1)")  # rights on later tables

    # A new password on request (e.g. one that leaked): the old one stops working.
    with pytest.raises(MembersError, match="cannot renew carol, owner"):
        members.sync(project, settings(), renew=["carol", "owner"])
    lines = members.sync(project, settings(), renew=["Bob"])
    assert "bob: new database password encrypted (renewed)" in lines
    with pytest.raises(psycopg.OperationalError, match="authentication failed"):
        as_member(postgres, bob_env)
    values = sealed.unseal(member / "postgres.age", tmp_path / "bob" / ".ssh" / "id_ed25519")
    bob_env.write_text("".join(f"{k}={v}\n" for k, v in values.items()))

    github.world["collaborators"]["bob"] = "write"  # promoted on GitHub
    github.remove("alice")
    lines = members.sync(project, settings())
    assert "bob: database login is now a member's" in lines
    assert {"alice: database login removed", "alice: credentials folder removed"} <= set(lines)
    with pytest.raises(psycopg.OperationalError, match="authentication failed"):
        as_member(postgres, project / ".env")
    with as_member(postgres, bob_env) as conn:
        conn.execute("INSERT INTO experiments (name) VALUES ('dogs')")

    new_key = machine(tmp_path / "bob-new-laptop", monkeypatch)
    monkeypatch.setenv("HOME", str(owner_home))
    github.world["keys"]["bob"] = [new_key]
    github.save()
    assert "bob: keys changed: new database password encrypted" in members.sync(project, settings())
    with pytest.raises(psycopg.OperationalError, match="authentication failed"):
        as_member(postgres, bob_env)  # the old password no longer works
    assert sealed.sealed_to(member) == [" ".join(new_key.split()[:2])]


def test_aws_groups_follow_github(
    project: Path, github: GitHub, mock_aws, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIELDBOOK_PROJECT", "vision")
    key = machine(tmp_path / "alice", monkeypatch)
    github.add("alice", "write", [key])
    github.add("bob", "read", [key])
    github.add("carol", "write", [key])
    iam = aws.client("iam")
    for user in ("alice", "bob"):
        iam.create_user(UserName=user)
    lines = members.sync(project, settings())
    assert "alice: added to AWS group fieldbook-vision-write" in lines
    assert "bob: added to AWS group fieldbook-vision-read" in lines
    assert any(line.startswith("carol: no IAM user named carol") for line in lines)
    assert aws.group_users("fieldbook-vision-write") == {"alice"}
    assert not any(line.startswith("commit") for line in lines)  # B13: no file changed
    policy = iam.get_group_policy(GroupName="fieldbook-vision-write", PolicyName="fieldbook")
    text = json.dumps(policy["PolicyDocument"])
    assert "arn:aws:s3:::data-bucket/fieldbook/vision/*" in text and "ec2:RunInstances" in text
    assert "repository/fieldbook" in text and "fieldbook-run-" in text
    assert "ecr:CreateRepository" in text  # B2: the first build may be a member's
    read = json.dumps(iam.get_group_policy(GroupName="fieldbook-vision-read",
                                           PolicyName="fieldbook")["PolicyDocument"])  # fmt: skip
    assert "s3:GetObject" in read and "PutObject" not in read and "ec2" not in read
    github.remove("alice")
    github.world["collaborators"]["bob"] = "write"
    github.save()
    iam.create_user(UserName="Old-Member")  # D6: removed by IAM's own spelling of the name
    iam.add_user_to_group(GroupName="fieldbook-vision-read", UserName="Old-Member")
    lines = members.sync(project, settings())
    assert "Old-Member: removed from AWS group fieldbook-vision-read" in lines
    assert "alice: removed from AWS group fieldbook-vision-write" in lines
    assert "bob: removed from AWS group fieldbook-vision-read" in lines
    assert aws.group_users("fieldbook-vision-write") == {"bob"}
    result = runner.invoke(app, ["access", "aws-policy", "--read"])
    assert result.exit_code == 0 and "data-bucket/fieldbook/vision/*" in result.output


def test_runpod_keys_are_listed_and_encrypted(
    project: Path, github: GitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIELDBOOK_REMOTE_CLOUD", "runpod")
    owner_home = Path(os.environ["HOME"])
    key = machine(tmp_path / "alice", monkeypatch)
    monkeypatch.setenv("HOME", str(owner_home))
    github.add("alice", "write", [key])
    lines = members.sync(project, settings())
    assert "database: not a shared Postgres database; no logins to manage" in lines
    assert any("create one named fieldbook-vision-alice" in line for line in lines)
    result = runner.invoke(app, ["access", "runpod-key", "alice"], input="rp-secret\n")
    assert result.exit_code == 0 and "rp-secret" not in result.output, result.output
    assert not any("alice" in line for line in members.sync(project, settings())[1:])
    machine(tmp_path / "alice", monkeypatch)
    line, values = members.receive(project, tmp_path / "state")
    assert values == {"RUNPOD_API_KEY": "rp-secret"} and "RUNPOD_API_KEY" in line

    monkeypatch.setenv("HOME", str(owner_home))
    github.world["keys"]["alice"] = [machine(tmp_path / "alice-new", monkeypatch)]
    github.save()
    monkeypatch.setenv("HOME", str(owner_home))
    lines = members.sync(project, settings())
    assert any(
        "delete fieldbook-vision-alice in the RunPod console, create a new one" in line
        for line in lines
    )  # the old key could be read with the old SSH key
    members.runpod_key(project, "alice", "rp-2")  # with no database: recorded the new keys
    assert not any("alice" in line for line in members.sync(project, settings())[1:])
    github.remove("alice")
    lines = members.sync(project, settings())
    assert "alice: credentials folder removed" in lines


def test_a_database_address_with_a_user_name_is_refused(
    project: Path, github: GitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user name in the address would win over each member's own PGUSER."""
    monkeypatch.setenv("FIELDBOOK_TRACKING_URI", "postgresql://owner@db.example.com/vision")
    with pytest.raises(MembersError, match="remove the user name from tracking_uri"):
        members.sync(project, settings())


def test_sync_needs_the_owners_mlflow_tables_first(
    project: Path, github: GitHub, postgres: str
) -> None:
    """B9: rights on tables added by later MLflow upgrades come from their owner's defaults."""
    import psycopg

    with psycopg.connect(f"{postgres}/vision", autocommit=True) as conn:
        conn.execute("DROP TABLE experiments, runs")
    with pytest.raises(MembersError, match="run `fieldbook init` first"):
        members.sync(project, settings(), dry_run=True)
    with psycopg.connect(f"{postgres}/vision", user="admin", password="admin-pw",
                         autocommit=True) as conn:  # fmt: skip
        conn.execute("CREATE TABLE experiments (id serial PRIMARY KEY)")  # someone else's
    with pytest.raises(MembersError, match="belong to admin, not to your login .owner."):
        members.sync(project, settings())


def test_the_owner_is_not_told_to_wait_for_credentials(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B10: with a database login already in .env, having no member folder is expected."""
    key = machine(tmp_path / "alice", monkeypatch)
    sealed.seal(
        sealed.folder(project, "alice"), "postgres", {"PGUSER": "a"}, [key.rsplit(" ", 1)[0]]
    )
    machine(tmp_path / "owner", monkeypatch)  # another person's machine
    line, _ = members.receive(project, tmp_path / "state")
    assert "none for this machine's keys" in line
    (project / ".env").write_text("PGUSER=owner\nPGPASSWORD=owner-pw\n")
    assert members.receive(project, tmp_path / "state") == (None, {})


def test_a_refused_login_points_at_changed_credentials() -> None:
    """B8: after the owner re-encrypts, the old password stops working."""
    refused = 'FATAL:  password authentication failed for user "vision/alice"'
    assert "`git pull`, then `fieldbook init`" in store.login_hint(refused)
    assert store.login_hint("could not translate host name") == ""


def test_the_sync_command(project: Path, github: GitHub, tmp_path: Path, monkeypatch) -> None:
    """D4: `fieldbook access sync` itself: what it prints, and how it fails."""
    github.add("alice", "write", [machine(tmp_path / "alice", monkeypatch)])
    result = runner.invoke(app, ["access", "sync", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "GitHub: 2 members, 0 viewers (you, owner, are skipped)" in result.output
    assert result.output.strip().endswith("dry run: nothing changed")
    result = runner.invoke(app, ["access", "sync", "--renew", "nobody"])
    assert result.exit_code == 1 and "error: cannot renew nobody" in result.output
    (tmp_path / "gh-bin" / "gh").write_text("#!/bin/sh\necho 'gh: not logged in' >&2\nexit 4\n")
    result = runner.invoke(app, ["access", "sync"])
    assert result.exit_code == 1 and "error: gh api graphql: gh: not logged in" in result.output
    assert "Traceback" not in result.output
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no GitHub CLI at all
    result = runner.invoke(app, ["access", "sync"])
    assert result.exit_code == 1 and "brew install gh" in result.output


def test_a_key_with_a_passphrase_is_asked_for_on_the_terminal(tmp_path: Path) -> None:
    """D2: age reads the passphrase from the terminal, not from fieldbook's pipes."""
    import fcntl
    import pty
    import termios

    key = tmp_path / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "open sesame", "-f", key], check=True
    )
    public = " ".join(key.with_suffix(".pub").read_text().split()[:2])
    member = tmp_path / "repo" / sealed.MEMBERS_DIR / "alice"
    sealed.seal(member, "postgres", {"PGPASSWORD": "pw"}, [public])
    code = (
        "import sys; from pathlib import Path; from fieldbook.members import sealed; "
        "print('VALUES', sealed.unseal(Path(sys.argv[1]), Path(sys.argv[2])))"
    )
    master, slave = pty.openpty()

    def own_terminal() -> None:  # the child gets the pseudo-terminal as its terminal
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    process = subprocess.Popen(
        [sys.executable, "-c", code, str(member / "postgres.age"), str(key)],
        stdin=slave, stdout=slave, stderr=slave, preexec_fn=own_terminal,
    )  # fmt: skip
    os.close(slave)
    output, answered = "", False
    while True:
        try:
            chunk = os.read(master, 1024).decode(errors="replace")
        except OSError:  # the child closed the terminal
            break
        if not chunk:
            break
        output += chunk
        if not answered and "passphrase" in output.lower():
            os.write(master, b"open sesame\n")
            answered = True
    assert process.wait(timeout=30) == 0, output
    assert answered and "VALUES {'PGPASSWORD': 'pw'}" in output, output


def _mlflow(code: str, base: str, user: str, password: str) -> subprocess.CompletedProcess:
    """Run MLflow code against the project's database as one login, in its own process."""
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"{base}/vision", "PGUSER": user,
           "PGPASSWORD": password}  # fmt: skip
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


@pytest.mark.slow
def test_members_and_viewers_on_mlflows_own_tables(
    project: Path, github: GitHub, postgres: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1: the groups' rights are enough for MLflow itself, and a viewer's are read-only."""
    import psycopg

    pytest.importorskip("mlflow")
    with psycopg.connect(f"{postgres}/vision", autocommit=True) as conn:
        conn.execute("DROP TABLE experiments, runs")  # the stand-ins: MLflow makes its own
    created = _mlflow("import mlflow; mlflow.MlflowClient().search_experiments()",
                      postgres, "owner", "owner-pw")  # fmt: skip
    assert created.returncode == 0, created.stderr[-2000:]
    owner_home = Path(os.environ["HOME"])
    alice, bob = machine(tmp_path / "alice", monkeypatch), machine(tmp_path / "bob", monkeypatch)
    monkeypatch.setenv("HOME", str(owner_home))
    github.add("alice", "write", [alice])
    github.add("bob", "read", [bob])
    members.sync(project, settings())

    def login(name: str) -> tuple[str, str]:
        key = tmp_path / name / ".ssh" / "id_ed25519"
        values = sealed.unseal(sealed.folder(project, name) / "postgres.age", key)
        return values["PGUSER"], values["PGPASSWORD"]

    log_a_run = (
        "import mlflow\n"
        "mlflow.set_experiment('vision')\n"
        "with mlflow.start_run() as run:\n"
        "    mlflow.log_param('lr', 0.1); mlflow.log_metric('loss', 0.5, step=1)\n"
        "    mlflow.set_tag('fieldbook.author', 'alice')\n"
        "client = mlflow.MlflowClient()\n"
        "client.create_registered_model('cats')\n"
        "print('LOGGED', run.info.run_id)\n"
    )
    member = _mlflow(log_a_run, postgres, *login("alice"))
    assert member.returncode == 0 and "LOGGED" in member.stdout, member.stderr[-3000:]

    read = (
        "import mlflow\n"
        "client = mlflow.MlflowClient()\n"
        "experiment = client.get_experiment_by_name('vision')\n"
        "runs = client.search_runs([experiment.experiment_id])\n"
        "models = [m.name for m in client.search_registered_models()]\n"
        "print('READ', len(runs), runs[0].data.params, models)\n"
    )
    viewer = _mlflow(read, postgres, *login("bob"))
    assert viewer.returncode == 0, viewer.stderr[-3000:]
    assert "READ 1 {'lr': '0.1'} ['cats']" in viewer.stdout
    refused = _mlflow(log_a_run, postgres, *login("bob"))
    assert refused.returncode != 0 and "permission denied" in refused.stderr
    user, password = login("bob")
    monkeypatch.setenv("PGUSER", user)
    monkeypatch.setenv("PGPASSWORD", password)
    assert store.read_only(f"{postgres}/vision") is True
