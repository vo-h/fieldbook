"""A Postgres login per person on the project's run database (sharing.md §2).

The project's database (one per project on the team's server) gets two group roles,
`<project>:writers` and `<project>:readers`; each person gets a login, `<project>/<login>`,
in one of them. Only these groups and the owner may connect. The groups' rights cover
MLflow's tables and sequences now, and, through the owner's default privileges, tables a
later MLflow upgrade adds (upgrades are run by the owner, who owns the tables).

Works with the owner's login (`PGUSER`/`PGPASSWORD` in the owner's `.env`), which must be
allowed to create roles: the database's owner on Neon, RDS, and Cloud SQL is.
"""

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fieldbook.members import MembersError
from fieldbook.tracking.store import POSTGRES

MAX_NAME = 63  # Postgres truncates longer names
WRITE = {"tables": "SELECT, INSERT, UPDATE, DELETE", "sequences": "USAGE, SELECT, UPDATE"}
READ = {"tables": "SELECT", "sequences": "SELECT"}


@dataclass(frozen=True)
class Login:
    user: str
    password: str


def check_address(uri: str) -> None:
    """Members log in with their own `PGUSER`; a user name in the address would win over it,
    so every member would log in under that name and be refused."""
    from urllib.parse import urlsplit

    if urlsplit(uri).username:
        raise MembersError(
            "the run database's address names a user; members log in with their own: "
            "remove the user name from tracking_uri (`fieldbook config set tracking_uri "
            "postgresql://host/database`) and put yours in PGUSER in your .env"
        )


def scram_verifier(password: str) -> str:
    """What Postgres stores for a password (SCRAM-SHA-256, RFC 5803), computed here, as
    `psql`'s `\\password` does: the statement that sets it, which a server may log, then
    never holds the password itself. For the ASCII passwords made here."""
    import base64
    import hashlib
    import hmac

    salt, iterations = secrets.token_bytes(16), 4096
    salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    client = hmac.digest(salted, b"Client Key", "sha256")
    stored = hashlib.sha256(client).digest()
    server = hmac.digest(salted, b"Server Key", "sha256")
    encode = lambda value: base64.b64encode(value).decode()  # noqa: E731
    return f"SCRAM-SHA-256${iterations}:{encode(salt)}${encode(stored)}:{encode(server)}"


def login_name(project: str, github_login: str) -> str:
    name = f"{project}/{github_login}"
    if len(name.encode()) > MAX_NAME:
        raise MembersError(f"database login {name!r} is longer than {MAX_NAME} bytes")
    return name


def groups(project: str) -> tuple[str, str]:
    """(writers, readers); ':' cannot occur in a GitHub login, so no login is named so."""
    return f"{project}:writers", f"{project}:readers"


class Database:
    """The owner's connection to the project's database."""

    def __init__(self, uri: str, project: str):
        self.uri = POSTGRES.sub("postgresql://", uri)
        self.project = project
        self.writers, self.readers = groups(project)

    @contextmanager
    def connect(self) -> Iterator["Database"]:
        try:
            import psycopg
        except ImportError:
            raise MembersError("Postgres needs pip install 'psycopg[binary]'") from None
        try:
            self._conn = psycopg.connect(self.uri, autocommit=True)
        except psycopg.Error as e:
            raise MembersError(f"could not connect to the run database: {e}") from None
        try:
            yield self
        finally:
            self._conn.close()

    def logins(self) -> dict[str, str]:
        """{login: group} of the project's existing logins: those named `<project>/...`.
        The owner is in the groups too (Postgres 16 makes whoever creates a role a member)
        and must never be taken for a member who left."""
        rows = self._query(
            "SELECT m.rolname, g.rolname FROM pg_roles m "
            "JOIN pg_auth_members a ON a.member = m.oid JOIN pg_roles g ON g.oid = a.roleid "
            "WHERE g.rolname IN (%s, %s) AND m.rolcanlogin AND starts_with(m.rolname, %s)",
            (self.writers, self.readers, f"{self.project}/"),
        )
        return {member: group for member, group in rows}

    def check_tables(self) -> None:
        """MLflow's tables must exist and belong to the owner before logins are given out:
        rights on tables a later MLflow upgrade adds come from their owner's defaults."""
        rows = self._query(
            "SELECT tablename, tableowner, current_user FROM pg_tables WHERE schemaname = 'public'"
        )
        if not any(name == "experiments" for name, _, _ in rows):
            raise MembersError(
                "the run database has no MLflow tables yet: run `fieldbook init` first (it creates "
                "them with your login), then `fieldbook access sync`"
            )
        others = sorted({owner for _, owner, me in rows if owner != me})
        if others:
            me = rows[0][2]
            raise MembersError(
                f"MLflow's tables belong to {', '.join(others)}, not to your login ({me}): "
                f"members would lose access to tables a later MLflow upgrade adds. As "
                f'{others[0]} or an administrator, run: REASSIGN OWNED BY "{others[0]}" TO '
                f'"{me}"; then sync again'
            )

    def prepare(self) -> None:
        """Create the groups and give them their rights on this database (idempotent)."""
        from psycopg import sql

        for group, rights in ((self.writers, WRITE), (self.readers, READ)):
            if not self._query("SELECT 1 FROM pg_roles WHERE rolname = %s", (group,)):
                self._run(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(group)))
            self._grant(group, rights)
        database = sql.Identifier(self._query("SELECT current_database()")[0][0])
        self._run(sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(database))
        self._run(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(
                database, sql.Identifier(self.writers), sql.Identifier(self.readers)
            )
        )

    def add(self, user: str, group: str) -> Login:
        """Create a login with a new random password in `group`."""
        from psycopg import sql

        password = secrets.token_urlsafe(24)
        self._run(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} IN ROLE {}").format(
            sql.Identifier(user), sql.Literal(scram_verifier(password)),
            sql.Identifier(group)))  # fmt: skip
        return Login(user, password)

    def new_password(self, user: str) -> Login:
        from psycopg import sql

        password = secrets.token_urlsafe(24)
        self._run(sql.SQL("ALTER ROLE {} PASSWORD {}").format(
            sql.Identifier(user), sql.Literal(scram_verifier(password))))  # fmt: skip
        return Login(user, password)

    def move(self, user: str, group: str) -> None:
        """Put a login in `group` (a member who became a viewer, or the reverse)."""
        from psycopg import sql

        other = self.readers if group == self.writers else self.writers
        self._run(sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(other), sql.Identifier(user)))
        self._run(sql.SQL("GRANT {} TO {}").format(sql.Identifier(group), sql.Identifier(user)))

    def remove(self, user: str) -> None:
        """Drop a login. Members own nothing (the owner creates and upgrades the tables) and
        get their rights through a group, which `DROP ROLE` leaves. (`DROP OWNED` would need
        the member's own privileges, which an owner who is not a superuser, as on Neon or
        RDS, does not have.)"""
        from psycopg import sql

        self._run(sql.SQL("DROP ROLE {}").format(sql.Identifier(user)))

    def _grant(self, group: str, rights: dict[str, str]) -> None:
        from psycopg import sql

        role = sql.Identifier(group)
        self._run(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        now = sql.SQL("GRANT {} ON ALL {} IN SCHEMA public TO {}")
        later = sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT {} ON {} TO {}")
        for kind, privileges in rights.items():
            for statement in (now, later):
                self._run(statement.format(sql.SQL(privileges), sql.SQL(kind.upper()), role))

    def _query(self, query: str, params: tuple = ()) -> list[tuple]:
        with self._conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def _run(self, statement) -> None:
        import psycopg

        try:
            with self._conn.cursor() as cursor:
                cursor.execute(statement)
        except psycopg.errors.DependentObjectsStillExist as e:
            raise MembersError(
                f"run database: {str(e).strip()} (the login owns tables or rights given to "
                "it directly; move them to the owner, e.g. REASSIGN OWNED, then sync again)"
            ) from None
        except psycopg.Error as e:
            raise MembersError(f"run database: {str(e).strip()}") from None
