"""The registry for environment images (cloud-runs.md §5): an AWS ECR repository
(other registries are deferred). Logins come from AWS credentials (12-hour passwords);
whether an image exists is asked of the registry HTTP API (nothing is pulled).
"""

import base64
import urllib.error
import urllib.request
from dataclasses import dataclass

from fieldbook.clouds import aws

MANIFEST_TYPES = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])  # fmt: skip


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class Repository:
    host: str  # e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com
    name: str  # e.g. fieldbook

    @classmethod
    def parse(cls, value: str) -> "Repository":
        host, sep, name = value.partition("/")
        if not sep or not name or aws.ecr_region(host) is None:
            raise RegistryError(
                f"image_registry {value!r} must be an ECR repository, "
                "<account>.dkr.ecr.<region>.amazonaws.com/<name> (other registries are not "
                "supported yet)"
            )
        return cls(host, name.rstrip("/"))

    def ref(self, tag: str) -> str:
        return f"{self.host}/{self.name}:{tag}"

    @property
    def region(self) -> str:
        return aws.ecr_region(self.host) or ""


def login(repo: Repository) -> tuple[str, str]:
    """(user name, password) for `docker login`."""
    return aws.ecr_login(repo.region)


def ensure_repository(repo: Repository) -> None:
    """Create the repository if it does not exist."""
    aws.ensure_ecr_repository(repo.region, repo.name)


def digest(repo: Repository, tag: str, credentials: tuple[str, str]) -> str | None:
    """The image's digest (`sha256:...`) if `tag` exists, else None."""
    basic = base64.b64encode(":".join(credentials).encode()).decode()
    request = urllib.request.Request(
        f"https://{repo.host}/v2/{repo.name}/manifests/{tag}", method="HEAD",
        headers={"Accept": MANIFEST_TYPES, "Authorization": f"Basic {basic}"},
    )  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RegistryError(f"{repo.host}: HTTP {e.code} checking {tag}") from None
