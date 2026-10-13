"""The ECR registry API calls, against a fake registry (no network)."""

import base64
import io
import urllib.error
import urllib.request
from email.message import Message

import pytest

from fieldbook.images import registry

DIGEST = "sha256:" + "a" * 64
CREDENTIALS = ("AWS", "secret")
HOST = "1.dkr.ecr.us-west-2.amazonaws.com"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    """A registry with image tags `env-1`, taking the login as basic authentication."""
    urls = []

    def urlopen(request: urllib.request.Request, timeout: float):
        url = request.full_url
        urls.append(url)
        assert (
            request.get_header("Authorization")
            == "Basic " + base64.b64encode(b"AWS:secret").decode()
        )
        if url != f"https://{HOST}/v2/fieldbook/manifests/env-1":
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        response = io.BytesIO()
        response.headers = {"Docker-Content-Digest": DIGEST}
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


def test_existing_images_are_found(fake) -> None:
    repo = registry.Repository.parse(f"{HOST}/fieldbook")
    assert registry.digest(repo, "env-1", CREDENTIALS) == DIGEST
    assert registry.digest(repo, "env-2", CREDENTIALS) is None
    assert repo.ref("env-1") == f"{HOST}/fieldbook:env-1" and repo.region == "us-west-2"


def test_other_registry_errors_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(registry.RegistryError, match="HTTP 403"):
        registry.digest(registry.Repository.parse(f"{HOST}/fieldbook"), "env-1", CREDENTIALS)


def test_only_ecr_is_supported() -> None:
    for other in ("ghcr.io/me/fieldbook", "docker.io/me/fieldbook", f"{HOST}"):
        with pytest.raises(registry.RegistryError, match="must be an ECR repository"):
            registry.Repository.parse(other)
