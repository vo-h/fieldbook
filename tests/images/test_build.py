"""Finding and building environment images (fieldbook/images/build.py), against a fake
registry, a fake local build, and a fake SkyPilot (D2, D3)."""

import os
from pathlib import Path

import pytest

from fieldbook import images
from fieldbook.code import environments
from fieldbook.config import settings
from fieldbook.images import build, registry, spec
from fieldbook.launch import skypilot

REGISTRY = "1.dkr.ecr.us-west-2.amazonaws.com/fieldbook"
DIGEST = "sha256:" + "d" * 64


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with a lock, the registry setting, and no real interpreter query."""
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n')
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setenv("FIELDBOOK_IMAGE_REGISTRY", REGISTRY)
    monkeypatch.setattr(environments, "query_interpreter", lambda python: {"python": "3.12.4"})
    monkeypatch.setattr(spec, "_fieldbook_source", lambda: None)  # pack the installed files
    return repo


@pytest.fixture
def pushed(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """A fake registry: the tags in the returned set exist; the login is ("AWS", "pw")."""
    tags: set[str] = set()
    monkeypatch.setattr(registry, "login", lambda repo: ("AWS", "pw"))
    monkeypatch.setattr(registry, "ensure_repository", lambda repo: None)
    monkeypatch.setattr(
        registry, "digest", lambda repo, tag, login: DIGEST if tag in tags else None
    )
    return tags


def test_an_existing_image_is_found_not_built(
    project: Path, pushed: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = build._environment(project, settings(), "linux/amd64")
    pushed.add(env.tag)
    monkeypatch.setattr(build, "_local_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(build, "_cloud_build", lambda *a: pytest.fail("built"))
    found = images.image(project, settings())
    assert found.ref == f"{REGISTRY}:{env.tag}@{DIGEST}" and found.login == ("AWS", "pw")
    with pytest.raises(images.ImageError, match="not built yet"):
        pushed.clear()
        images.image(project, settings(), build=False)


def test_a_missing_image_is_built_locally(
    project: Path, pushed: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_local(context: Path, login: tuple[str, str]) -> None:
        seen["files"] = sorted(p.name for p in context.iterdir())
        seen["script"] = (context / "build.sh").read_text()
        seen["dockerfile"] = (context / "Dockerfile").read_text()
        pushed.add(build._environment(project, settings(), "linux/amd64").tag)  # the push

    monkeypatch.setattr(build, "_local_build", fake_local)
    found = images.image(project, settings(), local=True)
    assert found.ref.endswith(f"@{DIGEST}")
    wheel = next(f for f in seen["files"] if f.endswith(".whl"))
    assert {"Dockerfile", "build.sh", "pyproject.toml", "uv.lock"} <= set(seen["files"])
    assert f"COPY {wheel}" in seen["dockerfile"] and "fieldbook==" not in seen["dockerfile"]
    assert "--password-stdin" in seen["script"] and f"name={REGISTRY}:env-" in seen["script"]
    assert "pw" not in seen["script"]  # the password goes in a variable, never the file


def test_a_missing_image_is_built_on_a_cloud_machine(
    project: Path, pushed: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    launched = {}

    def fake_launch(
        task: dict, cluster: str, *, secrets: dict, envs: dict, detach: bool, cfg, role=None
    ) -> None:  # noqa: ANN001
        launched.update(task=task, cluster=cluster, secrets=secrets, detach=detach, role=role)
        pushed.add(build._environment(project, settings(), "linux/amd64").tag)

    monkeypatch.setattr(skypilot, "launch", fake_launch)
    monkeypatch.setattr(
        skypilot, "machine_role", lambda cfg, cloud, build=False: f"role-{cloud}-{build}"
    )
    images.image(project, settings())
    assert (
        launched["task"]["run"] == "bash build.sh"
        and launched["task"]["resources"]["infra"] == "aws"
    )
    assert launched["cluster"].startswith("fieldbook-build-") and launched["detach"] is False
    assert launched["secrets"] == {
        "FIELDBOOK_REGISTRY_USERNAME": "AWS",
        "FIELDBOOK_REGISTRY_PASSWORD": "pw",
    }
    assert launched["role"] == "role-aws-True"  # the build role, not the run role
    monkeypatch.setenv("FIELDBOOK_BUILD_CLOUD", "runpod")
    pushed.clear()
    with pytest.raises(images.ImageError, match="images are built on AWS"):
        images.image(project, settings())


def test_build_context_from_a_source_checkout(tmp_path: Path) -> None:
    """D3: from a checkout of fieldbook, the image gets a wheel built from the source."""
    if build.spec._fieldbook_source() is None:
        pytest.skip("fieldbook is not installed from a source checkout")
    env = spec.Environment(tmp_path, "3.12", "ubuntu:24.04", (), "linux/amd64", (), "version = 1\n",
                           '[project]\nname = "p"\nversion = "0"\n', "source:test")  # fmt: skip
    context = spec.build_context(env, tmp_path / "context")
    wheel = next(context.glob("fieldbook-*.whl"))
    assert f"COPY {wheel.name}" in (context / "Dockerfile").read_text()
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        assert "fieldbook/images/build.py" in zf.namelist()  # the code on this machine
    assert os.environ.get("UV_CACHE_DIR")  # built offline from the real cache (D4)
