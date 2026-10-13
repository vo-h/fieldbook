"""Finding an environment image in the registry, and building it if missing: on a cloud
machine (default) or with local Docker."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fieldbook.code import environments
from fieldbook.config import Settings
from fieldbook.images import registry, spec
from fieldbook.images.spec import ImageError
from fieldbook.launch import note, skypilot

BUILD_RESOURCES = {"cpus": "4+", "memory": "16+", "disk_size": 100}  # room for CUDA wheels


@dataclass(frozen=True)
class Image:
    ref: str  # <repository>:<tag>@<digest>
    repository: registry.Repository
    login: tuple[str, str]  # for machines pulling it (ECR: valid 12 hours)


def image(repo: Path, cfg: Settings, *, local: bool = False, platform: str = "linux/amd64",
          build: bool = True) -> Image:  # fmt: skip
    """The environment image for `repo`, built first if missing."""
    if not cfg.image_registry:
        raise ImageError(
            "set image_registry to an ECR repository for cloud runs; see fieldbook config"
        )
    env = _environment(repo, cfg, platform)
    repo_ref = registry.Repository.parse(cfg.image_registry)
    login = registry.login(repo_ref)
    found = registry.digest(repo_ref, env.tag, login)
    if not found:
        if not build:
            raise spec.ImageMissing(
                f"environment image {env.tag} is not built yet (run fieldbook build)"
            )
        where = "local Docker" if local else cfg.build_cloud
        note(f"environment changed or new: building image {env.tag} ({where})")
        registry.ensure_repository(repo_ref)
        with tempfile.TemporaryDirectory(prefix="fieldbook-build-") as tmp:
            spec.build_context(env, Path(tmp))
            (Path(tmp) / "build.sh").write_text(_build_script(repo_ref, env))
            if local:
                _local_build(Path(tmp), login)
            else:
                _cloud_build(Path(tmp), login, env, cfg)
        found = registry.digest(repo_ref, env.tag, login)
        if not found:
            raise ImageError(
                f"the build finished but {repo_ref.ref(env.tag)} is not in the registry"
            )
    return Image(f"{repo_ref.ref(env.tag)}@{found}", repo_ref, login)


def _environment(repo: Path, cfg: Settings, platform: str) -> spec.Environment:
    python = environments.query_interpreter(str(environments.find(repo, repo).python))
    return spec.environment(repo, cfg, python.get("python", "3.12"), platform=platform)


def _build_script(repo_ref: registry.Repository, env: spec.Environment) -> str:
    """Build and push with the cache kept in the registry (the build machine is deleted).
    No attestations and Docker media types: the format checked in design §12."""
    image_ref = repo_ref.ref(env.tag)
    cache = repo_ref.ref("buildcache")
    return f"""set -euo pipefail
echo "$FIELDBOOK_REGISTRY_PASSWORD" | docker login --username "$FIELDBOOK_REGISTRY_USERNAME" --password-stdin {repo_ref.host} >/dev/null
docker buildx create --use --driver docker-container --name fieldbook-build >/dev/null 2>&1 || docker buildx use fieldbook-build
docker buildx build --platform {env.platform} --provenance=false --sbom=false \\
  --cache-from type=registry,ref={cache} \\
  --cache-to type=registry,ref={cache},mode=max,image-manifest=true,oci-mediatypes=true \\
  --output type=image,name={image_ref},push=true,oci-mediatypes=false .
"""  # noqa: E501


def _login_env(login: tuple[str, str]) -> dict[str, str]:
    return {"FIELDBOOK_REGISTRY_USERNAME": login[0], "FIELDBOOK_REGISTRY_PASSWORD": login[1]}


def _local_build(context: Path, login: tuple[str, str]) -> None:
    if not shutil.which("docker"):
        raise ImageError("--local needs Docker; without it, fieldbook build uses a cloud machine")
    env = {**os.environ, **_login_env(login)}
    if subprocess.run(["bash", "build.sh"], cwd=context, env=env).returncode != 0:
        raise ImageError("the image build failed (output above)")


def _cloud_build(context: Path, login: tuple[str, str], env: spec.Environment,
                 cfg: Settings) -> None:  # fmt: skip
    if skypilot.cloud_name(cfg.build_cloud) != "aws":  # RunPod pods cannot run Docker builds
        raise ImageError(f"build_cloud {cfg.build_cloud!r}: images are built on AWS")
    cluster = f"fieldbook-build-{env.tag.removeprefix('env-')[:8]}"
    task = {
        "name": cluster,
        "workdir": str(context),
        "resources": {"infra": cfg.build_cloud, **BUILD_RESOURCES},
        "run": "bash build.sh",
    }
    role = skypilot.machine_role(cfg, "aws", build=True)
    skypilot.launch(task, cluster, secrets=_login_env(login), envs={}, detach=False, cfg=cfg,
                    role=role)  # fmt: skip
