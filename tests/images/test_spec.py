"""What an environment image is built from (fieldbook/images/spec.py): tag, Dockerfile, wheel."""

from pathlib import Path

import pytest

from fieldbook import images
from fieldbook.config import settings
from fieldbook.images import spec

LOCK = """version = 1
requires-python = ">=3.12"

[[package]]
name = "torch"
version = "2.9.0"
source = { registry = "https://download.pytorch.org/whl/cu128" }

[[package]]
name = "numpy"
version = "2.5.3"
source = { registry = "https://pypi.org/simple" }
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n')
    (tmp_path / "uv.lock").write_text(LOCK)
    return tmp_path


def test_environment_tag_changes_only_with_the_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    env = images.environment(project, settings(), "3.12.9")
    assert env.tag == images.environment(project, settings(), "3.12.1").tag  # same minor
    (project / "train.py").write_text("print(1)")  # code is not part of the environment
    assert env.tag == images.environment(project, settings(), "3.12.9").tag
    monkeypatch.setenv("FIELDBOOK_IMAGE_APT", '["ffmpeg"]')
    assert env.tag != images.environment(project, settings(), "3.12.9").tag
    monkeypatch.delenv("FIELDBOOK_IMAGE_APT")
    (project / "uv.lock").write_text(LOCK.replace("2.5.3", "2.6.0"))
    assert env.tag != images.environment(project, settings(), "3.12.9").tag


def test_dockerfile_layers(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    monkeypatch.setenv("FIELDBOOK_IMAGE_APT", '["ffmpeg"]')
    text = spec.dockerfile(images.environment(project, settings(), "3.12"), "fieldbook-1.whl")
    lines = text.splitlines()
    heavy = next(i for i, line in enumerate(lines) if "torch==2.9.0" in line)
    rest = next(i for i, line in enumerate(lines) if line.startswith("RUN uv sync"))
    assert heavy < rest  # torch in its own, earlier layer
    assert "--index https://download.pytorch.org/whl/cu128 --default-index" in lines[heavy]
    assert "--no-install-package fieldbook" in lines[rest]
    assert 'uv pip install --python /opt/venv "./fieldbook-1.whl[s3]"' in text
    assert "ffmpeg" in lines[3] and "uv venv /opt/venv --python 3.12" in text
    (project / "uv.lock").unlink()
    with pytest.raises(images.ImageError, match="uv lock"):
        images.environment(project, settings(), "3.12")


def test_installed_fieldbook_goes_into_images_as_is(tmp_path: Path) -> None:
    """A11: without a source checkout, the image gets a wheel of the installed files, never
    a package of the same name from PyPI."""
    import importlib.metadata
    import zipfile

    site = tmp_path / "site"
    (site / "fieldbook" / "__pycache__").mkdir(parents=True)
    (site / "fieldbook" / "__init__.py").write_text("__version__ = '9.9'\n")
    (site / "fieldbook" / "__pycache__" / "x.pyc").write_bytes(b"compiled")
    info = site / "fieldbook-9.9.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fieldbook\nVersion: 9.9\n")
    (info / "INSTALLER").write_text("uv\n")
    (info / "RECORD").write_text(
        "fieldbook/__init__.py,,\nfieldbook/__pycache__/x.pyc,,\n"
        "fieldbook-9.9.dist-info/METADATA,,\nfieldbook-9.9.dist-info/INSTALLER,,\n"
        "fieldbook-9.9.dist-info/RECORD,,\n../../bin/fieldbook,,\n"
    )
    dist = importlib.metadata.PathDistribution(info)
    out = tmp_path / "out"
    out.mkdir()
    wheel = spec.pack_installed(dist, out)
    assert wheel == "fieldbook-9.9-py3-none-any.whl"
    with zipfile.ZipFile(out / wheel) as zf:
        names = sorted(zf.namelist())
        record = zf.read("fieldbook-9.9.dist-info/RECORD").decode()
    assert names == [
        "fieldbook-9.9.dist-info/METADATA", "fieldbook-9.9.dist-info/RECORD",
        "fieldbook/__init__.py",
    ]  # fmt: skip
    assert "fieldbook/__init__.py,sha256=" in record and "x.pyc" not in record
    text = spec.dockerfile(spec.Environment(tmp_path, "3.12", "ubuntu:24.04", (), "linux/amd64",
                                            (), "", "", "9.9"), wheel)  # fmt: skip
    assert f"COPY {wheel}" in text and "fieldbook==" not in text


def test_a_lock_entry_for_another_system_is_left_out(project: Path) -> None:
    """CPU-only torch is locked once per system; only the image's is installed."""
    entry = """
[[package]]
name = "torch"
version = "{version}"
source = {{ registry = "https://download.pytorch.org/whl/cpu" }}
resolution-markers = ["{marker}"]
"""
    lock = entry.format(version="2.14.1", marker="sys_platform == 'darwin'")
    lock += entry.format(version="2.14.1+cpu", marker="sys_platform != 'darwin'")
    (project / "uv.lock").write_text(lock)
    env = images.environment(project, settings(), "3.12")
    assert env.heavy_packages() == [("torch", "2.14.1+cpu", "https://download.pytorch.org/whl/cpu")]
