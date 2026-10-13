"""The tests mirror the code, so each concern's tests are found in one place and a new
package does not go untested: `tests/<package>/` for `fieldbook/<package>/`, and
`tests/test_<module>[_topic].py` for a top-level module."""

from pathlib import Path

ROOT = Path(__file__).parent
PACKAGE = ROOT.parent / "fieldbook"


def _packages() -> set[str]:
    return {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").is_file()}


def _modules() -> set[str]:
    return {p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__"}


def test_test_folders_match_packages() -> None:
    folders = {p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    stray = sorted(folders - _packages())
    assert not stray, f"test folders without a package: {stray}"


def test_every_package_has_tests() -> None:
    untested = sorted(p for p in _packages() if not any((ROOT / p).glob("test_*.py")))
    assert not untested, f"packages without a tests/<package>/ folder: {untested}"


def test_top_level_test_files_are_named_after_modules() -> None:
    allowed = _modules() | {"layout"}
    for path in ROOT.glob("test_*.py"):
        name = path.stem.removeprefix("test_")
        assert any(name == m or name.startswith(f"{m}_") for m in allowed), (
            f"{path.name}: name it after a top-level module (test_<module>[_topic].py) "
            "or move it to the package's folder"
        )


def test_references_to_design_sections_exist() -> None:
    """`name.md §N` in the documents and the code points at a section that exists. A name
    is looked up next to the file it appears in, then in design/, then in docs/ (so
    `README.md §3` inside design/ is the design's own README)."""
    import re

    repo = ROOT.parent
    docs = [*repo.glob("design/*.md"), *repo.glob("docs/*.md"), repo / "README.md"]
    heading = re.compile(r"^#+\s+(\d+(?:\.\d+)*[a-z]?)\.?\s", re.M)
    sections = {p: set(heading.findall(p.read_text())) for p in docs}
    reference = re.compile(r"((?:design/|docs/)?[\w-]+\.md) §(\d+(?:\.\d+)*[a-z]?)")

    def find(name: str, source: Path) -> Path | None:
        places = [repo] if "/" in name else [source.parent, repo / "design", repo / "docs"]
        return next((place / name for place in places if place / name in sections), None)

    broken = []
    for path in [*docs, *PACKAGE.rglob("*.py"), *ROOT.rglob("*.py")]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for name, section in reference.findall(line):
                doc = find(name, path)
                if doc is None:
                    if name.split("/")[-1] != "name.md":  # this test's own example
                        broken.append(f"{path.relative_to(repo)}:{number}: no document {name}")
                elif not any(s == section or s.startswith(f"{section}.") for s in sections[doc]):
                    broken.append(f"{path.relative_to(repo)}:{number}: {name} §{section}")
    assert not broken, "references to missing documents or sections:\n" + "\n".join(broken)
