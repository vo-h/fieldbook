"""Freeing what old runs left behind (runs.md §22): `fieldbook log rm`. Checkpoints, kept
output and files, and environment images are the large things nothing else deletes;
`fieldbook data rm --old` frees dataset files, code snapshots, and package lists (data.md §7).

A run's record (that it ran, how it ended, its notes, its cost) is small and is kept: the
history stays whole, only the bulk goes.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fieldbook import storage
from fieldbook.config import Settings
from fieldbook.launch import records

RETENTION = timedelta(days=30)
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
FOLDERS = {"checkpoints": "checkpoints", "kept": records.FOLDER}  # what a run leaves, by kind
KEPT_IMAGE_TAGS = ("buildcache",)  # the build cache, reused by every build
GOING = (records.RUNNING, records.LAUNCHING)


@dataclass
class Report:
    dry_run: bool
    checkpoints: dict[str, int] = field(default_factory=dict)  # run ID -> bytes
    kept: dict[str, int] = field(default_factory=dict)  # run ID -> bytes of output and files
    images: dict[str, int] = field(default_factory=dict)  # image tag -> bytes
    spared: list[str] = field(default_factory=list)  # old runs kept because noted `good`
    image_error: str = ""  # why images were not looked at or deleted

    @property
    def total(self) -> int:
        return sum(map(sum, (self.checkpoints.values(), self.kept.values(), self.images.values())))


def of_run(run_id: str, cfg: Settings, *, dry_run: bool = False) -> Report:
    """Free one run's checkpoints and kept output and files, whatever its age."""
    report = Report(dry_run)
    _free(storage.fs(cfg), [run_id], report)
    return report


def prune(cfg: Settings, *, retention: timedelta = RETENTION, dry_run: bool = False) -> Report:
    """Free the checkpoints and kept output and files of runs older than `retention`,
    and the environment images not pushed or pulled since. Kept: runs still going, runs
    noted `good` (their checkpoints are the ones worth having), and the newest image."""
    from fieldbook import notes

    files = storage.fs(cfg)
    cutoff = datetime.now(UTC) - retention
    old = sorted(run_id for run_id in _with_bulk(files)
                 if records.started_at(run_id) < cutoff.timestamp())  # fmt: skip
    states = records.states(cfg)
    old = [r for r in old if r not in states or states[r].status() not in GOING]
    noted = notes.of(old, cfg)
    report = Report(dry_run)
    report.spared = [r for r in old if notes.verdict(noted.get(r, [])) == "good"]
    _free(files, [r for r in old if r not in report.spared], report)
    if cfg.image_registry:
        try:
            _images(cfg, cutoff, report)
        except Exception as e:  # no permission, no network: say so, the rest is done
            report.image_error = f"{type(e).__name__}: {e}"
    return report


def _with_bulk(files) -> set[str]:  # noqa: ANN001
    """IDs of the runs that have a checkpoint folder or kept output and files."""
    found = set()
    for folder in FOLDERS.values():
        try:
            names = files.ls(folder, detail=False)
        except FileNotFoundError:
            continue
        found |= {name.rsplit("/", 1)[-1] for name in names}
    return {name for name in found if RUN_ID.match(name)}


def _free(files, run_ids: list[str], report: Report) -> None:  # noqa: ANN001
    for run_id in run_ids:
        for kind, folder in FOLDERS.items():
            path = f"{folder}/{run_id}"
            if not files.exists(path):
                continue
            getattr(report, kind)[run_id] = int(files.du(path))
            if not report.dry_run:
                files.rm(path, recursive=True)


def _images(cfg: Settings, cutoff: datetime, report: Report) -> None:
    """Environment images (cloud-runs.md §5) neither pushed nor pulled since `cutoff`: ECR
    records an image's last pull, so an image a run still uses is not one of them."""
    from fieldbook.clouds import aws
    from fieldbook.images import registry

    repo = registry.Repository.parse(cfg.image_registry)
    ecr = aws.client("ecr", repo.region)
    found = []
    for page in ecr.get_paginator("describe_images").paginate(repositoryName=repo.name):
        for image in page["imageDetails"]:
            tags = [t for t in image.get("imageTags", []) if t.startswith("env-")]
            if tags and not set(image.get("imageTags", [])) & set(KEPT_IMAGE_TAGS):
                used = max(image["imagePushedAt"], image.get("lastRecordedPullTime")
                           or image["imagePushedAt"])  # fmt: skip
                found.append((image["imagePushedAt"], used, tags, image.get("imageSizeInBytes", 0)))
    found.sort(key=lambda item: item[0])
    for _, used, tags, size in found[:-1]:  # never the newest: the current environment's
        if used < cutoff:
            for tag in tags:
                report.images[tag] = size
            if not report.dry_run:
                ecr.batch_delete_image(repositoryName=repo.name,
                                       imageIds=[{"imageTag": tag} for tag in tags])  # fmt: skip
