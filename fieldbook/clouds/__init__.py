"""What differs between clouds, one module per cloud (as SkyPilot's `sky/clouds/`): AWS
and RunPod for now (others are deferred):
credentials, permissions, registries, regions, machine IDs, and billing. Launching
machines is SkyPilot's job and reading buckets fsspec's; the modules that use these keep
everything cloud-neutral.

Each cloud module has `machine()` (MLflow tags naming the machine a run is on) and
`cost(tags, start, end)` (a finished launch's cost from the cloud's records).
"""

from types import ModuleType


def get(cloud: str | None) -> ModuleType | None:
    """The module for `cloud` (`aws`, `runpod`), or None for a cloud without one."""
    if cloud == "aws":
        from fieldbook.clouds import aws

        return aws
    if cloud == "runpod":
        from fieldbook.clouds import runpod

        return runpod
    return None
