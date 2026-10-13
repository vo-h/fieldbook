"""Shared by the command modules."""

import typer


def fail(message: str) -> typer.Exit:
    """Print an error and return the exit to raise: `raise fail(...) from None`."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def operational_errors() -> tuple[type[BaseException], ...]:
    """Errors from the user's setup or the outside world (settings, missing extras, storage,
    registries, IAM, clouds, network, files), as opposed to bugs in fieldbook."""
    import urllib.error

    from fieldbook import images, storage
    from fieldbook.clouds import aws, runpod
    from fieldbook.code import snapshot
    from fieldbook.config import ConfigError
    from fieldbook.data.dataset import DatasetError
    from fieldbook.launch import LaunchError

    errors: list[type[BaseException]] = [
        ConfigError, DatasetError, LaunchError, snapshot.SnapshotError, storage.StorageError,
        images.registry.RegistryError, aws.AWSError, runpod.RunPodError, images.ImageError,
        urllib.error.URLError, OSError,
    ]  # fmt: skip
    try:
        from botocore.exceptions import BotoCoreError, ClientError

        errors += [BotoCoreError, ClientError]
    except ImportError:
        pass
    return tuple(errors)
