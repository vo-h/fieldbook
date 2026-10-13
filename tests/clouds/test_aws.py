"""AWS (fieldbook/clouds/aws.py): IAM roles for cloud machines, against moto's in-process IAM."""

import json
from pathlib import Path

import pytest

from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.launch import skypilot


def test_roles_are_named_per_project_and_scoped(mock_aws, monkeypatch: pytest.MonkeyPatch) -> None:
    run, build = aws.roles(settings())
    assert run.name.startswith("fieldbook-run-") and build.name.startswith("fieldbook-build-")
    assert run.name.split("-")[-1] == build.name.split("-")[-1]  # one pair per project
    text = json.dumps(run.policy)
    assert (
        "arn:aws:s3:::data-bucket/fieldbook/test/*" in text and "s3:::*" not in text
    )  # its folder
    monkeypatch.setenv("FIELDBOOK_PROJECT", "audio")  # another project in the same bucket
    assert aws.roles(settings())[0].name != run.name
    assert "s3:" not in json.dumps(build.policy)  # builds need no data
    for role in (run, build):  # both can delete their own (fieldbook-*) instances
        assert "fieldbook-*" in json.dumps(role.policy) and "ec2:TerminateInstances" in json.dumps(
            role.policy
        )
    assert aws.roles_apply(settings())


def test_roles_do_not_apply_off_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    monkeypatch.setenv("FIELDBOOK_STORAGE_ENDPOINT", "http://127.0.0.1:9000")  # not AWS
    assert not aws.roles_apply(settings())
    monkeypatch.delenv("FIELDBOOK_STORAGE_ENDPOINT")
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", str(Path.cwd()))  # local storage
    assert not aws.roles_apply(settings())


def test_created_once_and_updated_when_settings_change(
    mock_aws, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = aws.roles(settings())
    assert aws.role_state(run) == "missing"
    with pytest.raises(skypilot.SkyPilotError, match="run `fieldbook init`"):
        skypilot.machine_role(settings(), "aws")
    aws.ensure_role(run, settings())
    assert aws.role_state(run) == "ok"
    aws.ensure_role(run, settings())  # idempotent
    assert aws.role_state(run) == "ok" and skypilot.machine_role(settings(), "aws") == run
    changed = aws.Role(
        run.name, run.purpose, {"Version": "2012-10-17", "Statement": aws.SELF_MANAGE}
    )
    assert aws.role_state(changed) == "outdated"


def test_launches_without_permission_to_read_iam(mock_aws, monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError

    class Denied:
        exceptions = type("E", (), {"NoSuchEntityException": KeyError})

        def get_role_policy(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetRolePolicy")

    monkeypatch.setattr(aws, "client", lambda service, region=None: Denied())
    run, _ = aws.roles(settings())
    assert aws.role_state(run) == "unknown"
    assert skypilot.machine_role(settings(), "aws") == run  # with a warning
