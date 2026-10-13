"""AWS: credentials, bucket regions, ECR, IAM roles for machines, machine IDs, and cost.

IAM roles (cloud-runs.md §7): without a role, SkyPilot copies the laptop's AWS
credentials to every machine. With one, a machine gets only what it needs. One pair of
roles per AWS account and project:

- `fieldbook-run-<hash>`: objects in the project's folder of the storage root, and deleting
  its own instances (SkyPilot's teardown runs on the machine);
- `fieldbook-build-<hash>`: deleting its own instances only.

Images are pulled and pushed with a registry login from the laptop, so neither role needs
registry permissions. `fieldbook init` creates or updates the roles, after asking; launches only
check that they exist, and go ahead with a warning when they may not read IAM.

Members' groups (sharing.md §2, §6): `fieldbook-<project>-write` and `-read`, kept
in line with the GitHub repository by `fieldbook access sync`.
"""

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from fieldbook.config import Settings

ECR_HOST = re.compile(r"^(\d+)\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$")
IMDS = "http://169.254.169.254/latest"


class AWSError(Exception):
    pass


def client(service: str, region: str | None = None):
    try:
        import botocore.session
    except ImportError:
        raise AWSError("AWS needs botocore: pip install 'fieldbook[aws]'") from None
    return botocore.session.get_session().create_client(service, region_name=region)


# --- regions --------------------------------------------------------------------------


def bucket_region(bucket: str) -> str | None:
    """An S3 bucket's region: S3 reports it in a header even when it refuses the request."""
    request = urllib.request.Request(f"https://{bucket}.s3.amazonaws.com", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as e:
        return e.headers.get("x-amz-bucket-region")
    except OSError:
        return None


def credentials_env() -> dict[str, str]:
    """This machine's AWS credentials (from the environment, ~/.aws, or SSO) as variables,
    for machines on other clouds that read an S3 bucket. SSO and role credentials are
    temporary, so a run that outlives them loses bucket access."""
    try:
        import botocore.session
    except ImportError:
        raise AWSError("AWS needs botocore: pip install 'fieldbook[aws]'") from None
    found = botocore.session.get_session().get_credentials()
    if found is None:
        raise AWSError(
            "no AWS credentials for the S3 bucket: run `aws configure` or `aws sso login`"
        )
    frozen = found.get_frozen_credentials()
    env = {"AWS_ACCESS_KEY_ID": frozen.access_key, "AWS_SECRET_ACCESS_KEY": frozen.secret_key}
    if frozen.token:
        env["AWS_SESSION_TOKEN"] = frozen.token
    return env


def default_region_env(env: dict[str, str], region: str | None) -> dict[str, str]:
    """`AWS_DEFAULT_REGION` set to the bucket's `region`, unless `env` already sets one: a
    run's own S3 code (Lightning, boto3, s3fs) otherwise signs for the default region,
    which a bucket elsewhere refuses as "Access Denied"."""
    if not region or env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION"):
        return {}
    return {"AWS_DEFAULT_REGION": region}


# --- ECR ------------------------------------------------------------------------------


def ecr_region(host: str) -> str | None:
    """The region of an ECR registry host; None for other registries."""
    match = ECR_HOST.match(host)
    return match.group(2) if match else None


def ecr_login(region: str) -> tuple[str, str]:
    """(user name, password) for `docker login`, valid for 12 hours."""
    token = client("ecr", region).get_authorization_token()["authorizationData"][0]
    user, _, password = base64.b64decode(token["authorizationToken"]).decode().partition(":")
    return user, password


def ensure_ecr_repository(region: str, name: str) -> None:
    ecr = client("ecr", region)
    try:
        ecr.describe_repositories(repositoryNames=[name])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=name)


# --- IAM roles for machines -----------------------------------------------------------

POLICY_NAME = "fieldbook"
TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}
# SkyPilot's teardown (autodown) runs on the machine: it looks up the cluster's instances,
# VPC, and security groups, moves the instances to the default security group, and
# terminates them (sky/provision/aws/instance.py terminate_instances).
SELF_MANAGE = [
    {
        "Effect": "Allow",
        "Action": [
            "ec2:DescribeInstances",
            "ec2:DescribeInstanceStatus",
            "ec2:DescribeVpcs",
            "ec2:DescribeSecurityGroups",
            "ec2:DescribeSubnets",
        ],
        "Resource": "*",
    },
    {
        "Effect": "Allow",
        "Action": ["ec2:TerminateInstances", "ec2:StopInstances", "ec2:ModifyInstanceAttribute"],
        "Resource": "*",
        "Condition": {"StringLike": {"ec2:ResourceTag/skypilot-cluster-name": "fieldbook-*"}},
    },
]


@dataclass(frozen=True)
class Role:
    name: str  # role and instance profile
    purpose: str
    policy: dict

    @property
    def document(self) -> str:
        return json.dumps(self.policy, indent=2, sort_keys=True)


def roles_apply(cfg: Settings) -> bool:
    """Whether cloud machines use roles: data in S3 (not R2) and runs on AWS."""
    return cfg.storage_root.startswith("s3://") and not cfg.storage_endpoint


def roles(cfg: Settings) -> tuple[Role, Role]:
    """(run role, build role) for these settings."""
    from fieldbook import storage  # storage imports this module

    location = storage.project_root(cfg)  # the project's folder only (sharing.md §3)
    key = f"{location}\0{cfg.image_registry or ''}"
    suffix = hashlib.sha256(key.encode()).hexdigest()[:8]
    bucket, _, prefix = location.removeprefix("s3://").partition("/")
    prefix = prefix.strip("/")
    objects = f"arn:aws:s3:::{bucket}/{prefix}/*" if prefix else f"arn:aws:s3:::{bucket}/*"
    listing = {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"}
    if prefix:
        listing["Condition"] = {"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}}
    data = [
        listing,
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
                       "s3:ListMultipartUploadParts"],
            "Resource": objects,
        },
    ]  # fmt: skip
    run = Role(f"fieldbook-run-{suffix}", "run machines",
               {"Version": "2012-10-17", "Statement": data + SELF_MANAGE})  # fmt: skip
    build = Role(f"fieldbook-build-{suffix}", "image build machines",
                 {"Version": "2012-10-17", "Statement": SELF_MANAGE})  # fmt: skip
    return run, build


def role_state(role: Role) -> str:
    """`missing`, `outdated` (policy differs from the settings), `ok`, or `unknown` when
    these credentials may not read IAM (common when an admin created the roles)."""
    from botocore.exceptions import ClientError

    iam = client("iam")
    try:  # a missing role fails here too
        current = iam.get_role_policy(RoleName=role.name, PolicyName=POLICY_NAME)["PolicyDocument"]
        iam.get_instance_profile(InstanceProfileName=role.name)
    except iam.exceptions.NoSuchEntityException:
        return "missing"
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("AccessDenied", "AccessDeniedException"):
            return "unknown"
        raise
    if isinstance(current, str):  # URL-encoded JSON in some SDK versions
        from urllib.parse import unquote

        current = json.loads(unquote(current))
    return "ok" if current == role.policy else "outdated"


def ensure_role(role: Role, cfg: Settings) -> None:
    """Create the role and its instance profile, or bring the policy up to date."""
    from fieldbook import storage

    iam = client("iam")
    tags = [
        {"Key": "fieldbook:managed", "Value": "true"},
        {"Key": "fieldbook:storage-root", "Value": storage.project_root(cfg)[:256]},
        {"Key": "fieldbook:purpose", "Value": role.purpose},
    ]
    try:
        iam.get_role(RoleName=role.name)
    except iam.exceptions.NoSuchEntityException:
        description = f"fieldbook {role.purpose} ({storage.project_root(cfg)})"[:1000]
        iam.create_role(RoleName=role.name, AssumeRolePolicyDocument=json.dumps(TRUST), Tags=tags,
                        Description=description)  # fmt: skip
    iam.put_role_policy(
        RoleName=role.name, PolicyName=POLICY_NAME, PolicyDocument=json.dumps(role.policy)
    )
    try:
        profile = iam.get_instance_profile(InstanceProfileName=role.name)["InstanceProfile"]
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=role.name, Tags=tags)
        profile = {"Roles": []}
    if not any(r["RoleName"] == role.name for r in profile["Roles"]):
        iam.add_role_to_instance_profile(InstanceProfileName=role.name, RoleName=role.name)


# --- members' groups -------------------------------------------------------------------


def group_names(project: str) -> tuple[str, str]:
    """(writers, readers) IAM groups of a project."""
    return f"fieldbook-{project}-write", f"fieldbook-{project}-read"


def member_policies(cfg: Settings) -> tuple[dict, dict]:
    """(writers, readers) policies: writers use the project's folder and image repository
    (creating it if no image was built yet), launch machines (SkyPilot's minimal policy,
    docs.skypilot.ai cloud-permissions/aws), pass the project's machine roles, and read
    CloudTrail for costs; readers read the project's folder."""
    from fieldbook import storage

    account = _account(cfg)
    bucket, _, prefix = storage.project_root(cfg).removeprefix("s3://").partition("/")
    listing = {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
               "Condition": {"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}}}  # fmt: skip
    objects = f"arn:aws:s3:::{bucket}/{prefix}/*"
    read = [listing, {"Effect": "Allow", "Action": "s3:GetObject", "Resource": objects}]
    write = [
        listing,
        {"Effect": "Allow", "Resource": objects,
         "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts"]},
        {"Effect": "Allow", "Action": "ec2:RunInstances", "Resource": "arn:aws:ec2:*::image/ami-*"},
        {"Effect": "Allow", "Action": "ec2:RunInstances",
         "Resource": [f"arn:aws:ec2:*:{account}:{kind}/*" for kind in
                      ("instance", "network-interface", "subnet", "volume", "security-group")]},
        {"Effect": "Allow", "Resource": f"arn:aws:ec2:*:{account}:instance/*",
         "Action": ["ec2:TerminateInstances", "ec2:DeleteTags", "ec2:StartInstances",
                    "ec2:CreateTags", "ec2:StopInstances"]},
        {"Effect": "Allow", "Action": "ec2:Describe*", "Resource": "*"},
        {"Effect": "Allow", "Resource": f"arn:aws:ec2:*:{account}:*",
         "Action": ["ec2:CreateSecurityGroup", "ec2:AuthorizeSecurityGroupIngress"]},
        {"Effect": "Allow", "Action": "iam:CreateServiceLinkedRole", "Resource": "*",
         "Condition": {"StringEquals": {"iam:AWSServiceName": "spot.amazonaws.com"}}},
        {"Effect": "Allow", "Action": ["iam:GetRole", "iam:PassRole"],
         "Resource": [f"arn:aws:iam::{account}:role/{r.name}" for r in roles(cfg)]},
        {"Effect": "Allow", "Action": "iam:GetInstanceProfile",
         "Resource": [f"arn:aws:iam::{account}:instance-profile/{r.name}" for r in roles(cfg)]},
        {"Effect": "Allow", "Action": "cloudtrail:LookupEvents", "Resource": "*"},
    ]  # fmt: skip
    if cfg.image_registry:
        host, _, name = cfg.image_registry.partition("/")
        region = ecr_region(host)
        write += [
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Effect": "Allow", "Resource": f"arn:aws:ecr:{region}:{account}:repository/{name}",
             "Action": ["ecr:CreateRepository", "ecr:DescribeRepositories", "ecr:DescribeImages",
                        "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload", "ecr:PutImage"]},
        ]  # fmt: skip
    version = "2012-10-17"
    return {"Version": version, "Statement": write}, {"Version": version, "Statement": read}


def ensure_group(name: str, policy: dict) -> None:
    """Create the group if missing and set its policy."""
    iam = client("iam")
    try:
        iam.get_group(GroupName=name)
    except iam.exceptions.NoSuchEntityException:
        iam.create_group(GroupName=name)
    iam.put_group_policy(GroupName=name, PolicyName=POLICY_NAME, PolicyDocument=json.dumps(policy))


def group_users(name: str) -> set[str]:
    iam = client("iam")
    pages = iam.get_paginator("get_group").paginate(GroupName=name)
    return {user["UserName"] for page in pages for user in page["Users"]}


def user_name(name: str) -> str | None:
    """The IAM user called `name`, as IAM spells it (IAM finds names whatever their case),
    or None if there is none."""
    iam = client("iam")
    try:
        return iam.get_user(UserName=name)["User"]["UserName"]
    except iam.exceptions.NoSuchEntityException:
        return None


def add_to_group(group: str, user: str) -> None:
    client("iam").add_user_to_group(GroupName=group, UserName=user)


def remove_from_group(group: str, user: str) -> None:
    client("iam").remove_user_from_group(GroupName=group, UserName=user)


def _account(cfg: Settings) -> str:
    """The AWS account: the registry's, else the one these credentials belong to."""
    match = ECR_HOST.match((cfg.image_registry or "").partition("/")[0])
    return match.group(1) if match else client("sts").get_caller_identity()["Account"]


def skypilot_config(role: Role, cluster: str) -> dict:
    """A SkyPilot project config giving `cluster` this role (SkyPilot does not allow setting
    it in a task, so each launch passes its own file via SKYPILOT_PROJECT_CONFIG)."""
    return {"aws": {"remote_identity": [{cluster: role.name}]}}


# --- machines and cost (cloud-runs.md §10) ----------------------------------


def machine() -> dict[str, str]:
    """MLflow tags naming this EC2 instance (`fieldbook.machine`, `fieldbook.machine_type`)."""
    instance_id, instance_type = instance()
    return {"fieldbook.machine": instance_id, "fieldbook.machine_type": instance_type}


def instance() -> tuple[str, str]:
    """(instance ID, instance type) from the instance metadata service (IMDSv2); empty
    strings off EC2."""
    try:
        token = _read(urllib.request.Request(
            f"{IMDS}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        ))  # fmt: skip
        headers = {"X-aws-ec2-metadata-token": token}

        def get(path: str) -> str:
            return _read(urllib.request.Request(f"{IMDS}/meta-data/{path}", headers=headers))

        return get("instance-id"), get("instance-type")
    except OSError:
        return "", ""


def _read(request: urllib.request.Request) -> str:
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode()


def cost(tags: dict[str, str], start: float, end: float) -> tuple[float, str] | None:
    """The instance's launch-to-termination time (CloudTrail) times its on-demand price
    (SkyPilot's catalog) times nodes; per-second billing, one-minute minimum. None until
    CloudTrail has both events."""
    times = cloudtrail_times(tags["fieldbook.machine"], tags["fieldbook.region"])
    if times is None:
        return None
    seconds = max(60.0, times[1] - times[0])
    hourly = hourly_price(tags["fieldbook.machine_type"], tags["fieldbook.region"])
    return hourly * int(tags.get("fieldbook.nodes", "1")) * seconds / 3600, "aws-cloudtrail"


def cloudtrail_times(instance_id: str, region: str) -> tuple[float, float] | None:
    """(launch, termination) times of an EC2 instance; None until both are recorded
    (CloudTrail keeps events 90 days and delivers them within about 5 to 15 minutes)."""
    pages = (
        client("cloudtrail", region)
        .get_paginator("lookup_events")
        .paginate(
            LookupAttributes=[{"AttributeKey": "ResourceName", "AttributeValue": instance_id}]
        )
    )
    times: dict[str, float] = {}
    for page in pages:
        for event in page["Events"]:
            name, at = event["EventName"], event["EventTime"].timestamp()
            if name in ("RunInstances", "TerminateInstances"):
                times[name] = min(at, times.get(name, at))
    if "RunInstances" not in times or "TerminateInstances" not in times:
        return None
    return times["RunInstances"], times["TerminateInstances"]


def hourly_price(instance_type: str, region: str) -> float:
    from sky import catalog

    return catalog.get_hourly_cost(instance_type, False, region, None, clouds="aws")
