"""`fieldbook run --on remote` (fieldbook/launch/cloud.py): resources, command, and environment."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fieldbook import images, storage
from fieldbook.cli import app
from fieldbook.clouds import aws
from fieldbook.config import settings
from fieldbook.images import registry
from fieldbook.launch import budget, cloud, records, repeats, runlog, skypilot, tasks
from fieldbook.tracking import setup

runner = CliRunner()


def test_remote_command() -> None:
    command = cloud._remote_command(
        "r1", "abc", ["train.py", "--lr", "0.1 0.2"], ["--seed", "3"], 1, "src"
    )
    assert "fieldbook log code abc r1" in command and "cd src" in command
    assert (
        command.splitlines()[-1]
        == "fieldbook run --run-id r1 --code abc --seed 3 train.py --lr '0.1 0.2'"
    )
    multi = cloud._remote_command("r1", "abc", ["torchrun", "--nproc-per-node", "4", "t.py"], [], 2)
    assert "torchrun --nnodes=$SKYPILOT_NUM_NODES --node-rank=$SKYPILOT_NODE_RANK" in multi
    single = cloud._remote_command("r1", "abc", ["torchrun", "t.py"], [], 1)
    assert "SKYPILOT_NUM_NODES" not in single


def test_resources_from_task_file_and_flags(tmp_path: Path) -> None:
    (tmp_path / "task.yaml").write_text(
        "resources: {accelerators: A100:8, infra: aws/us-east-1}\nnum_nodes: 2\nrun: python t.py\n"
    )
    task = tasks.load_task(tmp_path / "task.yaml", [])
    cfg = settings()
    resources, nodes, _ = cloud._resources(task, cloud.RemoteOptions(), cfg, "reg/img:t@sha256:1")
    assert resources == {
        "accelerators": "A100:8",
        "infra": "aws/us-east-1",
        "image_id": "docker:reg/img:t@sha256:1",
    }
    assert nodes == 2
    options = cloud.RemoteOptions(gpus="A10G:1", cloud="aws", nodes=1)
    resources, nodes, _ = cloud._resources(task, options, cfg, "x")
    assert (resources["accelerators"], resources["infra"], nodes) == ("A10G:1", "aws", 1)
    assert "max_hourly_cost" not in resources
    # by memory, or several in order, in SkyPilot's own syntax; a price ceiling per machine
    options = cloud.RemoteOptions(gpus="RTX4090, 24GB+", max_price=1.0)
    resources, _, _ = cloud._resources(task, options, cfg, "x")
    assert resources["accelerators"] == ["RTX4090", "24GB+"] and resources["max_hourly_cost"] == 1.0
    resources, _, _ = cloud._resources(task, cloud.RemoteOptions(gpus="48GB+:2"), cfg, "x")
    assert resources["accelerators"] == "48GB+:2"


IMAGE = images.Image(
    "1.dkr.ecr.us-east-1.amazonaws.com/fieldbook:env-1@sha256:1",
    registry.Repository.parse("1.dkr.ecr.us-east-1.amazonaws.com/fieldbook"),
    ("AWS", "registry-password"),
)


def test_credentials_go_by_name_and_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "bucket_region", lambda bucket: "eu-west-1")  # no network
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    monkeypatch.setenv("FIELDBOOK_IMAGE_REGISTRY", "1.dkr.ecr.us-east-1.amazonaws.com/fieldbook")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "postgresql://bob:pw@db.example.com/mlflow")
    monkeypatch.setenv("PGPASSWORD", "pg")
    monkeypatch.setenv("PGUSER", "vision/alice")  # a member's login goes along with it
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("HF_TOKEN", "hf")
    envs, secrets = cloud._run_environment(settings(), cloud.RemoteOptions(), IMAGE)
    assert secrets == {
        "MLFLOW_TRACKING_URI": "postgresql://bob:pw@db.example.com/mlflow",
        "PGPASSWORD": "pg",
        "SKYPILOT_DOCKER_USERNAME": "AWS",
        "SKYPILOT_DOCKER_SERVER": "1.dkr.ecr.us-east-1.amazonaws.com",
        "SKYPILOT_DOCKER_PASSWORD": "registry-password",  # all three secret, or SkyPilot refuses
    }
    assert not any(name.startswith("SKYPILOT_DOCKER") for name in envs)
    assert (
        envs["FIELDBOOK_STORAGE_ROOT"] == "s3://bucket/fieldbook"
        and envs["FIELDBOOK_IMAGE"] == IMAGE.ref
    )
    assert envs["FIELDBOOK_AUTHOR"] == "tester"  # the laptop's author, not the machine's user
    assert envs["PGUSER"] == "vision/alice"
    assert envs["FIELDBOOK_STORAGE_REGION"] == "eu-west-1"  # detected on the laptop, passed along
    options = cloud.RemoteOptions(secrets=["HF_TOKEN"])
    envs, secrets = cloud._run_environment(settings(), options, IMAGE)
    assert secrets["HF_TOKEN"] == "hf" and "AWS_ACCESS_KEY_ID" not in secrets  # the IAM role
    assert envs["FIELDBOOK_SECRETS"] == ",".join(sorted(secrets))  # names only, for masking
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    _, secrets = cloud._run_environment(settings(), options, IMAGE, "runpod")
    assert (secrets["AWS_ACCESS_KEY_ID"], secrets["AWS_SECRET_ACCESS_KEY"]) == ("key", "secret")
    assert skypilot.machine_role(settings(), "runpod") is None  # keys, not a role, off AWS
    with pytest.raises(cloud.RemoteError, match="--secret MISSING"):
        cloud._run_environment(settings(), cloud.RemoteOptions(secrets=["MISSING"]), IMAGE)


def test_cloud_side_of_fieldbook_run_uses_the_given_run_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # a checked-out snapshot: not a git repository
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))  # and no git, as in images
    (tmp_path / "show.py").write_text(
        "import os\nopen('seen.txt', 'w').write(os.environ['PLATFORM_RUN_ID'] + ' ' + "
        "os.environ['PLATFORM_CODE'] + ' ' + os.environ['PLATFORM_CHECKPOINT_DIR'])\n"
    )
    monkeypatch.setattr(setup, "mlflow_env", lambda *args: {})
    result = runner.invoke(app, ["run", "--run-id", "r-given", "--code", "c0ffee", "show.py"])
    assert result.exit_code == 0, result.output
    run_id, code, checkpoints = (tmp_path / "seen.txt").read_text().split()
    assert (run_id, code) == ("r-given", "c0ffee") and checkpoints.endswith("checkpoints/r-given")


def test_remote_options_need_remote_and_a_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "t.py").write_text("")
    result = runner.invoke(app, ["run", "--gpus", "A10G:1", "t.py"])
    assert result.exit_code == 1 and "--gpus apply to --on remote" in result.output
    result = runner.invoke(app, ["run", "--on", "remote", "t.py"])  # local storage root
    assert result.exit_code == 1 and "cloud runs read and write data in a bucket" in result.output
    assert (
        registry.Repository.parse("1.dkr.ecr.eu-west-1.amazonaws.com/fieldbook").region
        == "eu-west-1"
    )
    for other in ("ghcr.io/me/fieldbook", "fieldbook"):
        with pytest.raises(registry.RegistryError, match="ECR repository"):
            registry.Repository.parse(other)
    os.environ.pop("FIELDBOOK_IMAGE_REGISTRY", None)


def test_task_file_cloud_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "task.yaml").write_text(
        "resources: {cloud: aws, region: us-east-2, accelerators: L4:1}\nrun: python t.py\n"
    )
    task = tasks.load_task(tmp_path / "task.yaml", [])
    monkeypatch.setenv("FIELDBOOK_REMOTE_CLOUD", "aws/us-west-2")
    resources, _, chosen = cloud._resources(task, cloud.RemoteOptions(), settings(), "img")
    assert resources["region"] == "us-east-2" and "infra" not in resources  # the file's choice
    resources, _, chosen = cloud._resources(
        task, cloud.RemoteOptions(cloud="RunPod"), settings(), "img"
    )
    assert chosen == "runpod" and resources["infra"] == "RunPod"  # the flag replaces the file's
    assert "cloud" not in resources and "region" not in resources
    plain = tasks.Task(run=["python", "t.py"])
    resources, _, chosen = cloud._resources(plain, cloud.RemoteOptions(), settings(), "img")
    assert resources["infra"] == "aws/us-west-2" and chosen == "aws"  # the setting
    monkeypatch.delenv("FIELDBOOK_REMOTE_CLOUD")
    resources, _, _ = cloud._resources(plain, cloud.RemoteOptions(), settings(), "img")
    assert resources["infra"] == "aws"  # never a cloud SkyPilot picks
    with pytest.raises(cloud.RemoteError, match="only aws and runpod"):
        cloud._resources(plain, cloud.RemoteOptions(cloud="gcp"), settings(), "img")


def test_task_file_in_a_subfolder_runs_from_the_launch_folder() -> None:
    command = cloud._remote_command("r1", "abc", ["tasks/train.yaml"], [], 1, ".")
    assert 'export FIELDBOOK_CODE_ROOT="$PWD"' in command and "cd ." in command
    assert command.splitlines()[-1].endswith("tasks/train.yaml")  # found from the repository root


def test_cloud_side_environment_record_uses_the_checkout_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    (tmp_path / "uv.lock").write_text("version = 1\n")  # at the checkout's root
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "show.py").write_text(
        "import os\nopen('env.json', 'w').write(os.environ['PLATFORM_ENV'])\n"
    )
    monkeypatch.chdir(tmp_path / "tasks")
    monkeypatch.setenv("FIELDBOOK_CODE_ROOT", str(tmp_path))
    monkeypatch.setattr(setup, "mlflow_env", lambda *args: {})
    result = runner.invoke(app, ["run", "--run-id", "r1", "--code", "c0ffee", "show.py"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "tasks" / "env.json").read_text())["uv_lock"]


def test_a_cloud_launch_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D2: what `fieldbook run --on remote` hands SkyPilot, with fakes for the snapshot, the image,
    and SkyPilot itself."""
    from fieldbook.code import snapshot
    from fieldbook.tracking import costs

    repo = tmp_path / "repo"
    (repo / "jobs").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.chdir(repo / "jobs")
    monkeypatch.setenv("FIELDBOOK_STORAGE_ROOT", "s3://bucket/fieldbook")
    monkeypatch.setattr(aws, "bucket_region", lambda bucket: "us-west-2")
    monkeypatch.setattr(storage, "_regions", {})
    monkeypatch.setattr(cloud, "_check_ready", lambda cfg, workdir: None)
    monkeypatch.setattr(
        costs,
        "record_in_background",
        lambda cfg, report: type("T", (), {"join": lambda self, timeout: None})(),
    )
    monkeypatch.setattr(
        cloud, "_snapshot", lambda workdir, cfg, size: snapshot.Snapshot("c0de" * 10, repo, True, 1)
    )
    monkeypatch.setattr(images, "image", lambda repo, cfg, build=True: IMAGE)
    monkeypatch.setattr(skypilot, "machine_role", lambda cfg, cloud: "run-role")
    monkeypatch.setattr(skypilot, "candidates", lambda resources, nodes: [("L4", 0.8)])
    recorded = []
    monkeypatch.setattr(budget, "record", lambda launch, cfg: recorded.append(launch))
    started, ended = [], []
    monkeypatch.setattr(records, "start", lambda cfg, run_id, fields: started.append(fields))
    monkeypatch.setattr(records, "end", lambda cfg, run_id, code, **kw: ended.append((code, kw)))
    kept = type("Log", (), {"__init__": lambda self, *a, **kw: None, "close": lambda self: None})
    monkeypatch.setattr(runlog, "RunLog", kept)
    earlier, marked = [], []
    monkeypatch.setattr(repeats, "find", lambda mark, cfg: earlier[0] if earlier else None)
    monkeypatch.setattr(repeats, "mark", lambda mark, run_id, cfg: marked.append(run_id))
    sent = {}
    monkeypatch.setattr(
        skypilot,
        "launch",
        lambda spec, cluster, **kw: sent.update(spec=spec, cluster=cluster, **kw),
    )
    task = tasks.Task(run=["python", "train.py"], resources={"accelerators": "L4:1"})
    options = cloud.RemoteOptions(run_args=["--seed", "7"])
    cluster = cloud.launch(task, ["train.py", "--lr", "1"], options)
    spec = sent["spec"]
    run_id = cluster.removeprefix("fieldbook-")
    assert (
        sent["cluster"] == cluster == spec["name"] and sent["detach"] and sent["role"] == "run-role"
    )
    assert spec["resources"] == {
        "accelerators": "L4:1",
        "infra": "aws",
        "image_id": f"docker:{IMAGE.ref}",
    }
    command = spec["run"]
    assert (
        f"fieldbook run --run-id {run_id} --code {'c0de' * 10} --seed 7 train.py --lr 1" in command
    )
    assert "cd jobs" in command  # runs from the folder it was launched from
    assert f"timeout -k 60 {24 * 3600} fieldbook run" in command  # no budget set: a day at most
    assert [(r.run_id, r.cluster, r.price, r.limit) for r in recorded] == [
        (run_id, cluster, 0.8, 24 * 3600)
    ]
    assert (
        sent["envs"]["FIELDBOOK_IMAGE"] == IMAGE.ref
        and sent["envs"]["FIELDBOOK_STORAGE_REGION"] == "us-west-2"
    )
    assert sent["secrets"]["SKYPILOT_DOCKER_PASSWORD"] == "registry-password"
    assert sent["envs"]["FIELDBOOK_SECRETS"] == ",".join(sorted(sent["secrets"]))
    # the run's record starts at launch; what SkyPilot prints is kept as its launch log
    assert started[0]["command"] == "train.py --lr 1" and started[0]["cluster"] == cluster
    assert started[0]["where"] == "cloud" and started[0]["limit"] == 24 * 3600
    assert isinstance(sent["log"], kept) and not ended
    # the same again is not launched (it costs money), unless asked for
    assert marked == [run_id]
    earlier.append(repeats.Repeat(run_id, f"run {run_id}; finished; val_loss=0.25"))
    with pytest.raises(
        cloud.RemoteError, match="repeats an earlier run.*val_loss=0.25.*--allow-repeat"
    ):
        cloud.launch(task, ["train.py", "--lr", "1"], options)
    assert len(marked) == 1 and len(recorded) == 1
    options.again = True
    assert cloud.launch(task, ["train.py", "--lr", "1"], options) != cluster


def test_the_command_line_passes_its_options_to_the_cloud_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fieldbook run --on remote` with every option, through the real CLI (the restructuring once
    shadowed the --cloud option with a module of the same name)."""
    from typer.testing import CliRunner

    from fieldbook.cli import app

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "train.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf")
    seen = {}
    monkeypatch.setattr(
        cloud, "launch", lambda task, target, options: seen.update(target=target, options=options)
    )
    result = CliRunner().invoke(app, [
        "run", "--on", "remote", "--cloud", "runpod", "--gpus", "L4:1", "--nodes", "2",
        "--secret", "HF_TOKEN", "--no-build", "--max-hours", "2", "--seed", "7",
        "train.py", "--lr", "1",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    options = seen["options"]
    assert (options.cloud, options.gpus, options.nodes, options.build) == (
        "runpod",
        "L4:1",
        2,
        False,
    )
    assert options.max_hours == 2
    assert options.secrets == ["HF_TOKEN"] and options.run_args[:2] == ["--seed", "7"]
    assert seen["target"] == ["train.py", "--lr", "1"]


def test_the_price_of_a_run_is_its_dearest_candidate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fits = [("RTXA5000", 0.27), ("L4", 0.59), ("RTX4090", 0.89)]
    monkeypatch.setattr(skypilot, "candidates", lambda resources, nodes: fits)
    wanted = {"accelerators": "24GB+", "max_hourly_cost": 1.0}
    assert cloud._price(wanted, 1, "runpod") == 0.89  # any of them may be launched
    assert (
        "3 machines fit (cheapest tried first): RTXA5000 $0.27, L4 $0.59" in capsys.readouterr().err
    )
    monkeypatch.setattr(skypilot, "candidates", lambda resources, nodes: [])
    with pytest.raises(
        cloud.RemoteError, match=r"no machine on runpod has GPUs 24GB\+ under \$1.00"
    ):
        cloud._price(wanted, 1, "runpod")
    assert cloud._price({}, 1, "aws") is None  # nothing asked for, nothing priced
    monkeypatch.setattr(skypilot, "candidates", lambda resources, nodes: None)  # no catalog
    assert cloud._price(wanted, 1, "runpod") is None


def test_a_dry_run_says_what_a_launch_would_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fieldbook.code import snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "train.py").write_text("")
    (repo / "fieldbook.toml").write_text('project = "p"\nbudget_monthly_usd = 10\n')
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cloud, "_check_ready", lambda cfg, workdir: None)
    monkeypatch.setattr(
        cloud, "_snapshot", lambda workdir, cfg, size: snapshot.Snapshot("c0de" * 10, repo, True, 1)
    )
    monkeypatch.setattr(images, "image", lambda repo, cfg, build=True: IMAGE)
    fits = [("RTXA5000", 0.27), ("RTX4090", 0.89)]
    monkeypatch.setattr(skypilot, "candidates", lambda resources, nodes: fits)
    monkeypatch.setattr(repeats, "find", lambda mark, cfg: None)
    for name in ("launch", "machine_role"):  # a dry run must not get this far
        monkeypatch.setattr(skypilot, name, lambda *a, **kw: pytest.fail("launched"))
    args = ["run", "--on", "remote", "--dry-run", "--gpus", "24GB+", "--max-hours", "2"]
    result = runner.invoke(app, [*args, "--json", "train.py"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output[result.output.index("{") :])
    assert plan["would_launch"] and plan["refused"] == [] and plan["image"]["built"]
    assert plan["candidates"][1] == {"machine": "RTX4090", "usd_per_hour": 0.89}
    assert plan["usd_per_hour_max"] == 0.89 and plan["time_limit_seconds"] == 7200
    assert plan["cost_max_usd"] == round(0.89 * (7200 + 600) / 3600, 2)  # with the start-up
    assert plan["budget"] == {"run_usd": None, "monthly_usd": 10, "spent_usd": 0.0,
                              "left_usd": 10.0}  # fmt: skip
    assert not storage.fs(settings()).exists("launches")  # nothing recorded
    text = runner.invoke(app, [*args, "train.py"]).output
    assert "RTXA5000         $0.27 an hour" in text and "time limit 2.0 h, at most $1.93" in text
    assert "$10.00 left" in text and "nothing was launched; it would be" in text
    # what would stop a launch is said, not raised; the exit code tells
    monkeypatch.setattr(repeats, "find", lambda mark, cfg: repeats.Repeat("r9", "run r9; finished"))

    def missing(repo, cfg, build=True):  # noqa: ANN001, ANN202
        raise images.ImageMissing("not built yet")

    result = runner.invoke(app, [*args[:-1], "20", "train.py"])  # 20 hours: over the budget
    assert result.exit_code == 1 and "would not launch: it repeats run r9" in result.output
    assert "would not launch: over the project's budget" in result.output
    monkeypatch.setattr(images, "image", missing)
    result = runner.invoke(app, [*args, "train.py"])
    assert result.exit_code == 0 and "image: not built yet" in result.output  # nothing to repeat
    assert (
        "--dry-run apply to --on remote"
        in runner.invoke(app, ["run", "--dry-run", "train.py"]).output
    )
