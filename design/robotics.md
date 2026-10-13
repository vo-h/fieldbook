# Robot learning with LeRobot

Status: Designed, not built; decisions open (§9). Written before datasets moved into Lance tables (data.md §14): what it says about stored objects, links, and the download cache (§4) predates `fieldbook data checkout` and is to be revised · Last updated: 2026-10-13

How a robot-learning project (an SO-101 arm, imitation learning, then reinforcement learning) uses fieldbook with [LeRobot](https://huggingface.co/docs/lerobot), Hugging Face's library for robot learning. LeRobot drives the arm, records demonstrations, trains policies, and runs them; fieldbook versions the recordings, launches training on cloud GPUs, and links data, code, runs, policies, and their results on the real robot. fieldbook does not wrap LeRobot's commands (N9): they run as they are, under `fieldbook run`.

## 1. Requirements

| # | Requirement |
|---|---|
| R1 | Recordings are kept private (the project's bucket, not only the public Hub) and versioned; adding episodes stores only what changed. |
| R2 | Each recording session and each training run records what it used and produced: dataset versions, code, environment, command, robot calibration, and cameras. |
| R3 | Training runs on a cloud GPU when the laptop has none (an Apple Silicon Mac cannot run the RL learner, which needs NVIDIA). |
| R4 | Policies are compared by their success on the real robot, each linked to the data and run that produced it. |
| R5 | Reinforcement learning (HIL-SERL) runs with the learner on a cloud GPU and the actor on the laptop the arm is plugged into, as one run. |
| R6 | LeRobot's tools are used as they are; fieldbook adds no wrappers around them. |

## 2. How a LeRobot project works

| Step | LeRobot | Where |
|---|---|---|
| Teleoperate and record demonstrations | `lerobot-record --robot.type=so101_follower --teleop.type=so101_leader --dataset.repo_id=… --dataset.root=…` (`--resume=true` adds episodes) | The laptop, with the arm on USB |
| Dataset (v3.0) | A folder: `data/` (Parquet: joint states, actions, timestamps; many episodes per file), `videos/<camera>/` (MP4, many episodes per file), `meta/info.json` (features, fps, robot type, path templates), `meta/stats.json`, `meta/tasks`, `meta/episodes/` (per-episode offsets). By default in `~/.cache/huggingface/lerobot/<repo_id>` and pushed to the Hub | Laptop, Hub |
| Train an imitation policy (ACT, Diffusion, SmolVLA, Pi0) | `lerobot-train --dataset.repo_id=… --dataset.root=… --policy.type=act --output_dir=… --policy.device=cuda`; metrics to Weights & Biases only (`--wandb.enable=true`); checkpoints in `<output_dir>/checkpoints/<step>/pretrained_model/` | A GPU (CUDA, or Apple Silicon with `mps`); "several hours" |
| Run a policy on the arm | `lerobot-rollout --policy.path=… --strategy.type=episodic` (records evaluation episodes with `--dataset.repo_id`) | Laptop and arm |
| RL (HIL-SERL) | A learner (`python -m lerobot.rl.learner --config_path …`: SAC, opens a gRPC server) and an actor (`python -m lerobot.rl.actor …`: drives the arm, connects to the learner); a human intervenes with a gamepad or the leader arm; rewards from a trained success classifier (`lerobot-train` with a reward-classifier config) | Learner: an NVIDIA GPU. Actor: the machine with the arm |

## 3. What works today

| Need | With fieldbook now |
|---|---|
| Train on a cloud GPU, with code, environment, command, and cost recorded | `fieldbook run --on remote --gpus L4:1 lerobot-train …` (`fieldbook run` runs any command on the project environment's path). Images need `image_apt = ["ffmpeg"]` for video decoding. |
| Version a recorded dataset privately | `fieldbook data add so101-cubes ~/.cache/huggingface/lerobot/me/cubes`: files are stored once by content, so a new session that rewrites the last data and video files and `meta/` stores only those. |
| Several people on one project | Shared projects (sharing.md). |

What blocks the rest: LeRobot reads a dataset only from a local folder or the Hub, and fieldbook cannot produce a dataset version as a folder (§4.1); recordings are not saved as versions by the session that made them (§4.2); LeRobot's metrics go only to Weights & Biases (§4.4); and the RL learner and actor must run on two machines that talk to each other (§4.6).

## 4. Features

### 4.1 Check out a dataset version as a folder (general)

`fieldbook data checkout so101-cubes@4 DIR` and `fb.dataset("so101-cubes", version=4).checkout(DIR)` write the version's files into `DIR`, ready for any tool that reads folders.

- **Copies, not links:** recording with `--resume` writes into the dataset folder, so links into the object store could change stored files. On local disk files are cloned (APFS `clonefile`, Linux reflinks: instant, no extra space until changed) and copied where cloning is unavailable; from a bucket they are downloaded in parallel.
- **Download cache:** on a machine reading from a bucket, files are kept by hash in the project's `.fieldbook/cache/objects/`, so a second checkout of a related version downloads only new files.
- **Lineage:** a checkout inside a run records `so101-cubes@4` as the run's input, as a read does. `DIR/.fieldbook-checkout.json` records the dataset and version, so §4.2 knows what a changed folder was based on.
- **Incremental:** checking out into a folder that already holds an earlier version replaces only files that differ.

### 4.2 Save a run's output folder as a dataset version (general)

`fieldbook run --save DIR=so101-cubes COMMAND …` saves `DIR` as a new version of `so101-cubes` after the command succeeds: new files added, changed files updated, missing files removed (an `Edit.sync_dir`). The version is the run's output, so lineage links it to the session's code, command, and inputs. With §4.1 a recording session is:

```bash
fieldbook data checkout so101-cubes data/cubes                      # the latest version, to add to
fieldbook run --save data/cubes=so101-cubes lerobot-record --resume=true \
    --dataset.root=data/cubes --dataset.push_to_hub=false --dataset.num_episodes=10 …
```

### 4.3 LeRobot dataset details (LeRobot-specific)

When a saved folder holds `meta/info.json` with a LeRobot `codebase_version`, its version gets dataset info: fps, robot type, cameras (video features), episode and frame counts, and tasks. The robot's calibration file, found from the session's command (`--robot.type`, `--robot.id`: `~/.cache/huggingface/lerobot/calibration/robots/<type>/<id>.json`), is stored with the version, since a policy trained on one calibration may not transfer to a recalibrated arm. `fieldbook data ls so101-cubes` then shows episodes and tasks per version, not only files.

### 4.4 Training metrics in MLflow (LeRobot-specific)

`lerobot-train` logs to Weights & Biases only. Under `fieldbook run`, fieldbook's bootstrap also sends what LeRobot logs (`WandBLogger.log_dict`: loss, gradient norm, learning rate, step time) to the run's MLflow run, with Weights & Biases on or off; when Weights & Biases is on, its run link is a tag. The hook is a small patch of one LeRobot method, checked against the LeRobot version in `uv.lock` and skipped with a warning if that method changes. (Decision 1.)

### 4.5 Policies and evaluation on the robot (LeRobot-specific)

- **Policies:** after a successful `lerobot-train` run, its last `pretrained_model/` folder is logged to the MLflow run and registered as a model version (`so101-cubes-act`), so the model registry links each policy to its run and the run to its dataset version.
- **Using one:** `--policy.path=models:/so101-cubes-act/3` in an `fieldbook run` command is downloaded to a local folder and replaced by its path, since LeRobot reads local folders or the Hub. The run records the model version as an input.
- **Evaluation:** `fieldbook run --save data/eval=so101-cubes-eval lerobot-rollout --strategy.type=episodic --policy.path=models:/so101-cubes-act/3 …` records evaluation episodes. Success per episode is entered when the session ends (`fieldbook run` asks: "episode 1 succeeded? [y/n]", or `--success 1,1,0,1`) and logged as `success_rate` on the run, so the MLflow UI compares policies by real-robot success. (Decision 3.)

### 4.6 Reinforcement learning across two machines (LeRobot-specific)

A task file describes the two halves; `fieldbook run rl.yaml` runs them as one run:

```yaml
# rl.yaml
learner:                       # SAC learner on a cloud GPU
  on: remote
  gpus: L4:1
  command: python -m lerobot.rl.learner --config_path configs/train_hilserl.json
  port: 50051                  # its gRPC server
actor:                         # drives the arm, here
  command: python -m lerobot.rl.actor --config_path configs/train_hilserl.json
```

- fieldbook launches the learner's cluster, waits until the port answers, opens an SSH tunnel from the laptop to it (`ssh -N -L 50051:localhost:50051 <cluster>`; SkyPilot sets up SSH to its clusters), and starts the actor locally pointed at `localhost:50051`. The learner's gRPC port is never open to the internet: LeRobot's gRPC has no authentication.
- One run ID, one MLflow run (both halves' metrics, §4.4), the demonstrations and replay data as inputs, the learner's checkpoints in the bucket (`PLATFORM_CHECKPOINT_DIR`), and the cloud cost of the learner.
- Ending the actor (or Ctrl-C) stops the learner and deletes its cluster; a lost tunnel is reopened, and the run stops if it cannot be.

## 5. Where the code lives

- §4.1 and §4.2 are general (any tool that reads or writes folders): `fieldbook/data/` and `fieldbook run`.
- §4.3–§4.6 are LeRobot-specific: `fieldbook/integrations/lerobot/`, installed with the extra `fieldbook[lerobot]`, active only when LeRobot is installed in the project. (Decision 4.)

## 6. Order and estimate

| Order | Work | Estimate |
|---|---|---|
| 0 | Spike: install `lerobot` and `fieldbook` in one environment (LeRobot pins torch, `datasets`, and video libraries); record one episode and train ACT for a few steps under `fieldbook run`, locally (`mps`) and on a cloud GPU | 0.5 day |
| 1 | Checkout (§4.1) | 1 day |
| 2 | Save a run's output folder (§4.2) | 1 day |
| 3 | LeRobot dataset details and calibration (§4.3) | 0.5 day |
| 4 | Training metrics in MLflow (§4.4) | 0.5 day |
| 5 | Policies, `models:/` arguments, evaluation with success rates (§4.5) | 1–1.5 days |
| 6 | RL across two machines (§4.6), checked with the arm and a cloud GPU | 2 days |

Steps 0–3 (about 3 days) cover imitation learning end to end; about 6.5–7 days in all.

## 7. Risks

- **LeRobot changes quickly** (v2.1 to v3.0 changed the dataset layout; scripts were renamed). §4.3 and §4.4 depend on its file layout and one method: pinned by `uv.lock`, checked by tests against that version, and they warn rather than fail when it changes.
- **RL over the internet:** the actor sends transitions and receives weights every few seconds (`policy_parameters_push_frequency`, default 4 s). A distant cloud region adds latency; to be measured in step 6, choosing the region nearest the laptop.
- **Large video datasets on a laptop:** checkout copies (or clones) files; a 50 GB dataset needs 50 GB free unless the disk supports cloning.
- **Calibration found from the command line:** if a session sets the robot in a config file instead, §4.3 reads it there too, or records none with a note.

## 8. Not built

LeRobot's recorder, teleoperation, policies, training loop, Hub upload, and visualization (`lerobot-dataset-viz`, Rerun) are used as they are. LeRobot's own Lance dataset layout (`lerobot-lancedb`) may later replace checkout for training reads, since fieldbook's indexes are Lance tables already.

## 9. Open decisions

| # | Question | Recommendation |
|---|---|---|
| 1 | Training metrics: MLflow (with the §4.4 hook), Weights & Biases (link only), or both? | Both: MLflow for the comparison with data and success rates; Weights & Biases stays available for its live dashboards. |
| 2 | Datasets: the project's bucket only, or also pushed to the Hub? | The bucket by default (`--dataset.push_to_hub=false`); push to the Hub only to publish. |
| 3 | Success per evaluation episode: asked at the end of the session, given as a list, or labelled later from the videos? | Asked at the end of the session, with `--success` for scripts; relabelling later is a dataset edit. |
| 4 | LeRobot support in fieldbook's core, or an optional extra? | An extra (`fieldbook[lerobot]`); the general parts (§4.1, §4.2) in the core. |
| 5 | RL learner: cloud GPU (as designed) or a local NVIDIA machine? | Cloud GPU, nearest region; a local NVIDIA machine works with the same task file and `on: local`. |

Sources: [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3), [imitation learning on real robots](https://huggingface.co/docs/lerobot/en/il_robots), [HIL-SERL](https://huggingface.co/docs/lerobot/en/hilserl) (checked 2026-10-09).
