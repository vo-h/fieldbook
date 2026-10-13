# Storage

Status: Built; checked on local disk and a real AWS S3 bucket · Last updated: 2026-10-09

Where everything a project stores lives, and the one interface all reads and writes go through. Code: `fieldbook/storage.py`. What is stored there is described in [data.md](data.md) (datasets), [runs.md](runs.md) (code snapshots, package lists), [tracking.md](tracking.md) (MLflow artifacts), and [cloud-runs.md](cloud-runs.md) (checkpoints).

## 1. Storage root and project folders

The **storage root** (`storage_root` setting) is a folder on local disk or an AWS S3 bucket. Each project keeps everything in its own folder of it, named after the project, so a team's projects share one root and each project's members can be given access to its folder alone ([sharing.md](sharing.md) §3):

```
<root>/<project>/                  # everything of one project; <root> may hold several
  datasets/<name>.lance            # one Lance table per dataset: its files or rows, and every version (data.md)
  tmp/                             # staging area for code snapshots being written
  code/<tree-hash>.tar.gz          # code snapshots (runs.md §4)
  packages/<hash>.txt              # installed-package lists of runs (runs.md §5)
  mlflow/artifacts/                # MLflow artifacts (tracking.md §2)
  checkpoints/<run-id>/            # run checkpoints (cloud-runs.md §8)
  scratch/<id>/datasets/...        # scratch areas of smoke runs and reruns (runs.md §9, §17)
  notes/<run ID>/<time>-<id>.json  # notes on runs (runs.md §15)
  fingerprints/<hash>/<run ID>     # one empty file per run, to notice repeats (runs.md §16)
  launches/<run ID>.json, .machine.json, .ended, .cost.<USD>   # cloud launches and what they cost (cloud-runs.md §10)
  runs/<run ID>.json, .alive, .end.<code>   # a run's record (runs.md §18)
  runs/<run ID>/                   # its output, launch log, and the files it wrote (runs.md §18)
```

- Without a `storage_root`, the root is `<project>/.fieldbook/data` on this machine ([projects.md](projects.md) §3).
- A dataset is its table and nothing else: its files are rows of it, not objects of their own (data.md §1).
- Projects do not read each other's folders ([sharing.md](sharing.md) §4).

```python
fs = fb.storage.fs()                  # fsspec filesystem bound to the project's folder
folder = fb.storage.project_root()    # e.g. "s3://acme-data/vision"
```

All access goes through URIs and fsspec, never hard-coded local paths, so the same code and data formats work on local disk and in a bucket.

## 2. Local disk and S3

| Root | `storage_root` | Notes |
|---|---|---|
| Local disk | A path, or `file:///path` | Default: the project's `.fieldbook/data` |
| AWS S3 | `s3://bucket[/prefix]` | Needs `fieldbook[s3]` (s3fs, maintained by the fsspec project). `storage_endpoint` names an S3-compatible endpoint instead of AWS (used by tests) |

With a bucket root, the laptop and cloud machines read and write the same datasets, snapshots, and artifacts, with no upload or sync step.

**Region.** `storage_region` is optional on AWS: the bucket's region is detected (S3 reports it in a response header, even to a refused request) once per machine and kept in the project's `.fieldbook/bucket-regions.json`; runs inherit it as `FIELDBOOK_STORAGE_REGION`. Requests signed for another region are refused as "Access Denied", so a bucket outside the default region would otherwise fail. Runs also get `AWS_DEFAULT_REGION` set to the bucket's region unless the user set a region, so the run's own S3 code (boto3, s3fs, Lightning) reaches the bucket.

**Errors.** A dataset is reported missing only when the store says "not found"; credential, permission, and network errors are reported as such.

## 3. Credentials

fieldbook never stores storage credentials; they come from AWS's standard sources: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `~/.aws` profiles (`AWS_PROFILE`), SSO, or a machine's IAM role. This works with SSO, cloud machine roles, and company-managed logins. Cloud machines on AWS get an IAM role limited to the project's folder ([cloud-runs.md](cloud-runs.md) §7).

## 4. What stays on the machine

SQLite needs a local disk, so this machine's MLflow run database is not under a bucket root: it lives in the project's local state folder (`state_dir`, default the project's `.fieldbook/`; [projects.md](projects.md) §3). MLflow artifacts (models, plots) stay under the storage root, where they sit next to the data, survive losing the laptop, and are readable by cloud machines. Runs from several machines are seen in one place through a shared tracking database ([tracking.md](tracking.md) §1).

## 5. How each layer uses the root

| Layer | Used for | How |
|---|---|---|
| fsspec | Code snapshots, package lists, run records, scratch areas, cleanup | `storage.fs()` passes the endpoint and region to s3fs |
| Lance | Datasets: tables, files, and commits | Every `lance.dataset()` and commit gets `storage_options` (endpoint, region) from `storage.lance_options()`. Lance reads credentials from the standard sources itself |
| DuckDB | `fieldbook data query` | Reads a dataset through its Lance table, so it uses Lance's access; DuckDB itself opens nothing in the bucket |
| Other tools | PIL, Hugging Face, PyTorch | `ds.open(path)` returns a file object, `ds.read_many(paths)` bytes, and `ds.checkout(dir)` real files (data.md §10); a file has no address in the bucket |
| Cleanup | Freeing space | Datasets: Lance deletes the data files of deleted versions. Code snapshots and package lists: by each object's last-modified time; one that is used again is refreshed (on S3 by copying it onto itself: one request, nothing downloaded) |

## 6. Commit safety and the bucket check

Dataset commits are safe with several writers (laptops and cloud machines) only if the store enforces "write only if absent" (`If-None-Match: *`); Lance relies on it. AWS S3 enforces it. A store that accepts the header without enforcing it loses committed versions silently ([lance#9762](https://github.com/lance-format/lance/issues/9762)), which is why other S3-compatible stores are not supported without a check.

`fieldbook init` checks a bucket root (`storage.check`): credentials work; write, read, list, and delete succeed in the project's folder; and a second conditional write to the same key is rejected. On failure it reports which step failed and the error (missing permission, wrong endpoint, missing bucket), and removes its probe file. A project's viewer, whose access is read-only, is not failed: when the write is refused the check lists the folder instead and reports read-only access ([sharing.md](sharing.md) §2).

## 7. Consequences of a bucket root

| Area | Consequence |
|---|---|
| Speed | Every read from a laptop goes over the internet. Heavy training reads belong in cloud runs, near the bucket |
| Cost | AWS charges for data leaving its cloud. Reading a dataset on a laptop costs that; reading it on an AWS machine in the bucket's region does not |
| Cleanup | Versions kept because runs used them are found in the tracking database: with a shared database that covers every machine; with this machine's database, only its runs |

## 8. Other providers

Only AWS S3 is supported, so nothing is claimed that was never run. Google Cloud Storage, Azure Blob Storage, and Cloudflare R2 were designed (all enforce the conditional write Lance needs) and deferred: adding one means a storage scheme here, a module in `fieldbook/clouds/`, an extra for its fsspec library, and tests against the real service.

## 9. Testing

S3 is tested with moto in server mode (`tests/test_storage_bucket.py`): datasets, commits under concurrency, cleanup, queries, the conditional-write check, and the read-only check. The same paths were checked on a real S3 bucket, including concurrent writers ([data.md](data.md) §6).
