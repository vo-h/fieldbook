"""Tracking (tracking.md): MLflow runs, experiments, and models.

- `store`: where runs log (this machine's SQLite database, or a shared Postgres one) and
  where artifacts go;
- `setup`: the MLflow variables of a run, creating its experiment;
- `plugin`: the tags every MLflow run started in a launched process gets (an MLflow
  entry point);
- `records`: the run records cleanup reads to keep the dataset versions runs used;
- `costs`: the cost of cloud runs, logged to their MLflow runs;
- `server`: the MLflow UI (`fieldbook log ui`).

Nothing is imported here, so MLflow's plugin loads only what it needs.
"""
