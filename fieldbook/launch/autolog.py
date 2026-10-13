"""MLflow autologging for training frameworks, switched on without code (tracking.md §8).

When a run logs to MLflow and one of the frameworks below is imported, MLflow's own
autologging for it is turned on: parameters and metrics are recorded with no logging calls
in the script. Only training frameworks: `mlflow.autolog()` would also hook scikit-learn
(a data transform that fits a scaler would start MLflow runs) and LLM client libraries.
Models and checkpoints are not uploaded automatically; the script logs those it wants.

Standard library only until a framework is imported: the bootstrap calls this for every run.
"""

import importlib
import inspect
import os
from types import ModuleType

from fieldbook.launch import seeding

# imported library -> the MLflow module whose `autolog` covers it
FRAMEWORKS = {
    "lightning": "mlflow.pytorch",
    "pytorch_lightning": "mlflow.pytorch",
    "xgboost": "mlflow.xgboost",
    "lightgbm": "mlflow.lightgbm",
    "keras": "mlflow.keras",
    "tensorflow": "mlflow.tensorflow",
}
# What autologging records: parameters and metrics, quietly; nothing large, nothing extra.
OPTIONS = {
    "log_models": False,
    "log_datasets": False,
    "log_input_examples": False,
    "checkpoint": False,
    "silent": True,
    # Not `disable_for_unsupported_versions`: MLflow's list of tested versions lags the
    # frameworks (Lightning 2.6 with MLflow 3.16), and its patches fall back to the plain
    # call if they fail, so an untested version costs nothing worse than no autologging.
}


def enable() -> None:
    """Arrange autologging for this process, if its run logs to MLflow and the `autolog`
    setting is on (`FIELDBOOK_AUTOLOG`, passed by the launcher)."""
    if not os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("FIELDBOOK_AUTOLOG") == "0":
        return
    seeding.after_import({library: _hook(module) for library, module in FRAMEWORKS.items()})


def _hook(mlflow_module: str):  # noqa: ANN202
    def turn_on(_: ModuleType) -> None:
        try:
            autolog = importlib.import_module(mlflow_module).autolog
            accepted = inspect.signature(autolog).parameters
            autolog(**{name: value for name, value in OPTIONS.items() if name in accepted})
        except Exception:  # autologging is a convenience: never stop the script for it
            pass

    return turn_on
