"""MLflow integration: tags every MLflow run started in a launched process
(tracking.md §3). Loaded by MLflow through the `mlflow.run_context_provider`
entry point, so MLflow is imported only by processes that already use it."""

from mlflow.tracking.context.abstract_context import RunContextProvider

from fieldbook.data import lineage
from fieldbook.launch import context


class PlatformRunContext(RunContextProvider):
    def in_context(self) -> bool:
        return bool(context.current().run_id) or bool(lineage.run_tags())

    def tags(self) -> dict[str, str]:
        return lineage.run_tags()
