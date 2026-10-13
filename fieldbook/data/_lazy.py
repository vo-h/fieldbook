"""Lance, imported on first use. Importing it takes over a second (it loads pandas and a
REST client), which every `fieldbook` command and every script importing fieldbook would pay."""

import importlib
import os
from types import ModuleType
from typing import Any

# Lance logs a warning every time a dataset is created. Must be set before Lance loads.
os.environ.setdefault("LANCE_LOG", "error")


class _LazyModule(ModuleType):
    def __getattr__(self, attr: str) -> Any:
        return getattr(importlib.import_module(self.__name__), attr)


lance = _LazyModule("lance")
lance_commit = _LazyModule("lance.commit")  # not loaded by `import lance`
lance_dataset = _LazyModule("lance.dataset")
