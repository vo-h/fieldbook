"""Applying the run's seed (runs.md §8).

Python's `random` is seeded at once. NumPy and PyTorch are seeded when they are first
imported, so scripts that never use them do not pay their import time. With
`deterministic`, PyTorch also uses deterministic algorithms, which can be slower and make
some operations raise an error.
"""

import importlib.abc
import importlib.util
import os
import random
import sys
from collections.abc import Callable
from types import ModuleType


def apply(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    hooks: dict[str, Callable[[ModuleType], None]] = {
        "numpy": lambda np: np.random.seed(seed),
        "torch": lambda torch: _seed_torch(torch, seed, deterministic),
    }
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by cuBLAS
    after_import(hooks)


def after_import(hooks: dict[str, Callable[[ModuleType], None]]) -> None:
    """Run each hook once its library is imported: now if it already is, otherwise right
    after its first import. Libraries a script never imports cost nothing."""
    pending = {}
    for name, hook in hooks.items():
        if name in sys.modules:
            hook(sys.modules[name])
        else:
            pending[name] = hook
    if pending:
        sys.meta_path.insert(0, _AfterImport(pending))


def _seed_torch(torch: ModuleType, seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)  # also seeds CUDA devices
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


class _AfterImport(importlib.abc.MetaPathFinder):
    """Runs a hook right after a top-level module finishes importing."""

    def __init__(self, hooks: dict[str, Callable[[ModuleType], None]]):
        self.hooks = hooks
        self._finding = False

    def find_spec(self, name, path=None, target=None):
        hook = self.hooks.get(name)
        if hook is None or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._finding = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        exec_module = spec.loader.exec_module

        def exec_and_hook(module: ModuleType) -> None:
            exec_module(module)
            del self.hooks[name]
            hook(module)

        spec.loader.exec_module = exec_and_hook
        return spec
