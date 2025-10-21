"""Small, dependency-light helpers shared across the package."""

from __future__ import annotations

import importlib
import os
import random
from types import ModuleType

from harken.exceptions import DependencyError


def require(module: str, extra: str | None = None) -> ModuleType:
    """Import ``module`` or raise a helpful :class:`DependencyError`.

    Parameters
    ----------
    module:
        The importable module name, e.g. ``"transformers"``.
    extra:
        The name of the optional-dependency extra that provides it, used to
        build the install hint (``pip install harken[<extra>]``).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        hint = f"harken[{extra}]" if extra else module
        raise DependencyError(
            f"'{module}' is required for this feature. Install it with "
            f"`pip install {hint}`."
        ) from exc


def seed_everything(seed: int) -> int:
    """Seed Python and (if available) numpy / torch RNGs.

    Returns the seed so it can be logged. torch is imported lazily so the
    helper stays usable in environments without it.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed
