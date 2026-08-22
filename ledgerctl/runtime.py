"""Logging configuration and reproducible seeding."""
from __future__ import annotations

import logging
import random
import sys

import numpy as np

from ledgerctl.config import CONFIG

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stderr handler with a consistent format.

    Args:
        level: Root logging level.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)


def set_seed(seed: int | None = None) -> int:
    """Seed every stochastic source used by the package.

    Args:
        seed: Explicit seed; falls back to the configured seed.

    Returns:
        The seed that was applied.
    """
    resolved = CONFIG.seed if seed is None else seed
    random.seed(resolved)
    np.random.seed(resolved)
    return resolved
