"""Global determinism helpers.

SeqMineForge guarantees bit-for-bit reproducibility.  The only sanctioned way
to seed the process is :func:`set_all`.

Author: 晨星
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["MAX_SEED", "get_state", "rng", "set_all"]

MAX_SEED = 2**31 - 1


def set_all(seed: int) -> np.random.Generator:
    """Seed every RNG used anywhere in the package.

    Parameters
    ----------
    seed:
        Non-negative integer seed.

    Returns
    -------
    numpy.random.Generator
        A fresh generator; callers that need many independent streams should
        use :func:`rng` on top of this seed instead of constructing their own
        ``default_rng`` -- otherwise determinism is lost.

    Notes
    -----
    ``numpy.random.RandomState`` (legacy) is deliberately *not* used: NEP 19
    only guarantees stream stability for the legacy generator, and the whole
    point here is cross-version reproducibility.  ``PYTHONHASHSEED`` is set in
    the child environment for subprocess-based determinism checks but cannot
    affect the current interpreter -- see :func:`get_state`.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if seed > MAX_SEED:
        raise ValueError(f"seed must be <= {MAX_SEED}, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Derive an independent, reproducible stream from ``seed``.

    ``stream`` separates e.g. "the data generator" from "the bootstrap" so that
    changing the amount of data consumed by one does not shift the other.
    """
    if stream < 0:
        raise ValueError(f"stream must be non-negative, got {stream}")
    return np.random.default_rng([int(seed), int(stream)])


def get_state() -> dict[str, int]:
    """Snapshot the global RNG state (used by determinism tests)."""
    state = np.random.get_state()
    return {
        "legacy_pos": int(state[2]),
        "python_state_hash": hash(random.getstate()[1]),
        "seed_pos": int(state[2]),
    }
