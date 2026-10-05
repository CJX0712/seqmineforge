"""SeqMineForge -- verified deterministic sequential pattern mining.

Author: 晨星
"""

from __future__ import annotations

from .core import (
    Config,
    MiningResult,
    Pattern,
    PatternType,
    SequenceDatabase,
    load_config,
    rng,
    set_all,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "MiningResult",
    "Pattern",
    "PatternType",
    "SequenceDatabase",
    "__version__",
    "load_config",
    "rng",
    "set_all",
]
