"""SeqMineForge -- a verified, deterministic sequential pattern mining system.

Author: 晨星
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config
from .errors import (
    BackendUnavailableError,
    ConfigError,
    DataGenerationError,
    EmptyResultError,
    EvaluationError,
    GateFailureError,
    InvalidPatternError,
    MinerError,
    SeqMineError,
    UnknownMinerError,
)
from .seed import rng, set_all
from .types import MiningResult, Pattern, PatternType, SequenceDatabase, SupportKind

__all__ = [
    "BackendUnavailableError",
    "Config",
    "ConfigError",
    "DataGenerationError",
    "EmptyResultError",
    "EvaluationError",
    "GateFailureError",
    "InvalidPatternError",
    "MinerError",
    "MiningResult",
    "Pattern",
    "PatternType",
    "SeqMineError",
    "SequenceDatabase",
    "SupportKind",
    "UnknownMinerError",
    "__version__",
    "load_config",
    "rng",
    "set_all",
]
