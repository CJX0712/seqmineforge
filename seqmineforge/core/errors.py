"""Typed error hierarchy.

Every error carries an ``E<code>`` marker so that callers (and the CLI) can
report precisely which invariant tripped.  Codes are grouped:

======  ==========================================================
Range   Meaning
======  ==========================================================
E1xx    Input / configuration problems
E2xx    Data generation problems
E3xx    Mining / algorithm problems
E4xx    Evaluation & gate problems
E5xx    Backend / dependency problems
======  ==========================================================

Author: 晨星
"""

from __future__ import annotations

__all__ = [
    "BackendUnavailableError",
    "ConfigError",
    "DataGenerationError",
    "EmptyResultError",
    "EvaluationError",
    "GateFailureError",
    "InvalidPatternError",
    "MinerError",
    "SeqMineError",
    "UnknownMinerError",
]


class SeqMineError(Exception):
    """Base class for all SeqMineForge errors."""

    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")
        self.raw_message = message


class ConfigError(SeqMineError):
    """Invalid configuration value (E100)."""

    code = "E100"


class InvalidPatternError(SeqMineError):
    """Pattern is structurally invalid (E101)."""

    code = "E101"


class DataGenerationError(SeqMineError):
    """Synthetic database could not be generated as specified (E200)."""

    code = "E200"


class MinerError(SeqMineError):
    """Generic mining failure (E300)."""

    code = "E300"


class UnknownMinerError(MinerError):
    """Requested miner name is not registered (E301)."""

    code = "E301"


class EmptyResultError(MinerError):
    """A miner returned zero patterns where the contract forbids it (E302)."""

    code = "E302"


class EvaluationError(SeqMineError):
    """Generic evaluation failure (E400)."""

    code = "E400"


class GateFailureError(EvaluationError):
    """A predefined acceptance gate did not pass (E401)."""

    code = "E401"


class BackendUnavailableError(SeqMineError):
    """Optional Tier-0 backend missing or unusable (E500)."""

    code = "E500"
