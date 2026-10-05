"""Configuration with ``ENV_SMF_*`` overrides and schema validation.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, get_type_hints

from .errors import ConfigError

__all__ = ["ENV_PREFIX", "Config", "load_config"]

ENV_PREFIX = "ENV_SMF_"

_BOOL_TRUE = ("1", "true", "yes", "on")
_BOOL_FALSE = ("0", "false", "no", "off")


@dataclass(frozen=True)
class Config:
    """Global configuration.

    All fields can be overridden through environment variables named
    ``ENV_SMF_<FIELD_NAME_UPPER>`` with float / int / bool / str parsing.
    Validation happens in ``__post_init__`` so an invalid environment can never
    leak into a benchmark.

    Every declared field type is a builtin (``int`` / ``float`` / ``str`` /
    ``bool``); this invariant is asserted by a unit test because
    :func:`load_config` relies on it for coercion.
    """

    min_support: float = 0.05
    max_pattern_length: int = 4
    n_items: int = 12
    pattern_type: str = "subsequence"
    seed: int = 20261006
    #: Seeds used for the efficiency measurement (the G2 gate).
    n_seeds: int = 2
    n_bootstrap: int = 20
    # --- efficiency gate ---
    speedup_threshold: float = 1.5
    # --- downstream utility gate ---
    utility_noninferiority_tol: float = 1e-9
    compression_min_ratio: float = 3.0
    # --- runtime budget ---
    demo_time_budget_sec: float = 60.0
    # --- oracle feasibility ---
    #: Maximum number of *candidate* patterns the exhaustive oracle may examine.
    #: Beyond this the oracle is combinatorially infeasible (it is
    #: ``O(P(m,k) * n)`` in Python), so G1's exact-equality half is measured on
    #: the small :data:`~seqmineforge.pipeline.pipeline.VERIFY_SPECS` datasets
    #: instead, and full-scale runs rely on recount + cross-implementation
    #: agreement.  20 000 is chosen from measurement: at 26 404 candidates
    #: (14 items, length 4) the oracle alone cost ~8 s per run, i.e. >70 s across
    #: the 9-run benchmark -- over the whole runtime budget for evidence that the
    #: verification-scale run already provides exactly.
    oracle_candidate_budget: int = 20_000

    def __post_init__(self) -> None:
        if not 0.0 < self.min_support <= 1.0:
            raise ConfigError(f"min_support must lie in (0, 1], got {self.min_support}")
        if self.max_pattern_length < 1:
            raise ConfigError(f"max_pattern_length must be >= 1, got {self.max_pattern_length}")
        if self.n_items < 2:
            raise ConfigError(f"n_items must be >= 2, got {self.n_items}")
        if self.pattern_type not in ("subsequence", "contiguous"):
            raise ConfigError(
                f"pattern_type must be 'subsequence' or 'contiguous', got {self.pattern_type!r}"
            )
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")
        if self.n_bootstrap < 2:
            raise ConfigError(f"n_bootstrap must be >= 2, got {self.n_bootstrap}")
        if self.n_seeds < 2:
            raise ConfigError(
                f"n_seeds must be >= 2 (a single seed is not a measurement), got {self.n_seeds}"
            )
        if self.speedup_threshold <= 1.0:
            raise ConfigError(f"speedup_threshold must be > 1.0, got {self.speedup_threshold}")
        if self.compression_min_ratio < 1.0:
            raise ConfigError(
                f"compression_min_ratio must be >= 1.0, got {self.compression_min_ratio}"
            )
        if self.demo_time_budget_sec <= 0:
            raise ConfigError(f"demo_time_budget_sec must be > 0, got {self.demo_time_budget_sec}")
        if self.oracle_candidate_budget < 1:
            raise ConfigError(
                f"oracle_candidate_budget must be >= 1, got {self.oracle_candidate_budget}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(raw: str, target: type) -> Any:
    """Parse ``raw`` into the declared ``target`` builtin type."""
    if target is bool:
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigError(f"cannot parse boolean from {raw!r}")
    if target is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse int from {raw!r}") from exc
    if target is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse float from {raw!r}") from exc
    if target is str:
        return raw
    raise ConfigError(f"unsupported config field type {target!r}")


def load_config(**overrides: Any) -> Config:
    """Build a :class:`Config` from defaults + ``ENV_SMF_*`` + kwargs.

    Precedence: ``kwargs`` > environment > defaults.
    """
    # ``from __future__ import annotations`` turns ``f.type`` into a *string*,
    # so resolve annotations explicitly instead of trusting ``dataclasses``.
    spec = dict(get_type_hints(Config).items())
    values: dict[str, Any] = {}

    for name, typ in spec.items():
        raw = os.environ.get(ENV_PREFIX + name.upper())
        if raw is not None:
            values[name] = _coerce(raw, typ)

    for key, val in overrides.items():
        if key not in spec:
            raise ConfigError(f"unknown config field {key!r}; known: {sorted(spec)}")
        if val is None:
            continue
        target = spec[key]
        values[key] = _coerce(str(val), target) if isinstance(val, str) else target(val)

    return Config(**values)
