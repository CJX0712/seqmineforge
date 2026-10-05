"""Protocols (interfaces) every miner and feature extractor satisfies.

Modules depend on these protocols, never on concrete classes, so the pipeline
can swap in an offline Tier-1 implementation at runtime without touching the
callers.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import MiningResult, PatternType, SequenceDatabase

__all__ = ["Miner", "OracleMiner", "PatternFeatureExtractor"]


@runtime_checkable
class Miner(Protocol):
    """A sequential pattern miner.

    Contract
    --------
    * ``min_support`` is a **relative** support in ``(0, 1]`` -- this is a
      single, mandatory interpretation across every implementation.
    * The returned result must contain **every** pattern whose support is
      ``>= min_support`` and whose length is ``<= max_pattern_length`` -- no
      sampling, no top-k truncation.  Truncation would silently break the
      oracle equivalence gate.
    * A pattern appearing in the result at exactly ``min_support`` must not be
      duplicated.
    * ``elapsed_sec`` must measure only the mining work, excluding database
      construction (the caller times the DGP separately).
    """

    name: str

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult: ...


@runtime_checkable
class OracleMiner(Protocol):
    """Exhaustive ground-truth miner used only for verification.

    Deliberately naive: it enumerates candidate patterns combinatorially.  Its
    only virtue is *obviously correct*.
    """

    name: str

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult: ...


@runtime_checkable
class PatternFeatureExtractor(Protocol):
    """Turns patterns into a fixed-width numeric feature matrix.

    For sequence ``x`` the extractor returns ``phi(x) in R^{n_patterns}``.  The
    feature convention (binary occurrence vs count vs max-gap) is fixed by the
    implementation and documented in its docstring -- it is part of the
    contract because downstream linear models are only comparable when all
    extractors use the *same* convention.
    """

    name: str

    def transform(
        self, sequences: tuple[np.ndarray, ...], patterns: list[tuple[int, ...]]
    ) -> np.ndarray: ...
