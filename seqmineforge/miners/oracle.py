"""Brute-force exhaustive miner -- the ground truth of the whole system.

This module is the *only* implementation that is allowed to be "obviously
correct by inspection".  It enumerates every candidate pattern up to
``max_pattern_length`` over the item universe and keeps the ones whose support
clears the threshold.  No apriori pruning, no projection, no cleverness.

Because the candidate space is combinatorial, the oracle is only usable on
small databases -- which is exactly what makes it valuable: it lets us assert
*exact set equality* against the flagship and every baseline.

Invariants asserted by the accompanying tests
--------------------------------------------
O1. A sequence contains no repeated item (checked by the oracle itself, since
    a duplicate would make "first occurrence" semantics ambiguous).
O2. The oracle result is independent of the order in which sequences are fed.
O3. For every returned pattern, an independent counting pass agrees with the
    support value to within ``1e-12``.
O4. Downward closure: every sub-pattern of a returned pattern with
    ``len >= 2`` is also returned (for the ``SUBSEQUENCE`` semantics this holds
    because a subsequence of a subsequence is a subsequence).

Author: 晨星
"""

from __future__ import annotations

import time
from itertools import permutations

import numpy as np

from ..core.errors import DataGenerationError
from ..core.types import MiningResult, Pattern, PatternType, SequenceDatabase
from .budget import budget_for
from .matching import contains_contiguous, contains_subsequence

__all__ = ["BruteForceMiner", "brute_force_support"]


def brute_force_support(
    pattern: tuple[int, ...],
    sequences: tuple[np.ndarray, ...],
    pattern_type: PatternType = PatternType.SUBSEQUENCE,
) -> int:
    """Count sequences containing ``pattern`` (independent of miner code)."""
    arr = np.asarray(pattern, dtype=np.int64)
    hit = contains_contiguous if pattern_type is PatternType.CONTIGUOUS else contains_subsequence
    return sum(1 for seq in sequences if hit(arr, seq))


class BruteForceMiner:
    """Exhaustive enumeration miner.  ``name = "oracle"``."""

    name = "oracle"

    def __init__(self, *, allow_repeated_items: bool = False) -> None:
        self.allow_repeated_items = bool(allow_repeated_items)

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult:
        start = time.perf_counter()
        if min_support <= 0 or min_support > 1:
            raise ValueError(f"min_support must lie in (0, 1], got {min_support}")
        if max_pattern_length < 1:
            raise ValueError(f"max_pattern_length must be >= 1, got {max_pattern_length}")

        if not self.allow_repeated_items:
            for idx, seq in enumerate(db.sequences):
                if len(np.unique(seq)) != len(seq):
                    raise DataGenerationError(
                        f"sequence #{idx} contains repeated items; the oracle "
                        "requires item-distinct sequences (invariant O1)"
                    )

        n_sequences = db.n_sequences
        # Same canonical budget as every other miner -- see
        # :mod:`seqmineforge.miners.budget`.  In particular the shortest-sequence
        # bound on pattern length is *sound* and must be applied uniformly, or
        # the miners legitimately return longer patterns than the oracle.
        min_count, max_len = budget_for(min_support, max_pattern_length, db)

        if max_len < 1:
            return MiningResult(
                patterns=(),
                pattern_type=pattern_type,
                name=self.name,
                elapsed_sec=time.perf_counter() - start,
                stats={"n_candidates": 0.0, "n_patterns": 0.0, "min_count": float(min_count)},
            )

        hit = (
            contains_contiguous if pattern_type is PatternType.CONTIGUOUS else contains_subsequence
        )

        found: list[Pattern] = []
        n_candidates = 0
        present_items = np.flatnonzero(db.item_support() > 0)
        items = [int(v) for v in present_items]

        for length in range(1, max_len + 1):
            # Both semantics are order-sensitive: `[1, 0]` is a genuinely
            # different contiguous pattern from `[0, 1]`.  (An earlier version
            # enumerated only sorted `combinations` for the contiguous case,
            # which silently reported half the patterns -- the miner was right
            # and the oracle was wrong.)
            for combo in permutations(items, length):
                n_candidates += 1
                count = sum(
                    1 for seq in db.sequences if hit(np.asarray(combo, dtype=np.int64), seq)
                )
                if count >= min_count:
                    found.append(
                        Pattern(tuple(int(v) for v in combo), count / n_sequences, pattern_type)
                    )

        patterns = tuple(sorted(found, key=lambda p: (-p.support, p.items)))
        elapsed = time.perf_counter() - start
        return MiningResult(
            patterns=patterns,
            pattern_type=pattern_type,
            name=self.name,
            elapsed_sec=elapsed,
            stats={
                "n_candidates": float(n_candidates),
                "n_patterns": float(len(patterns)),
                "min_count": float(min_count),
                "max_len": float(max_len),
            },
        )
