"""Shared mining-budget arithmetic.

Every miner must agree on *exactly* what ``(min_support, max_pattern_length)``
means, otherwise "the flagship and the oracle return the same set" is a
statement about an accidental convention rather than about the search.

Two derived quantities are centralised here:

``support_threshold(min_support, n_seq)``
    The minimum **integer** sequence count that clears the threshold.  Support
    is a count, so the exact cut is ``ceil(min_support * n_seq)``.  Comparing
    the float support against the float threshold instead admits patterns one
    ULP below the cut -- the classic "one spurious extra pattern" diff.

``effective_max_length(max_pattern_length, db)``
    ``min(max_pattern_length, longest sequence, shortest sequence)``.

    The shortest-sequence bound is **sound**: a pattern longer than the
    shortest sequence cannot occur in that sequence, hence cannot reach any
    threshold above zero.  Applying it in one place (rather than in some miners
    and not others) is what keeps the oracle's output set equal to the miners'
    output set.  It is a real speedup for the oracle, not a licence to truncate
    differently per implementation.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import SequenceDatabase

__all__ = ["budget_for", "effective_max_length", "support_threshold"]

#: Tolerance absorbing float representation error in ``ceil(s * n)``.
_SUPPORT_EPS = 1e-9


def support_threshold(min_support: float, n_sequences: int) -> int:
    """Minimum sequence count clearing ``min_support``, clamped to ``[1, n]``."""
    if not 0 < min_support <= 1:
        raise ValueError(f"min_support must lie in (0, 1], got {min_support}")
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1, got {n_sequences}")
    cut = int(np.ceil(min_support * n_sequences - _SUPPORT_EPS))
    return max(1, min(cut, n_sequences))


def effective_max_length(max_pattern_length: int, db: SequenceDatabase) -> int:
    """Largest pattern length that can possibly occur in ``db``."""
    if max_pattern_length < 1:
        raise ValueError(f"max_pattern_length must be >= 1, got {max_pattern_length}")
    return min(max_pattern_length, db.max_length(), int(db.lengths.min()))


def budget_for(
    min_support: float,
    max_pattern_length: int,
    db: SequenceDatabase,
) -> tuple[int, int]:
    """``(min_count, max_len)`` -- the canonical mining budget for ``db``."""
    return support_threshold(min_support, db.n_sequences), effective_max_length(
        max_pattern_length, db
    )
