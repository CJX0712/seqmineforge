"""Core domain types for SeqMineForge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

__all__ = [
    "MiningResult",
    "Pattern",
    "PatternType",
    "SequenceDatabase",
    "SupportKind",
]


class PatternType(StrEnum):
    """Sub-pattern semantics used across the whole codebase.

    ``CONTIGUOUS``  -> substring patterns (a contiguous block of a sequence).
    ``SUBSEQUENCE`` -> classical *sequential* patterns (SPM literature sense):
        a pattern matches iff it is a **subsequence** of the sequence
        (order preserving, gaps allowed).

    The two semantics are deliberately distinguished at the type level: mixing
    them silently is the single most common correctness bug in sequential
    mining code (a contiguous matcher silently reported as a subsequence
    matcher).
    """

    CONTIGUOUS = "contiguous"
    SUBSEQUENCE = "subsequence"


class SupportKind(StrEnum):
    """How ``support`` is counted.

    ``RELATIVE`` -> ``n_sequences_containing / n_sequences`` in ``[0, 1]``.
    ``ABSOLUTE`` -> raw integer sequence count.

    Both are stored internally as a *relative* fraction to keep the algebra
    uniform; :attr:`MiningResult.support_abs` exposes the integer form.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


@dataclass(frozen=True)
class SequenceDatabase:
    """An immutable sequence database.

    Attributes
    ----------
    sequences:
        Tuple of integer item sequences.  Each sequence is stored as a
        ``np.int64`` array of item ids in ``[0, n_items)``.
    n_items:
        Size of the item universe ``[0, n_items)``.
    name:
        Human readable identifier of the database (used in reports).
    """

    sequences: tuple[np.ndarray, ...]
    n_items: int
    name: str = "db"

    def __post_init__(self) -> None:
        if len(self.sequences) == 0:
            raise ValueError("SequenceDatabase requires at least one sequence")
        if self.n_items <= 0:
            raise ValueError("n_items must be positive")

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.sequences], dtype=np.int64)

    @property
    def total_length(self) -> int:
        return int(self.lengths.sum())

    def max_length(self) -> int:
        return int(self.lengths.max()) if len(self.sequences) else 0

    def item_support(self) -> np.ndarray:
        """Number of sequences each item appears in (item-level support)."""
        counts = np.zeros(self.n_items, dtype=np.int64)
        for seq in self.sequences:
            for item in np.unique(seq):
                counts[int(item)] += 1
        return counts


@dataclass(frozen=True)
class Pattern:
    """A mined pattern.

    ``items`` is the item-id tuple in order.  Two patterns are considered equal
    iff their ``items`` tuples are equal (patterns are canonical tuples, not
    objects with identity -- hashing on the tuple is therefore safe).
    """

    items: tuple[int, ...]
    support: float
    pattern_type: PatternType = PatternType.SUBSEQUENCE

    def __post_init__(self) -> None:
        if len(self.items) == 0:
            raise ValueError("Pattern.items must be non-empty")
        if not np.isfinite(self.support):
            raise ValueError(f"Pattern.support must be finite, got {self.support!r}")
        if self.support < -1e-12 or self.support > 1.0 + 1e-12:
            raise ValueError(f"Pattern.support must lie in [0, 1], got {self.support}")

    @property
    def length(self) -> int:
        return len(self.items)

    def key(self) -> tuple[int, ...]:
        return self.items

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"<{''.join(str(i) for i in self.items)}:{self.support:.3f}>"


@dataclass
class MiningResult:
    """Result of a mining run.

    ``patterns`` is kept sorted by ``(-support, items)`` so that downstream
    consumers get a deterministic ordering without re-sorting.
    """

    patterns: tuple[Pattern, ...] = ()
    pattern_type: PatternType = PatternType.SUBSEQUENCE
    name: str = "miner"
    elapsed_sec: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patterns)

    @property
    def keys(self) -> frozenset[tuple[int, ...]]:
        return frozenset(p.items for p in self.patterns)

    @property
    def supports(self) -> dict[tuple[int, ...], float]:
        return {p.items: p.support for p in self.patterns}

    def sorted_patterns(self) -> tuple[Pattern, ...]:
        return tuple(sorted(self.patterns, key=lambda p: (-p.support, p.items)))

    def max_length(self) -> int:
        return max((p.length for p in self.patterns), default=0)

    def lattice_dominance(self) -> int:
        """Number of ``(pattern, super-pattern)`` pairs both present.

        Only meaningful for :attr:`PatternType.CONTIGUOUS` patterns where the
        sub-pattern relation is a genuine partial order.  Used as a cheap
        determinism / consistency invariant in tests.
        """
        keys = self.keys
        count = 0
        for pat in self.patterns:
            n = len(pat.items)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    sub = pat.items[i:j]
                    if len(sub) < n and sub in keys:
                        count += 1
        return count
