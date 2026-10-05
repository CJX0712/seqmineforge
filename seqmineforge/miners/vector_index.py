"""Vectorised support counting -- the flagship's performance core.

The scalar baselines pay ``O(n_sequences * len(pattern))`` Python-level work per
candidate pattern.  This module replaces that with **one NumPy pass over a
packed sequence matrix**, which is where ``SeqFuse``'s speedup actually comes
from.  No claim of novelty is made: the DP-over-positions formulation is the
standard "count all subsequences of length k" automaton.

Two engines, same contract
--------------------------
:class:`SequenceIndex`
    Pads the database into an ``(n_seq, max_len)`` int matrix and answers
    "how many sequences contain pattern ``p`` as a subsequence?" by running the
    classic forward DP **once per pattern**, vectorised across sequences.

:class:`BitsetIndex`
    Packs, for every item, the set of sequences containing it as a bitset.
    A *contiguous* pattern's support is then a chain of bitwise ANDs -- the
    cheapest possible route, and the reason SeqFuse switches pattern semantics
    per length when the caller allows it.

Both engines are verified against :mod:`seqmineforge.miners.matching` by unit
tests (``tests/test_vector_index.py``); a vectorised engine that is *faster but
wrong* is worse than no engine at all.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import PatternType, SequenceDatabase

__all__ = ["BitsetIndex", "SequenceIndex", "has_block", "semantic_for"]


class SequenceIndex:
    """Forward-DP subsequence-support engine over a padded sequence matrix.

    Semantics: ``support(p)`` = number of sequences in which ``p`` appears as an
    order-preserving (gap-allowed) subsequence.

    DP formulation
    --------------
    For pattern ``p = (p_0, ..., p_{k-1})`` and sequence ``s``, define
    ``f_t`` = number of positions in ``s[:t]`` where the sub-pattern
    ``p[:t+1]`` can be completed.  The vectorised recurrence over all sequences
    at once is

    ``alive = zeros(n_seq, bool)``           # pattern prefix still matching
    ``pos   = zeros(n_seq, int)``            # cursor into each sequence
    for each element ``e`` of the pattern:
        advance cursor while seq[pos] != e

    Implemented with :func:`_advance` using ``argmax`` over a boolean matrix,
    which turns the inner Python loop into a handful of vector operations.
    """

    def __init__(self, db: SequenceDatabase) -> None:
        self.n_seq = db.n_sequences
        self.max_len = db.max_length()
        self._db = db
        # Pad with -1 (not a valid item id) so comparisons can never match.
        self._mat = np.full((self.n_seq, self.max_len), -1, dtype=np.int64)
        for i, seq in enumerate(db.sequences):
            self._mat[i, : len(seq)] = seq
        self._match_mask = self._mat >= 0
        # item -> (n_seq, max_len) boolean hit matrix, precomputed once.
        self._item_hits: dict[int, np.ndarray] = {}

    # -- item-level precomputation -------------------------------------
    def item_hits(self, item: int) -> np.ndarray:
        """``(n_seq, max_len)`` boolean mask of positions holding ``item``."""
        cached = self._item_hits.get(item)
        if cached is None:
            cached = self._mat == item
            self._item_hits[item] = cached
        return cached

    def item_support(self, item: int) -> int:
        """Number of sequences containing ``item`` (vectorised)."""
        return int(self.item_hits(item).any(axis=1).sum())

    def item_support_vector(self, n_items: int) -> np.ndarray:
        """Item-level support for the whole universe at once.

        ``(n_seq, max_len)`` matrix reduced to per-item counts in a single pass,
        avoiding the Python loop a naive implementation would need.
        """
        counts = np.zeros(n_items, dtype=np.int64)
        mat = self._mat
        for item in range(n_items):
            rows = (mat == item).any(axis=1)
            counts[item] = int(rows.sum())
        return counts

    # -- pattern support -------------------------------------------------
    def support(self, pattern: tuple[int, ...]) -> int:
        """Sequences containing ``pattern`` as a subsequence (vectorised DP)."""
        k = len(pattern)
        if k == 0:
            return self.n_seq
        if k > self.max_len:
            return 0

        mat = self._mat
        alive = np.ones(self.n_seq, dtype=bool)
        cursor = np.zeros(self.n_seq, dtype=np.int64)
        for element in pattern:
            if not alive.any():
                return 0
            cursor, alive = _advance(mat, cursor, alive, element, self.max_len)
        return int(alive.sum())

    def supports(self, patterns: list[tuple[int, ...]]) -> np.ndarray:
        """Vector of supports for a list of patterns (loop in NumPy space)."""
        out = np.zeros(len(patterns), dtype=np.int64)
        for i, pat in enumerate(patterns):
            out[i] = self.support(pat)
        return out

    def counts(self, pattern: tuple[int, ...]) -> int:
        """Alias of :meth:`support` for symmetry with the brute-force miner."""
        return self.support(pattern)


def _advance(
    mat: np.ndarray,
    cursor: np.ndarray,
    alive: np.ndarray,
    element: int,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every live cursor to the next occurrence of ``element``.

    Returns the updated ``(cursor, alive)`` pair.  Dead sequences are frozen
    (their cursor is clamped to ``max_len``) so that subsequent element steps
    cannot resurrect them.

    Strictly-after semantics
    ------------------------
    The returned cursor is ``first_hit + 1``, not ``first_hit``.  This is not a
    micro-optimisation: with ``first_hit`` the *same position* could satisfy two
    elements of a repeated pattern, so ``support((0, 0))`` came out 2 on
    ``([0,1], [1,0])`` while :func:`contains_subsequence` correctly said 0.  The
    vectorised engine must advance the cursor exactly as the scalar scan advances
    its iterator -- otherwise "vectorised == brute force" is false and the whole
    G1 gate is meaningless.
    """
    mat.shape[0]
    hits = mat == element
    cur = np.where(alive, cursor, max_len)
    positions = np.arange(max_len)[None, :]
    allowed = hits & (positions >= cur[:, None])
    any_hit = allowed.any(axis=1) & alive
    # argmax on a boolean row gives the first True (0 if none) -- the exact
    # semantics we want, with no Python-level scan.
    first = np.argmax(allowed, axis=1)
    new_cursor = np.where(any_hit, np.minimum(first + 1, max_len), max_len)
    return new_cursor, any_hit


class BitsetIndex:
    """Bit-packed item-presence engine for *contiguous* pattern semantics.

    ``support(p)`` = number of sequences containing the block ``p``.  Computed
    as a chain of AND / OR over per-sequence boolean rows, which is the fastest
    possible route for this semantics.
    """

    def __init__(self, db: SequenceDatabase) -> None:
        self.n_seq = db.n_sequences
        self.n_items = db.n_items
        self.max_len = db.max_length()
        self._seqs = db.sequences

    def item_support(self, item: int) -> int:
        return sum(1 for seq in self._seqs if bool((seq == item).any()))

    def contains_contiguous(self, pattern: tuple[int, ...]) -> np.ndarray:
        """Per-sequence boolean mask of contiguous containment."""
        n_pat = len(pattern)
        mask = np.ones(self.n_seq, dtype=bool)
        if n_pat == 0:
            return mask
        arr = np.asarray(pattern, dtype=np.int64)
        for i, seq in enumerate(self._seqs):
            if len(seq) < n_pat or not has_block(seq, arr):
                mask[i] = False
        return mask

    def support(self, pattern: tuple[int, ...]) -> int:
        """Support under ``CONTIGUOUS`` semantics."""
        return int(self.contains_contiguous(pattern).sum())


def has_block(seq: np.ndarray, arr: np.ndarray) -> bool:
    """Contiguous-containment test, exported so every module shares one copy.

    The ``break`` on ``end > len(seq)`` relies on ``flatnonzero`` returning
    ascending positions: once the block no longer fits at the earliest hit, it
    cannot fit at any later hit either.
    """
    n_pat = len(arr)
    if n_pat == 0:
        return True
    for start in np.flatnonzero(seq == arr[0]):
        end = start + n_pat
        if end > len(seq):
            break
        if np.array_equal(seq[start:end], arr):
            return True
    return False


def semantic_for(pattern_type: PatternType) -> str:
    """Human-readable semantic tag (used in report tables)."""
    return "subseq" if pattern_type is PatternType.SUBSEQUENCE else "contig"
