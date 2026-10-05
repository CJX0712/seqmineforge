"""Shared pattern-matching primitives -- the correctness root of the package.

Every miner (baselines *and* the flagship) funnels its support counting and its
projection through the functions here.  That is deliberate: it makes "the oracle
and the flagship agree" a statement about the *search strategies*, not about two
independently-implemented matchers that might both be subtly wrong.

Three semantics are implemented and kept strictly separate:

``contains_subsequence(p, s)``
    Classical SPM semantics: order-preserving, gaps allowed.
``contains_contiguous(p, s)``
    ``p`` appears as a contiguous block of ``s``.
``project_sequence(p, s)``
    PrefixSpan-style projection: the suffix of ``s`` starting at the **last**
    matched element of ``p``.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PROJECTED_DB_CONSUMES_ONE",
    "apriori_prune",
    "contains_contiguous",
    "contains_subsequence",
    "extend_projection",
    "greedy_match_end",
    "profile_items",
    "project",
    "project_sequence",
    "right_extensions",
]

#: **Projected-database convention** -- the single most bug-prone detail in SPM.
#:
#: A projected database for pattern ``p`` is a list of suffixes, each **starting
#: at the position of ``p``'s last matched element** inside its original
#: sequence.  So for ``p = (a, b)`` and sequence ``s`` whose match of ``p``
#: starts at index ``t``, the projected suffix is ``s[t+1:]`` and therefore
#: ``suffix[0] == b``.
#:
#: Consequently the invariant is **one element consumed, not len(p)**: a
#: projected suffix of *any* depth has exactly one already-matched element at
#: position 0, and extension candidates therefore live at positions ``>= 1``.
#: :func:`profile_items` and :func:`extend_projection` both take ``depth=1`` for
#: this reason.  Passing ``depth=len(p)`` skips real elements and silently drops
#: patterns; re-searching the suffix for ``p`` fails outright because the earlier
#: elements are gone.
#:
#: Worked example (``s = [13, 10, 0, 5, 2, 6, 12, 4, 8, 9]``):
#:   project ``(0,)``    -> ``[0, 5, 2, 6, 12, 4, 8, 9]``   candidates at pos >= 1
#:   extend with ``2``   -> ``[2, 6, 12, 4, 8, 9]``          candidates at pos >= 1
#:   extend with ``6``   -> ``[6, 12, 4, 8, 9]``
PROJECTED_DB_CONSUMES_ONE = True


def contains_subsequence(pattern: np.ndarray, sequence: np.ndarray) -> bool:
    """True iff ``pattern`` is a (gap-allowed, order-preserving) sub-sequence.

    Greedy two-pointer scan.  This is the textbook algorithm and is
    ``O(len(sequence))``.  A greedy match is optimal for subsequence testing:
    matching each pattern element as early as possible leaves the largest
    remaining suffix for the rest, so a valid match is never missed.
    """
    if len(pattern) == 0:
        return True
    if len(pattern) > len(sequence):
        return False
    i = 0
    n_pat = len(pattern)
    for value in sequence:
        if value == pattern[i]:
            i += 1
            if i == n_pat:
                return True
    return False


def contains_contiguous(pattern: np.ndarray, sequence: np.ndarray) -> bool:
    """True iff ``pattern`` occurs as a contiguous block of ``sequence``."""
    n_pat = len(pattern)
    if n_pat == 0:
        return True
    if n_pat > len(sequence):
        return False
    for start in range(len(sequence) - n_pat + 1):
        if np.array_equal(sequence[start : start + n_pat], pattern):
            return True
    return False


def greedy_match_end(pattern: np.ndarray, sequence: np.ndarray) -> int:
    """End position of the earliest greedy subsequence match, or ``-1``.

    "Earliest greedy match" == match ``pattern[0]`` at the earliest occurrence,
    then ``pattern[1]`` at the earliest occurrence after it, and so on.  This
    is the standard choice behind PrefixSpan projection.
    """
    n_pat = len(pattern)
    if n_pat == 0:
        return 0
    i = 0
    for pos in range(len(sequence)):
        if sequence[pos] == pattern[i]:
            i += 1
            if i == n_pat:
                return pos + 1
    return -1


def project(pattern: np.ndarray, sequences: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    """Project ``sequences`` under ``pattern``, dropping non-matching entries.

    Thin list-level convenience wrapper over :func:`project_sequence`.  Each
    returned suffix satisfies ``suffix[:len(pattern)] == pattern``.
    """
    out: list[np.ndarray] = []
    for seq in sequences:
        projected = project_sequence(pattern, seq)
        if projected is not None:
            out.append(projected)
    return out


def profile_items(consumed: int, pdb: list[np.ndarray]) -> dict[int, int]:
    """Count projection over a projected database.

    ``consumed`` is the number of *already-matched* elements sitting at the front
    of every suffix.  Under the convention documented at module level that number
    is always ``1`` (:data:`PROJECTED_DB_CONSUMES_ONE`): the suffix begins at the
    last matched element and everything before it has been chopped away.
    Passing ``len(pattern)`` instead is a silent bug that skips real items.

    Each candidate item is counted **once per suffix**, which is what makes the
    count equal to "number of projected sequences in which this item can extend
    the pattern".
    """
    if consumed < 1:
        raise ValueError(f"consumed must be >= 1, got {consumed}")
    counts: dict[int, int] = {}
    for suffix in pdb:
        if len(suffix) <= consumed:
            continue
        seen: set[int] = set()
        for pos in range(consumed, len(suffix)):
            it = int(suffix[pos])
            if it in seen:
                continue
            seen.add(it)
            counts[it] = counts.get(it, 0) + 1
    return counts


def project_sequence(prefix: np.ndarray, seq: np.ndarray) -> np.ndarray | None:
    """Project a single sequence under ``prefix``; ``None`` if absent.

    The returned suffix starts **at** the last matched element, so
    ``suffix[0] == prefix[-1]`` and the consumed count is exactly ``1`` (see
    :data:`PROJECTED_DB_CONSUMES_ONE`).
    """
    end = greedy_match_end(prefix, seq)
    if end < 0:
        return None
    return seq[end - 1 :]


def extend_projection(suffix: np.ndarray, consumed: int, item: int) -> np.ndarray | None:
    """Extend one projected suffix by appending ``item``.

    This is the **only** correct way to build the projected database of
    ``prefix + (item,)`` from the projected database of ``prefix``.

    Why not ``project_sequence(prefix + (item,), suffix)``?
    --------------------------------------------------------
    A projected suffix has already had the earlier elements of ``prefix``
    chopped off.  Searching it for the *whole* extended pattern therefore fails
    even though the extended pattern genuinely occurs in the original sequence.
    Concrete failure on ``s = [0, 2, 6]`` with ``prefix = (0, 2)``: the projected
    suffix is ``[2, 6]``, so ``project_sequence((0,2,6), [2,6])`` returns
    ``None`` -- the ``0`` is gone -- and ``(0,2,6)`` is dropped although its
    support is non-zero.

    Why not search the whole suffix for ``item``?
    ---------------------------------------------
    ``suffix[0]`` is the previously matched element.  Searching from index 0
    would let ``item`` re-match that same position, producing a projected suffix
    that is too long and whose subsequent counts drift upward.

    Parameters
    ----------
    suffix:
        A projected suffix whose first ``consumed`` elements are already matched.
    consumed:
        Number of matched elements at the front of ``suffix`` -- ``1`` under the
        convention above.
    item:
        The item being appended.

    Returns
    -------
    numpy.ndarray or None
        The new projected suffix starting at the match of ``item``, or ``None``
        if ``item`` does not occur at any position ``>= consumed``.
    """
    if consumed < 1:
        raise ValueError(f"consumed must be >= 1, got {consumed}")
    if consumed > len(suffix):
        return None
    hits = np.flatnonzero(suffix[consumed:] == item)
    if hits.size == 0:
        return None
    pos = consumed + int(hits[0])
    return suffix[pos:]


def apriori_prune(sequences: tuple[np.ndarray, ...], pattern: tuple[int, ...]) -> bool:
    """Downward-closure screen (apriori) for *subsequence* patterns.

    If ``pattern`` occurs in some sequence, so does every one of its
    sub-patterns in that same sequence.  Hence: if an immediate prefix or
    suffix is absent everywhere, ``pattern`` cannot occur either.

    Used only as a cheap *sound* prune.  It never accepts a pattern -- that is
    exclusively the job of the support test.
    """
    if len(pattern) <= 1:
        return True
    for sub in (pattern[:-1], pattern[1:]):
        arr = np.asarray(sub, dtype=np.int64)
        if not any(contains_subsequence(arr, seq) for seq in sequences):
            return False
    return True


def right_extensions(prefix: tuple[int, ...], item_pool: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Right-extend ``prefix`` with each item in ``item_pool``.

    ``item_pool`` is normally the pre-filtered set of items whose item-level
    support already clears the threshold (apriori on items).  No duplicates can
    be produced because the database sequences are item-distinct.
    """
    return [(*prefix, item) for item in item_pool]
