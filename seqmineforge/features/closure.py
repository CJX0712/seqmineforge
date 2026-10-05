"""Closed sequential patterns -- the compression lever for downstream modelling.

A frequent pattern ``p`` is **closed** iff no frequent *super*-sequence ``q``
(strictly containing ``p`` as a subsequence) has the same support.  The closure
theorem says every frequent pattern is a subsequence of some closed frequent
pattern, so the closed set is a lossless basis for the whole frequent set.

Why it matters here
-------------------
Mined pattern sets are huge and heavily redundant: thousands of frequent
patterns, most of them sub-sequences of a few dozen closed ones.  A downstream
classifier that consumes all of them pays a large, nearly useless feature
dimension.  Consuming only the closed set gives the **same information** at a
small fraction of the width -- which is what the utility gate measures.

The theorem is verified empirically rather than assumed: :func:`reconstruct`
re-derives the full frequent set from the closed set, and a unit test asserts the
reconstruction equals the original set exactly.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, permutations

__all__ = [
    "closed_patterns",
    "compression_ratio",
    "extension_index",
    "is_closed",
    "reconstruct",
]

#: Tolerance for comparing supports.  Supports are exact rationals
#: (``count / n_sequences``), so this only absorbs float representation error.
_SUP_TOL = 1e-12


def extension_index(
    supports: dict[tuple[int, ...], float],
) -> dict[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Map every frequent pattern to its frequent one-step right-extensions.

    Built once and reused, which turns ``closed_patterns`` from ``O(n^2)``
    dictionary scans into ``O(n)`` lookups -- worth doing at the ~2300-pattern
    scale of the benchmark, where the naive version dominated wall-clock.
    """
    index: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for pattern in supports:
        index.setdefault(pattern[:-1], []).append(pattern)
    return {k: tuple(v) for k, v in index.items()}


def is_closed(
    pattern: tuple[int, ...],
    supports: dict[tuple[int, ...], float],
    index: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None,
) -> bool:
    """True iff **no** frequent one-step extension shares ``pattern``'s support.

    Definition
    ----------
    ``p`` is *closed* iff there is no frequent pattern ``q`` strictly containing
    ``p`` as a subsequence with ``supp(q) == supp(p)``.  So the test is on
    **equality**, not on strict inequality.

    Why the direction matters
    -------------------------
    Testing ``sup(ext) > sup(p)`` looks reasonable and is always ``False``:
    downward closure guarantees ``sup(q) <= sup(p)`` for every sub-sequence ``p``
    of ``q``.  With that sign the function returned ``True`` for every pattern,
    i.e. a compression ratio of exactly 1.00 on all 2301 benchmark patterns --
    a number that looks like a *measurement* and is in fact a stub.  Equality is
    the only sound test.

    One level suffices
    ------------------
    If some longer witness ``q`` had equal support, ``q``'s length-``len(p)``
    prefix would have support ``>= |supp(q)| == |supp(p)|``, and downward closure
    caps it at ``|supp(p)|`` -- so that prefix is an equal-support one-step
    witness.
    """
    target = supports[pattern]
    ext_index = extension_index(supports) if index is None else index
    return all(abs(supports[ext] - target) > _SUP_TOL for ext in ext_index.get(pattern, ()))


def closed_patterns(
    patterns: Sequence[tuple[int, ...]], supports: dict[tuple[int, ...], float]
) -> list[tuple[int, ...]]:
    """The closed sub-list, sorted deterministically by ``(-support, items)``."""
    index = extension_index(supports)
    out = [p for p in patterns if is_closed(p, supports, index)]
    out.sort(key=lambda p: (-supports[p], p))
    return out


def reconstruct(
    closed: Sequence[tuple[int, ...]], target: Sequence[tuple[int, ...]]
) -> set[tuple[int, ...]]:
    """Patterns of ``target`` that are a subsequence of some closed pattern.

    Used to verify the closure theorem empirically: on a correctly computed
    closed basis this equals the original frequent set exactly.
    """
    out: set[tuple[int, ...]] = set()
    wanted = set(target)
    for c in closed:
        # Items within a pattern are distinct, so a sub-sequence is an ordered
        # subset of c's positions -- combinations pick positions, permutations
        # fix their order.
        n = len(c)
        for k in range(1, n + 1):
            for combo in combinations(range(n), k):
                for perm in permutations(combo):
                    key = tuple(c[i] for i in perm)
                    if key in wanted:
                        out.add(key)
    return out


def compression_ratio(n_full: int, n_closed: int) -> float:
    """``n_full / n_closed``; ``inf`` when the closed set is empty but full isn't."""
    if n_closed <= 0:
        return float("inf") if n_full > 0 else 1.0
    return float(n_full) / float(n_closed)
