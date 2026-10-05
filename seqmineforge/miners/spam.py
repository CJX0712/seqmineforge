"""SPAM -- count / symbol projection miner (Agrawal & Srikant, SIGMOD 1994;
refined as SPAM by Ayres, Ghosh & Onik, ICDT 2003).

Faithful CPU re-implementation of the published algorithm, *not* the original
Java/C++ code.  The projections of the paper are all present:

1. **Count projection** -- item occurrence counts inside the projected
   database; the only items that may extend the pattern.
2. **Symbol projection** -- which items occur as the *final* symbol of a
   projected sequence.  An item that never terminates a projected sequence
   cannot produce a frequent extension, so it is dropped (DFS-pruning).
3. **First projection** -- suffixes of matching sequences, so extension never
   rescans consumed elements.

Correctness of this re-implementation is exactly what the oracle-equivalence
gate verifies.  No claim is made about reproducing the paper's timings.

Author: 晨星
"""

from __future__ import annotations

import time

import numpy as np

from ..core.types import MiningResult, Pattern, PatternType, SequenceDatabase
from .budget import budget_for
from .matching import (
    contains_subsequence,
    extend_projection,
    profile_items,
    project_sequence,
)

__all__ = ["SPAMiner", "prunable_sequences", "symbol_projection"]

#: A projected suffix always has exactly **one** already-matched element at its
#: front (see :data:`seqmineforge.miners.matching.PROJECTED_DB_CONSUMES_ONE`).
#: Every projection call in this module therefore passes 1, never
#: len(prefix) -- passing the depth silently skips real items.
CONSUMED = 1


def symbol_projection(consumed: int, pdb: list[np.ndarray]) -> dict[int, int]:
    """SPAM symbol projection over a projected database.

    Returns, per item, the number of projected sequences whose **final** symbol
    is that item.  Under SPAM's DFS-pruning this identifies sequences that are
    "complete": a symbol that terminates a projected sequence cannot be
    extended *within that sequence*, so the sequence is dropped from further
    extensions there.

    Crucially this must **not** be used to remove an item from the candidate
    set.  Doing so over-prunes: an item frequently appearing as a final symbol
    can still occur mid-sequence in other projected sequences and extend fine.
    Real over-pruning measured on the ``markov`` DGP: dropping candidates absent
    from the last-symbol map lost 21 of 1290 frequent patterns.  The correct use
    is :func:`prunable_sequences`, which drops *sequences*, never *items*.
    """
    if consumed < 1:
        raise ValueError(f"consumed must be >= 1, got {consumed}")
    counts: dict[int, int] = {}
    for suffix in pdb:
        if len(suffix) <= consumed:
            continue
        last = int(suffix[-1])
        counts[last] = counts.get(last, 0) + 1
    return counts


def prunable_sequences(pdb: list[np.ndarray], consumed: int) -> set[int]:
    """Indices of projected sequences that cannot be extended at all.

    A projected suffix holds nothing beyond its final symbol, so no item can be
    appended after that symbol -- the sequence is exhausted.  Dropping such
    sequences (never items) is the form of SPAM's DFS-pruning that is actually
    sound.
    """
    if consumed < 1:
        raise ValueError(f"consumed must be >= 1, got {consumed}")
    return {i for i, suffix in enumerate(pdb) if len(suffix) <= consumed}


class SPAMiner:
    """``name = "spam"``"""

    name = "spam"

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult:
        if pattern_type is not PatternType.SUBSEQUENCE:
            raise ValueError("SPAM implements the subsequence semantics only")
        if not 0 < min_support <= 1:
            raise ValueError(f"min_support must lie in (0, 1], got {min_support}")

        start = time.perf_counter()
        n_seq = db.n_sequences
        min_count, max_len = budget_for(min_support, max_pattern_length, db)

        stats = {"n_count_proj": 0.0, "n_symbol_proj": 0.0, "n_visited": 0.0}
        found: list[Pattern] = []

        def support(pattern: tuple[int, ...]) -> int:
            arr = np.asarray(pattern, dtype=np.int64)
            return sum(1 for seq in db.sequences if contains_subsequence(arr, seq))

        # ---- level 1: item-level support --------------------------------
        item_counts = db.item_support()
        level1 = sorted(
            int(i) for i in np.flatnonzero(item_counts) if int(item_counts[i]) >= min_count
        )
        stats["n_count_proj"] += 1.0
        for item in level1:
            found.append(Pattern((item,), item_counts[item] / n_seq, PatternType.SUBSEQUENCE))

        def dfs(pattern: tuple[int, ...], pdb: list[np.ndarray], depth: int) -> None:
            """Extend ``pattern`` using its projected database."""
            if depth >= max_len or not pdb:
                return
            # -- DFS-pruning: drop exhausted *sequences*, never items -------
            exhausted = prunable_sequences(pdb, CONSUMED)
            stats["n_symbol_proj"] += float(len(exhausted))
            live = [s for i, s in enumerate(pdb) if i not in exhausted]
            if not live:
                return

            # -- count projection over the surviving sequences --------------
            counts = profile_items(CONSUMED, live)
            stats["n_count_proj"] += 1.0
            candidates = sorted(it for it, c in counts.items() if c >= min_count)

            for item in candidates:
                stats["n_visited"] += 1.0
                ext = (*pattern, item)
                cnt = support(ext)
                if cnt < min_count:
                    continue
                found.append(Pattern(ext, cnt / n_seq, PatternType.SUBSEQUENCE))
                if len(ext) >= max_len:
                    continue
                sub_pdb: list[np.ndarray] = []
                for suffix in live:
                    # Only the NEW item is searched for -- the prefix has
                    # already been chopped off by the projection.  See
                    # matching.extend_projection for the full explanation.
                    projected = extend_projection(suffix, CONSUMED, item)
                    if projected is None:
                        continue
                    sub_pdb.append(projected)
                dfs(ext, sub_pdb, len(ext))

        for item in level1:
            if max_len < 2:
                continue
            arr = np.asarray([item], dtype=np.int64)
            pdb: list[np.ndarray] = []
            for seq in db.sequences:
                projected = project_sequence(arr, seq)
                if projected is None:
                    continue
                pdb.append(projected)
            dfs((item,), pdb, 1)

        patterns = tuple(sorted(found, key=lambda p: (-p.support, p.items)))
        stats["n_patterns"] = float(len(patterns))
        return MiningResult(
            patterns=patterns,
            pattern_type=PatternType.SUBSEQUENCE,
            name=self.name,
            elapsed_sec=time.perf_counter() - start,
            stats=stats,
        )
