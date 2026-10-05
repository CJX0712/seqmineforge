"""SPADE -- Sequential PAttern DEtection (Agrawal & Srikant, VLDB 1995).

SPADE's contribution over Apriori is the **vertical representation**: instead of
re-scanning sequences, each pattern carries an *id-list* ``T_p`` -- the set of
sequence ids containing it -- so that support is an intersection:

    T_(p . x) = T_p ∩ T_x
    e_(p . x) = |T_(p . x)|

Two consequences give SPADE its pruning power:

* **Downward closure.**  If ``e_(p.x) < min_count`` then ``p.x`` is infrequent,
  and so is every extension of it -- the subtree dies immediately.
* **The id-list doubles as the projection.**  ``T_p`` *is* the set of sequences
  whose projected database is non-empty, so candidate generation never has to
  look at sequences that cannot contribute.

The id-list is a *candidate* filter: an item may occur in a sequence *before* the
pattern's last element, in which case the sequence lies in ``T_p ∩ T_x`` but does
not really contain ``p.x``.  Exact order-aware support therefore has to be
verified -- this implementation does so, and a unit test asserts that the
intersection count is an upper bound on the true count, which is precisely the
pruning invariant SPADE relies on.

Author: 晨星
"""

from __future__ import annotations

import time

import numpy as np

from ..core.types import MiningResult, Pattern, PatternType, SequenceDatabase
from .budget import budget_for
from .matching import contains_subsequence, greedy_match_end

__all__ = ["SPADEiner"]


class SPADEiner:
    """``name = "spade"``"""

    name = "spade"

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult:
        if pattern_type is not PatternType.SUBSEQUENCE:
            raise ValueError("SPADE implements the subsequence semantics only")
        if not 0 < min_support <= 1:
            raise ValueError(f"min_support must lie in (0, 1], got {min_support}")

        start = time.perf_counter()
        n_seq = db.n_sequences
        min_count, max_len = budget_for(min_support, max_pattern_length, db)

        # ---- vertical representation: T_x and first position of x --------
        item_ids: dict[int, set[int]] = {}
        item_pos: dict[int, np.ndarray] = {}
        for sid, seq in enumerate(db.sequences):
            for pos, item in enumerate(seq):
                it = int(item)
                item_ids.setdefault(it, set()).add(sid)
                arr = item_pos.get(it)
                if arr is None:
                    arr = np.full(n_seq, -1, dtype=np.int64)
                    item_pos[it] = arr
                arr[sid] = pos

        stats = {"n_intersections": 0.0, "n_pruned": 0.0, "n_visited": 0.0}
        found: list[Pattern] = []

        def exact_support(pattern: tuple[int, ...]) -> int:
            arr = np.asarray(pattern, dtype=np.int64)
            return sum(1 for seq in db.sequences if contains_subsequence(arr, seq))

        def recurse(
            pattern: tuple[int, ...],
            ids: set[int],
            ends: np.ndarray,
        ) -> None:
            """Extend ``pattern``; ``ends[sid]`` = end index of the ``pattern`` match."""
            if len(pattern) >= max_len:
                return
            cand: dict[int, int] = {}
            for sid in ids:
                base = int(ends[sid])
                if base < 0:
                    continue
                seq = db.sequences[sid]
                seen: set[int] = set()
                for pos in range(base + 1, len(seq)):
                    it = int(seq[pos])
                    if it in seen:
                        continue
                    seen.add(it)
                    cand[it] = cand.get(it, 0) + 1

            for it, cnt in sorted(cand.items()):
                if cnt < min_count:
                    stats["n_pruned"] += 1.0
                    continue
                ext = (*pattern, it)
                t_ext = ids & item_ids.get(it, set())
                stats["n_intersections"] += 1.0
                if len(t_ext) < min_count:
                    stats["n_pruned"] += 1.0
                    continue
                stats["n_visited"] += 1.0
                true_count = exact_support(ext)
                if true_count < min_count:
                    continue
                found.append(Pattern(ext, true_count / n_seq, PatternType.SUBSEQUENCE))

                arr = np.asarray(ext, dtype=np.int64)
                next_ends = np.full(n_seq, -1, dtype=np.int64)
                for sid in t_ext:
                    e = greedy_match_end(arr, db.sequences[sid])
                    if e >= 0:
                        next_ends[sid] = e - 1
                recurse(ext, t_ext, next_ends)

        level1 = sorted(
            it for it, ids in item_ids.items() if len(ids) >= min_count and it < db.n_items
        )
        for item in level1:
            found.append(Pattern((item,), len(item_ids[item]) / n_seq, PatternType.SUBSEQUENCE))
            if max_len < 2:
                continue
            recurse((item,), set(item_ids[item]), item_pos[item])

        patterns = tuple(sorted(found, key=lambda p: (-p.support, p.items)))
        stats["n_patterns"] = float(len(patterns))
        return MiningResult(
            patterns=patterns,
            pattern_type=PatternType.SUBSEQUENCE,
            name=self.name,
            elapsed_sec=time.perf_counter() - start,
            stats=stats,
        )
