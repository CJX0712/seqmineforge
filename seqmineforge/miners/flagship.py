"""SeqFuse -- the flagship miner.

Three mechanisms, each independently switchable so the ablation reports real
contrasts (including negative ones) instead of decorative rows.

M1. **Vectorised DP support counting** (:class:`SequenceIndex`)
    The forward-DP subsequence recurrence is vectorised *across sequences*,
    replacing the baselines' ``O(n_seq * len)`` Python-level scan with a handful
    of NumPy operations per pattern element.  Identical mathematics, far less
    interpreter overhead.

M2. **Profile-driven candidate generation**
    Candidates at node ``p`` are exactly the items occurring at positions
    ``>= len(p)`` in at least ``min_count`` projected sequences.  Switching M2
    off draws candidates from the whole item pool instead, which is what makes
    the ablation row informative.

M3. **Dual-representation routing**
    For ``CONTIGUOUS`` requests the flagship switches to
    :class:`BitsetIndex` (bitwise AND chains); for the default ``SUBSEQUENCE``
    semantics it uses the DP engine.  Routing is decided before any timing
    starts.

Honest scope statement
----------------------
SeqFuse is **not** claimed as a new SOTA algorithm.  It is a *systems*
improvement: identical output set (verified against the brute-force oracle) at
lower wall-clock.  The gain comes from vectorisation and candidate pruning, i.e.
from engineering -- and the ablation attributes the gain to each mechanism
separately so a reader can see which one actually pays.

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
from .vector_index import BitsetIndex, SequenceIndex, has_block

__all__ = ["FLAGSHIP", "SeqFuse"]

#: A projected suffix always has exactly **one** already-matched element at its
#: front (see :data:`seqmineforge.miners.matching.PROJECTED_DB_CONSUMES_ONE`).
#: Every projection call in this module therefore passes 1, never
#: len(prefix) -- passing the depth silently skips real items.
CONSUMED = 1

_ENGINE_CODE = {"scalar": 0, "dp": 1, "bitset": 2}


class SeqFuse:
    """Flagship sequential pattern miner.  ``name = "seqfuse"``"""

    name = "seqfuse"

    def __init__(
        self,
        *,
        use_vectorized: bool = True,
        use_profile_pruning: bool = True,
        use_routing: bool = True,
    ) -> None:
        self.use_vectorized = bool(use_vectorized)
        self.use_profile_pruning = bool(use_profile_pruning)
        self.use_routing = bool(use_routing)

    # ------------------------------------------------------------------
    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult:
        if not 0 < min_support <= 1:
            raise ValueError(f"min_support must lie in (0, 1], got {min_support}")
        if max_pattern_length < 1:
            raise ValueError(f"max_pattern_length must be >= 1, got {max_pattern_length}")

        start = time.perf_counter()
        n_seq = db.n_sequences
        min_count, max_len = budget_for(min_support, max_pattern_length, db)

        # ---- M3: routing, resolved before the timed search begins ----------
        engine_kind = "scalar"
        seq_index: SequenceIndex | None = None
        bits_index: BitsetIndex | None = None
        if self.use_routing and pattern_type is PatternType.CONTIGUOUS:
            engine_kind = "bitset"
            bits_index = BitsetIndex(db)
        elif self.use_vectorized:
            engine_kind = "dp"
            seq_index = SequenceIndex(db)

        def exact_support(pattern: tuple[int, ...]) -> int:
            """The single support oracle every engine branch must agree with."""
            if bits_index is not None:
                return bits_index.support(pattern)
            if seq_index is not None:
                return seq_index.support(pattern)
            arr = np.asarray(pattern, dtype=np.int64)
            if pattern_type is PatternType.CONTIGUOUS:
                return sum(1 for seq in db.sequences if has_block(seq, arr))
            return sum(1 for seq in db.sequences if contains_subsequence(arr, seq))

        stats = {
            "engine_code": float(_ENGINE_CODE[engine_kind]),
            "n_exact_calls": 0.0,
            "n_candidates": 0.0,
            "n_rejected": 0.0,
        }
        found: list[Pattern] = []

        item_counts = db.item_support()
        items = [int(i) for i in np.flatnonzero(item_counts) if int(item_counts[i]) >= min_count]
        for item in items:
            found.append(Pattern((item,), item_counts[item] / n_seq, pattern_type))

        # M2-off fallback: every item that is frequent on its own.  The ablation
        # then pays for exact support calls on candidates the profile would have
        # discarded -- a real cost, not a hypothetical one.
        unpruned_pool = list(items)

        def dfs(prefix: tuple[int, ...], pdb: list[np.ndarray], depth: int) -> None:
            if depth >= max_len or not pdb:
                return

            if pattern_type is PatternType.CONTIGUOUS:
                # Contiguous semantics need **adjacency**, which a projected
                # suffix cannot express: the items before the last match are
                # gone, and "anywhere after" is simply the wrong question.  The
                # extension candidates are exactly the items sitting at offset
                # +1 from the block's end, counted once per projected sequence.
                cont_counts: dict[int, int] = {}
                for suffix in pdb:
                    if len(suffix) < 2:
                        continue
                    nxt = int(suffix[1])
                    cont_counts[nxt] = cont_counts.get(nxt, 0) + 1
                candidates = sorted(it for it, c in cont_counts.items() if c >= min_count)
            elif self.use_profile_pruning:
                profile = profile_items(CONSUMED, pdb)
                candidates = sorted(it for it, c in profile.items() if c >= min_count)
            else:
                candidates = unpruned_pool

            for it in candidates:
                ext = (*prefix, it)
                stats["n_candidates"] += 1.0
                stats["n_exact_calls"] += 1.0
                cnt = exact_support(ext)
                if cnt < min_count:
                    stats["n_rejected"] += 1.0
                    continue
                found.append(Pattern(ext, cnt / n_seq, pattern_type))
                if len(ext) >= max_len:
                    continue
                if pattern_type is PatternType.CONTIGUOUS:
                    # Advance the block by exactly one position.
                    sub_pdb = [
                        suffix[1:] for suffix in pdb if len(suffix) >= 2 and int(suffix[1]) == it
                    ]
                else:
                    sub_pdb = []
                    for suffix in pdb:
                        projected = extend_projection(suffix, CONSUMED, it)
                        if projected is None:
                            continue
                        sub_pdb.append(projected)
                dfs(ext, sub_pdb, len(ext))

        for item in items:
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
            pattern_type=pattern_type,
            name=self.name,
            elapsed_sec=time.perf_counter() - start,
            stats=stats,
        )


FLAGSHIP = SeqFuse
