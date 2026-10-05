"""PrefixSpan -- Pei et al., ICDM 2001.

The paper's insight: a pattern's support can be computed on a *projected
database* (pDB) instead of the full database, because only sequences containing
the pattern can contribute to its extensions, and within a sequence only the
suffix starting at the match matters.

Recursion
---------
``dfs(prefix, pdb, depth)``:
  1. **Count projection** -- item counts inside the pDB give the candidates.
  2. Keep those whose support clears the threshold; the *count itself* is the
     support, so no database rescan is needed.
  3. Project the pDB under ``prefix + (x,)`` and recurse.

Compared with SPADE (vertical id-lists) and SPAM (projected + symbol pruning),
PrefixSpan is the projection-only baseline: it performs **no** symbol / DFS
pruning, which makes it the honest "projection but no pruning" reference point.
Its weakness -- many candidate extensions -- is exactly what the flagship's
vectorised counter is measured against.

Projected-database convention
-----------------------------
A pDB suffix begins with the matched pattern, so extension candidates sit at
positions ``>= depth``.  ``depth`` is threaded through explicitly; re-searching
the suffix for the prefix instead is the classic silent SPM bug -- see
:data:`seqmineforge.miners.matching.PROJECTED_DB_MATCHES_PREFIX`.

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

__all__ = ["PrefixSpanMiner"]

#: A projected suffix always has exactly **one** already-matched element at its
#: front (see :data:`seqmineforge.miners.matching.PROJECTED_DB_CONSUMES_ONE`).
#: Every projection call in this module therefore passes 1, never
#: len(prefix) -- passing the depth silently skips real items.
CONSUMED = 1


class PrefixSpanMiner:
    """``name = "prefixspan"``"""

    name = "prefixspan"

    def mine(
        self,
        db: SequenceDatabase,
        *,
        min_support: float,
        max_pattern_length: int,
        pattern_type: PatternType = PatternType.SUBSEQUENCE,
    ) -> MiningResult:
        if pattern_type is not PatternType.SUBSEQUENCE:
            raise ValueError("PrefixSpan implements the subsequence semantics only")
        if not 0 < min_support <= 1:
            raise ValueError(f"min_support must lie in (0, 1], got {min_support}")

        start = time.perf_counter()
        n_seq = db.n_sequences
        min_count, max_len = budget_for(min_support, max_pattern_length, db)

        stats = {"n_projects": 0.0, "n_visited": 0.0}
        found: list[Pattern] = []

        item_counts = db.item_support()
        level1 = sorted(
            int(i) for i in np.flatnonzero(item_counts) if int(item_counts[i]) >= min_count
        )
        for item in level1:
            found.append(Pattern((item,), item_counts[item] / n_seq, PatternType.SUBSEQUENCE))

        def dfs(prefix: tuple[int, ...], pdb: list[np.ndarray], depth: int) -> None:
            if depth >= max_len or not pdb:
                return
            counts = profile_items(CONSUMED, pdb)

            for it in sorted(counts):
                stats["n_visited"] += 1.0
                if counts[it] < min_count:
                    continue
                ext = (*prefix, it)
                found.append(Pattern(ext, counts[it] / n_seq, PatternType.SUBSEQUENCE))
                if len(ext) >= max_len:
                    continue
                stats["n_projects"] += 1.0
                sub_pdb: list[np.ndarray] = []
                for suffix in pdb:
                    # Only the NEW item is searched for -- the prefix has
                    # already been chopped off by the projection.  See
                    # matching.extend_projection for the full explanation.
                    projected = extend_projection(suffix, CONSUMED, it)
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

        # The pDB count *is* the support, by construction of the projection.
        # Re-verify against an independent scan anyway: this converts a silent
        # projection bug into a loud failure instead of a plausible-looking
        # result table.
        for pat in found:
            true_count = sum(
                1
                for seq in db.sequences
                if contains_subsequence(np.asarray(pat.items, dtype=np.int64), seq)
            )
            if abs(true_count / n_seq - pat.support) > 1e-12:
                raise AssertionError(
                    f"PrefixSpan internal inconsistency for {pat.items}: "
                    f"pDB said {pat.support}, truth {true_count / n_seq}"
                )

        patterns = tuple(sorted(found, key=lambda p: (-p.support, p.items)))
        stats["n_patterns"] = float(len(patterns))
        return MiningResult(
            patterns=patterns,
            pattern_type=PatternType.SUBSEQUENCE,
            name=self.name,
            elapsed_sec=time.perf_counter() - start,
            stats=stats,
        )
