"""Pattern-derived sequence features for downstream classification.

Two extractors, deliberately sharing one convention so the comparison between
them is apples-to-apples:

``BinaryPatternFeatures``
    Column ``j`` = 1 iff pattern ``j`` occurs as a subsequence of the sequence.
``SupportPatternFeatures``
    Column ``j`` = support of pattern ``j`` (a constant per column, so it
    encodes *which patterns are frequent* rather than which occur -- useful as a
    contrast, not as the main model).

The convention is fixed as **subsequence containment**; a contiguous matcher
would quietly report a different task.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..miners.matching import contains_subsequence

__all__ = ["BinaryPatternFeatures", "SupportPatternFeatures"]


class BinaryPatternFeatures:
    """Binary occurrence matrix, shape ``(n_sequences, n_patterns)``."""

    name = "binary"

    def __init__(self, patterns: list[tuple[int, ...]]) -> None:
        self.patterns = list(patterns)
        self._pats = [np.asarray(p, dtype=np.int64) for p in self.patterns]

    def transform(self, sequences: tuple[np.ndarray, ...]) -> np.ndarray:
        if not self.patterns:
            return np.zeros((len(sequences), 0), dtype=np.float64)
        out = np.zeros((len(sequences), len(self.patterns)), dtype=np.float64)
        for i, seq in enumerate(sequences):
            for j, pat in enumerate(self._pats):
                if contains_subsequence(pat, seq):
                    out[i, j] = 1.0
        return out


class SupportPatternFeatures:
    """Broadcast each pattern's support as a constant column."""

    name = "support"

    def __init__(
        self, patterns: list[tuple[int, ...]], supports: dict[tuple[int, ...], float]
    ) -> None:
        self.patterns = list(patterns)
        self._sup = np.array([supports[p] for p in self.patterns], dtype=np.float64)

    def transform(self, sequences: tuple[np.ndarray, ...]) -> np.ndarray:
        if not self.patterns:
            return np.zeros((len(sequences), 0), dtype=np.float64)
        return np.tile(self._sup, (len(sequences), 1))
