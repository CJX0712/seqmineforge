"""Miner registry + availability probing.

Author: 晨星
"""

from __future__ import annotations

from ..core.errors import UnknownMinerError
from ..core.types import MiningResult, PatternType, SequenceDatabase
from .flagship import FLAGSHIP, SeqFuse
from .oracle import BruteForceMiner, brute_force_support
from .prefixspan import PrefixSpanMiner
from .spade import SPADEiner
from .spam import SPAMiner

__all__ = [
    "BASELINES",
    "FLAGSHIP",
    "MINERS",
    "BruteForceMiner",
    "PrefixSpanMiner",
    "SPADEiner",
    "SPAMiner",
    "SeqFuse",
    "brute_force_support",
    "get_miner",
    "mine_all",
]

#: Baselines only (the flagship is excluded).  Used for "strongest baseline"
#: selection so the efficiency gate always measures against the best rival.
BASELINES: dict[str, type] = {
    SPAMiner.name: SPAMiner,
    SPADEiner.name: SPADEiner,
    PrefixSpanMiner.name: PrefixSpanMiner,
}

#: Everything the pipeline can run, including the exhaustive oracle.
MINERS: dict[str, type] = {
    **BASELINES,
    FLAGSHIP.name: FLAGSHIP,
    "oracle": BruteForceMiner,
}


def get_miner(name: str):
    """Instantiate a miner by name.

    Raises
    ------
    UnknownMinerError
        With the list of registered names -- a typo must never silently fall
        back to a default miner, because that would invalidate a result table.
    """
    key = name.lower()
    if key not in MINERS:
        raise UnknownMinerError(f"unknown miner {name!r}; known: {sorted(MINERS)}")
    return MINERS[key]()


def mine_all(
    names: list[str],
    db: SequenceDatabase,
    *,
    min_support: float,
    max_pattern_length: int,
    pattern_type: PatternType = PatternType.SUBSEQUENCE,
) -> dict[str, MiningResult]:
    """Run several miners over the same database with identical parameters."""
    out: dict[str, MiningResult] = {}
    for name in names:
        miner = get_miner(name)
        out[name] = miner.mine(
            db,
            min_support=min_support,
            max_pattern_length=max_pattern_length,
            pattern_type=pattern_type,
        )
    return out
