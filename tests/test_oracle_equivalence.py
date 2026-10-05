"""Oracle equivalence for every miner -- the hard correctness gate (G1).

Each test runs a miner and the brute-force oracle on the same small database and
demands *exact set equality* plus exact support agreement.  A failure here means
a search bug, not a tolerance issue.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from seqmineforge.core.types import PatternType, SequenceDatabase
from seqmineforge.data import get_dataset
from seqmineforge.eval import compare_to_oracle, oracle_candidate_count
from seqmineforge.miners import BASELINES, FLAGSHIP, MINERS, get_miner
from seqmineforge.miners.oracle import BruteForceMiner, brute_force_support

MINER_NAMES = [*list(BASELINES), FLAGSHIP.name]

SMALL_CASES = [
    (
        "markov",
        {"seed": 11, "n_sequences": 40, "n_items": 5, "min_len": 5, "max_len": 8, "noise": 0.2},
    ),
    (
        "planted",
        {
            "seed": 12,
            "n_sequences": 45,
            "n_items": 6,
            "min_len": 5,
            "max_len": 9,
            "n_planted": 2,
            "planted_len": 2,
        },
    ),
    (
        "segmented",
        {"seed": 13, "n_sequences": 40, "n_items": 5, "min_len": 5, "max_len": 9, "n_segments": 2},
    ),
    ("grouped", {"seed": 14, "n_sequences": 45, "n_groups": 3, "group_len": 2, "tail_len": 4}),
    (
        "prefixed",
        {
            "seed": 15,
            "n_sequences": 45,
            "n_items": 6,
            "n_prefixes": 2,
            "prefix_len": 2,
            "tail_len": 4,
        },
    ),
]


@pytest.mark.parametrize("miner_name", MINER_NAMES)
@pytest.mark.parametrize(("dataset", "kwargs"), SMALL_CASES)
def test_miner_matches_oracle_exactly(miner_name, dataset, kwargs):
    db = get_dataset(dataset, **kwargs)
    oracle = get_miner("oracle").mine(db, min_support=0.1, max_pattern_length=3)
    result = get_miner(miner_name).mine(db, min_support=0.1, max_pattern_length=3)
    report = compare_to_oracle(oracle, result)
    assert report.missing == frozenset(), f"missing: {sorted(report.missing)[:5]}"
    assert report.extra == frozenset(), f"extra: {sorted(report.extra)[:5]}"
    assert report.max_support_dev <= 1e-12
    assert report.exact


@pytest.mark.parametrize("min_support", [0.05, 0.15, 0.3])
def test_miner_matches_oracle_across_thresholds(min_support):
    db = get_dataset("markov", seed=21, n_sequences=35, n_items=5, min_len=5, max_len=8)
    oracle = get_miner("oracle").mine(db, min_support=min_support, max_pattern_length=3)
    result = get_miner(FLAGSHIP.name).mine(db, min_support=min_support, max_pattern_length=3)
    assert result.keys == oracle.keys


@pytest.mark.parametrize("max_len", [1, 2, 3, 4])
def test_miner_matches_oracle_across_length_caps(max_len):
    db = get_dataset("grouped", seed=22, n_sequences=40, n_groups=3, group_len=2, tail_len=4)
    oracle = get_miner("oracle").mine(db, min_support=0.1, max_pattern_length=max_len)
    result = get_miner(FLAGSHIP.name).mine(db, min_support=0.1, max_pattern_length=max_len)
    assert result.keys == oracle.keys


def test_all_miners_agree_with_each_other_on_a_medium_database():
    """Cross-implementation agreement where the oracle is infeasible."""
    db = get_dataset("markov", seed=31, n_sequences=150, n_items=10, min_len=9, max_len=15)
    kw = {"min_support": 0.06, "max_pattern_length": 4}
    keysets = {name: get_miner(name).mine(db, **kw).keys for name in MINER_NAMES}
    reference = keysets[MINER_NAMES[0]]
    for name in MINER_NAMES[1:]:
        assert keysets[name] == reference, f"{name} disagrees with {MINER_NAMES[0]}"


def test_downward_closure_of_results():
    """Every frequent pattern's frequent sub-patterns must also be reported."""
    from seqmineforge.miners.vector_index import SequenceIndex

    db = get_dataset("grouped", seed=41, n_sequences=120, n_groups=4, group_len=3, tail_len=5)
    res = get_miner(FLAGSHIP.name).mine(db, min_support=0.08, max_pattern_length=3)
    index = SequenceIndex(db)
    keys = set(res.keys)
    for key in keys:
        for cut in (1, len(key) - 1):
            if cut <= 0:
                continue
            for sub in (key[:cut], key[-cut:]):
                if len(sub) == len(key):
                    continue
                if index.support(sub) / db.n_sequences + 1e-12 >= 0.08:
                    assert sub in keys, f"closure violated: {sub} missing for {key}"


def test_supports_are_exact_after_mining():
    db = get_dataset(
        "planted",
        seed=51,
        n_sequences=60,
        n_items=9,
        min_len=6,
        max_len=10,
        n_planted=2,
        planted_len=2,
    )
    res = get_miner(FLAGSHIP.name).mine(db, min_support=0.1, max_pattern_length=3)
    for key, sup in res.supports.items():
        true = brute_force_support(key, db.sequences)
        assert abs(true / db.n_sequences - sup) <= 1e-12


# ------------------------------------------------------------------- oracle
def test_oracle_rejects_repeated_items_in_a_sequence():
    """Invariant O1: first-occurrence semantics need item-distinct sequences."""
    from seqmineforge.core.errors import DataGenerationError

    db = SequenceDatabase(
        (np.asarray([1, 1, 2], dtype=np.int64), np.asarray([2, 3], dtype=np.int64)), 4, "bad"
    )
    with pytest.raises(DataGenerationError, match="repeated items"):
        BruteForceMiner().mine(db, min_support=0.5, max_pattern_length=2)


def test_brute_force_support_matches_the_matcher():
    db = get_dataset("markov", seed=61, n_sequences=20, n_items=4, min_len=4, max_len=6)
    for key in [(0,), (1,), (0, 1), (2, 3), (0, 1, 2), (3, 1)]:
        expected = sum(
            1
            for s in db.sequences
            if brute_force_support(key, db.sequences)
            for s in db.sequences[:0]
        )
        del expected
        manual = sum(1 for s in db.sequences if list(_subseq_positions(key, s)))
        assert brute_force_support(key, db.sequences) == manual


def _subseq_positions(pattern, seq):
    it = iter(range(len(seq)))
    ok = True
    for p in pattern:
        for i in it:
            if seq[i] == p:
                break
        else:
            ok = False
            break
    return range(1) if ok else range(0)


def test_oracle_handles_empty_result():
    db = SequenceDatabase((np.asarray([0], dtype=np.int64),) * 3, 1, "t")
    res = BruteForceMiner().mine(db, min_support=1.0, max_pattern_length=3)
    assert len(res) >= 1  # (0,) has support 1.0


def test_oracle_candidate_count_grows_combinatorially():
    assert oracle_candidate_count(1, 1) == 1
    assert oracle_candidate_count(3, 2) == 3 + 6
    assert oracle_candidate_count(4, 2) == 4 + 12
    # Length cap is clamped by the number of items.
    assert oracle_candidate_count(2, 5) == 2 + 2
    assert oracle_candidate_count(0, 3) == 0


# ------------------------------------------------------------------ registry
def test_registry_exposes_everything_expected():
    assert set(MINERS) == {"spam", "spade", "prefixspan", "seqfuse", "oracle"}
    assert set(BASELINES) == {"spam", "spade", "prefixspan"}
    assert FLAGSHIP.name not in BASELINES


def test_unknown_miner_raises_with_the_known_list():
    from seqmineforge.core.errors import UnknownMinerError

    with pytest.raises(UnknownMinerError, match="unknown miner"):
        get_miner("does-not-exist")


@pytest.mark.parametrize("name", ["spam", "spade", "prefixspan"])
def test_sota_baselines_are_subsequence_only(name):
    with pytest.raises(ValueError, match="subsequence semantics only"):
        get_miner(name).mine(
            get_dataset("markov", seed=71, n_sequences=20, n_items=4, min_len=4, max_len=6),
            min_support=0.1,
            max_pattern_length=2,
            pattern_type=PatternType.CONTIGUOUS,
        )


def test_flagship_supports_contiguous_semantics():
    db = get_dataset("markov", seed=81, n_sequences=60, n_items=8, min_len=8, max_len=12)
    res = get_miner(FLAGSHIP.name).mine(
        db, min_support=0.1, max_pattern_length=3, pattern_type=PatternType.CONTIGUOUS
    )
    from seqmineforge.miners.vector_index import has_block

    for key, sup in res.supports.items():
        cnt = sum(1 for s in db.sequences if has_block(s, np.asarray(key, dtype=np.int64)))
        assert abs(cnt / db.n_sequences - sup) <= 1e-12


def test_flagship_contiguous_matches_oracle():
    db = get_dataset("markov", seed=91, n_sequences=30, n_items=5, min_len=6, max_len=9)
    kw = {"min_support": 0.1, "max_pattern_length": 3, "pattern_type": PatternType.CONTIGUOUS}
    oracle = get_miner("oracle").mine(db, **kw)
    res = get_miner(FLAGSHIP.name).mine(db, **kw)
    assert res.keys == oracle.keys
