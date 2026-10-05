"""Invariants of the shared matching primitives.

These are the tests that matter most in the whole package: every miner and the
oracle agree *because* these functions are the single definition of "contains".
A regression here silently changes every reported number.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from seqmineforge.core.errors import ConfigError
from seqmineforge.core.types import PatternType, SequenceDatabase
from seqmineforge.miners.budget import (
    budget_for,
    effective_max_length,
    support_threshold,
)
from seqmineforge.miners.matching import (
    PROJECTED_DB_CONSUMES_ONE,
    apriori_prune,
    contains_contiguous,
    contains_subsequence,
    extend_projection,
    greedy_match_end,
    profile_items,
    project_sequence,
)


def arr(*vals: int) -> np.ndarray:
    return np.asarray(vals, dtype=np.int64)


# ---------------------------------------------------------------- matching
def test_contains_subsequence_gap_allowed():
    assert contains_subsequence(arr(1, 3), arr(1, 2, 3))
    assert contains_subsequence(arr(2, 3), arr(1, 2, 3))
    assert contains_subsequence(arr(1), arr(1, 2, 3))


def test_contains_subsequence_order_matters():
    """The direction of a pattern is part of its identity."""
    assert not contains_subsequence(arr(3, 1), arr(1, 2, 3))
    assert contains_subsequence(arr(1, 3), arr(1, 2, 3))


def test_contains_subsequence_trivial_cases():
    assert contains_subsequence(arr(), arr(1, 2))
    assert contains_subsequence(arr(), arr())  # empty pattern always matches
    assert contains_subsequence(arr(1, 2, 3), arr(1, 2, 3))
    # A non-empty pattern cannot occur in an empty sequence.
    assert not contains_subsequence(arr(1), arr())
    assert not contains_subsequence(arr(1, 2), arr(1))


def test_contains_contiguous_requires_adjacency():
    assert contains_contiguous(arr(2, 3), arr(1, 2, 3))
    assert not contains_contiguous(arr(1, 3), arr(1, 2, 3))
    assert contains_contiguous(arr(), arr(1))


def test_greedy_match_end_reports_end_position():
    s = arr(1, 2, 3, 1, 2)
    assert greedy_match_end(arr(1, 2), s) == 2  # matched at idx 0,1 -> end 2
    assert greedy_match_end(arr(1, 2, 3), s) == 3  # idx 0,1,2 -> end 3
    assert greedy_match_end(arr(3, 1), s) == 4  # idx 2,3 -> end 4 (not 5)
    assert greedy_match_end(arr(9), s) == -1


# ------------------------------------------------- projected-database contract
def test_project_suffix_starts_at_last_matched_element():
    s = arr(7, 1, 2, 3)
    projected = project_sequence(arr(1, 2), s)
    assert projected is not None
    # Suffix begins AT the last matched element (the `2`), so exactly one
    # element is consumed -- this is the whole point of the convention.
    assert projected.tolist() == [2, 3]
    assert int(projected[0]) == 2


def test_project_absent_returns_none():
    assert project_sequence(arr(9), arr(1, 2)) is None


def test_extend_projection_finds_only_the_new_item():
    """The canonical bug: searching the suffix for the *whole* pattern fails.

    ``[0, 2, 6]`` projected under ``(0, 2)`` gives ``[2, 6]``.  Re-searching that
    for ``(0,2,6)`` returns None because the ``0`` is gone, silently dropping a
    pattern that genuinely occurs.  Searching only for the new item works.
    """
    s = arr(0, 2, 6)
    p2 = project_sequence(arr(0, 2), s)
    assert p2 is not None and p2.tolist() == [2, 6]
    assert project_sequence(arr(0, 2, 6), p2) is None  # the wrong way
    p3 = extend_projection(p2, PROJECTED_DB_CONSUMES_ONE, 6)
    assert p3 is not None and p3.tolist() == [6]  # the right way


def test_extend_projection_does_not_reuse_the_consumed_slot():
    """Passing ``consumed=0`` is refused outright, so the item cannot re-match."""
    s = arr(0, 2, 6)
    p2 = project_sequence(arr(0, 2), s)  # [2, 6]
    # Item 2 sits at index 0 of the suffix -- that slot is the already-matched
    # element, so extending with 2 must fail rather than loop.
    assert extend_projection(p2, 1, 2) is None
    # And the misuse that would allow it is rejected at the boundary.
    with pytest.raises(ValueError, match="consumed must be >= 1"):
        extend_projection(p2, 0, 2)


def test_extend_projection_validates_inputs():
    with pytest.raises(ValueError, match="consumed must be >= 1"):
        extend_projection(arr(1, 2), 0, 1)


def test_profile_items_counts_beyond_the_consumed_slot():
    pdb = [arr(1, 2, 3), arr(1, 2), arr(1, 4)]
    counts = profile_items(1, pdb)
    # (1,) is consumed everywhere; 2 occurs in 2 suffixes, 3 in 1, 4 in 1.
    assert counts == {2: 2, 3: 1, 4: 1}
    assert 1 not in counts


def test_profile_items_validates_consumed():
    with pytest.raises(ValueError, match="consumed must be >= 1"):
        profile_items(0, [arr(1, 2)])


def test_apriori_prune_is_sound_but_not_selective():
    seqs = (arr(1, 2, 3), arr(4, 5))
    assert apriori_prune(seqs, (1, 2))  # both present
    assert not apriori_prune(seqs, (1, 9))  # suffix absent everywhere
    assert apriori_prune(seqs, (5,))  # length 1 always passes


# -------------------------------------------------------------------- budget
@pytest.mark.parametrize(
    ("min_support", "n", "expected"),
    [
        (0.05, 200, 10),
        (0.05, 100, 5),
        (0.10, 100, 10),
        (1.0, 100, 100),
        (0.001, 10, 1),  # ceil(0.01) -> clamped to 1
        (0.5, 3, 2),  # ceil(1.5) = 2
    ],
)
def test_support_threshold_is_ceil_and_clamped(min_support, n, expected):
    assert support_threshold(min_support, n) == expected


def test_support_threshold_rejects_out_of_range():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_support"):
            support_threshold(bad, 10)


def test_effective_max_length_bounds_by_shortest_sequence():
    db = SequenceDatabase((arr(1, 2, 3), arr(1, 2)), 5, name="t")
    # Nothing longer than the shortest sequence can occur.
    assert effective_max_length(10, db) == 2
    assert effective_max_length(1, db) == 1


def test_effective_max_length_rejects_zero():
    db = SequenceDatabase((arr(1, 2),), 3, name="t")
    with pytest.raises(ValueError, match="max_pattern_length"):
        effective_max_length(0, db)


def test_budget_for_combines_both_bounds():
    db = SequenceDatabase((arr(1, 2, 3), arr(1, 2)), 5, name="t")
    assert budget_for(0.4, 10, db) == (1, 2)


# --------------------------------------------------------------------- types
def test_database_rejects_empty():
    with pytest.raises(ValueError, match="at least one sequence"):
        SequenceDatabase((), 3, name="empty")


def test_database_rejects_nonpositive_universe():
    with pytest.raises(ValueError, match="n_items must be positive"):
        SequenceDatabase((arr(1),), 0, name="bad")


def test_item_support_counts_sequences_not_occurrences():
    # (1,2), (2,3), (1,3) over a 4-item universe [0..3]:
    # item 0 appears in none, items 1/2/3 each appear in exactly two.
    db = SequenceDatabase((arr(1, 2), arr(2, 3), arr(1, 3)), 4, name="t")
    counts = db.item_support()
    assert counts.tolist() == [0, 2, 2, 2]


def test_pattern_rejects_out_of_range_support():
    from seqmineforge.core.types import Pattern

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Pattern((1,), 1.5, PatternType.SUBSEQUENCE)


def test_pattern_rejects_empty():
    from seqmineforge.core.types import Pattern

    with pytest.raises(ValueError, match="non-empty"):
        Pattern((), 1.0, PatternType.SUBSEQUENCE)


def test_mining_result_sorting_is_deterministic():
    from seqmineforge.core.types import MiningResult, Pattern

    pats = (
        Pattern((2,), 0.5, PatternType.SUBSEQUENCE),
        Pattern((1,), 0.5, PatternType.SUBSEQUENCE),
        Pattern((0,), 0.9, PatternType.SUBSEQUENCE),
    )
    res = MiningResult(patterns=pats, name="t")
    ordered = [p.items for p in res.sorted_patterns()]
    assert ordered == [(0,), (1,), (2,)]


def test_config_rejects_bad_values():
    from seqmineforge.core.config import Config

    with pytest.raises(ConfigError, match="min_support"):
        Config(min_support=0.0)
    with pytest.raises(ConfigError, match="pattern_type"):
        Config(pattern_type="fuzzy")
    with pytest.raises(ConfigError, match="n_seeds"):
        Config(n_seeds=1)
