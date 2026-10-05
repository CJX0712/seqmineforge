"""Tests for the vectorised engines, the closure basis, and the CLI.

Author: 晨星
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seqmineforge.cli import main
from seqmineforge.core.types import PatternType, SequenceDatabase
from seqmineforge.data import get_dataset
from seqmineforge.eval import (
    available_sklearn,
    logistic_accuracy,
    make_binary_labels,
    stratified_split,
)
from seqmineforge.features import (
    BinaryPatternFeatures,
    SupportPatternFeatures,
    closed_patterns,
    compression_ratio,
    extension_index,
    is_closed,
    reconstruct,
)
from seqmineforge.miners.matching import contains_subsequence
from seqmineforge.miners.oracle import brute_force_support
from seqmineforge.miners.vector_index import BitsetIndex, SequenceIndex, has_block

PATTERNS = [(0,), (1,), (0, 1), (2, 3), (0, 1, 2), (1, 3)]


# ------------------------------------------------------------- SequenceIndex
@pytest.mark.parametrize("seed", [101, 102, 103])
def test_sequence_index_agrees_with_brute_force(seed):
    db = get_dataset("markov", seed=seed, n_sequences=80, n_items=10, min_len=8, max_len=14)
    index = SequenceIndex(db)
    for key in PATTERNS:
        assert index.support(key) == brute_force_support(key, db.sequences)


def test_sequence_index_item_support():
    db = get_dataset("grouped", seed=111, n_sequences=90, n_groups=4, group_len=3, tail_len=5)
    index = SequenceIndex(db)
    truth = db.item_support()
    for item in range(db.n_items):
        assert index.item_support(item) == int(truth[item])
    assert index.item_support_vector(db.n_items).tolist() == truth.tolist()


def test_sequence_index_degenerate_patterns():
    db = SequenceDatabase(
        (np.asarray([0, 1], dtype=np.int64), np.asarray([1, 0], dtype=np.int64)), 2, "t"
    )
    index = SequenceIndex(db)
    assert index.support(()) == 2  # empty pattern is in every sequence
    assert index.support((0, 1)) == 1
    assert index.support((1, 0)) == 1
    assert index.support((0, 1, 0)) == 0  # longer than any sequence
    assert index.support((0, 0)) == 0  # repeated item cannot match distinct items


def test_sequence_index_long_pattern_is_zero_not_crash():
    db = get_dataset("markov", seed=121, n_sequences=20, n_items=4, min_len=4, max_len=6)
    index = SequenceIndex(db)
    assert index.support((0, 1, 2, 3, 0, 1, 2, 3)) == 0


# --------------------------------------------------------------- BitsetIndex
def test_bitset_index_matches_has_block():
    db = get_dataset("markov", seed=131, n_sequences=70, n_items=9, min_len=8, max_len=12)
    index = BitsetIndex(db)
    for key in PATTERNS:
        arr = np.asarray(key, dtype=np.int64)
        expected = sum(1 for s in db.sequences if has_block(s, arr))
        assert index.support(key) == expected
        mask = index.contains_contiguous(key)
        assert int(mask.sum()) == expected


def test_has_block_break_is_valid_because_flatnonzero_is_ascending():
    # The `break` on overflow relies on ascending hit positions.
    s = np.asarray([1, 1], dtype=np.int64)
    assert has_block(s, np.asarray([1, 1], dtype=np.int64))
    assert not has_block(s, np.asarray([1, 1, 1], dtype=np.int64))
    assert has_block(s, np.asarray([], dtype=np.int64))


# ------------------------------------------------------------------ closure
def test_closure_detects_equal_support_extensions():
    """The hand-built case: sup(0)=sup(0,1)=sup(0,1,2)=1.0 leaves one closed."""
    supports = {(0,): 1.0, (0, 1): 1.0, (0, 1, 2): 1.0, (5,): 0.5}
    index = extension_index(supports)
    assert is_closed((0,), supports, index) is False
    assert is_closed((0, 1), supports, index) is False
    assert is_closed((0, 1, 2), supports, index) is True
    assert is_closed((5,), supports, index) is True
    assert closed_patterns(list(supports), supports) == [(0, 1, 2), (5,)]
    assert compression_ratio(4, 2) == 2.0


def test_closure_uses_equality_not_strict_inequality():
    """A regression guard: downward closure makes ``sup > target`` always False,
    which would mark every pattern closed and report ratio 1.00."""
    supports = {(0,): 1.0, (0, 1): 0.5, (0, 2): 1.0}
    index = extension_index(supports)
    assert is_closed((0,), supports, index) is False  # extension (0,2) ties


def test_closed_basis_reconstructs_the_full_set():
    db = get_dataset("grouped", seed=141, n_sequences=120, n_groups=4, group_len=3, tail_len=5)
    from seqmineforge.miners import get_miner

    res = get_miner("seqfuse").mine(db, min_support=0.08, max_pattern_length=3)
    closed = closed_patterns(list(res.keys), res.supports)
    assert reconstruct(closed, list(res.keys)) == res.keys


def test_compression_ratio_handles_empty_closed_set():
    assert compression_ratio(5, 0) == float("inf")
    assert compression_ratio(0, 0) == 1.0


def test_extension_index_is_prefix_keyed():
    supports = {(0,): 0.5, (0, 1): 0.4, (0, 2): 0.3, (1,): 0.6}
    idx = extension_index(supports)
    assert set(idx[()]) == {(0,), (1,)}
    assert set(idx[(0,)]) == {(0, 1), (0, 2)}


# ----------------------------------------------------------------- features
def test_binary_features_are_occurrence_matrix():
    seqs = (
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
    )
    feats = BinaryPatternFeatures([(0, 1), (0, 2), (3,)])
    x = feats.transform(seqs)
    assert x.shape == (2, 3)
    assert x[0].tolist() == [1.0, 1.0, 0.0]  # (0,2) is a subsequence of [0,1,2]
    assert x[1].tolist() == [0.0, 0.0, 1.0]


def test_binary_features_on_empty_pattern_list():
    x = BinaryPatternFeatures([]).transform((np.asarray([0], dtype=np.int64),))
    assert x.shape == (1, 0)


def test_support_features_broadcast_constants():
    seqs = (np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64))
    feats = SupportPatternFeatures([(0,), (1,)], {(0,): 0.5, (1,): 0.25})
    x = feats.transform(seqs)
    assert x.tolist() == [[0.5, 0.25], [0.5, 0.25]]


# --------------------------------------------------------------- downstream
def test_labels_and_split_have_no_leakage():
    db = get_dataset("grouped", seed=151, n_sequences=200, n_groups=4, group_len=3, tail_len=5)
    y = make_binary_labels(db, positive_items={0, 1, 2})
    tr, te = stratified_split(y, test_size=0.3, seed=151)
    assert np.intersect1d(tr, te).size == 0
    assert tr.size + te.size == db.n_sequences
    # Both folds must carry both classes, else accuracy degenerates.
    assert len(set(y[tr].tolist())) == 2
    assert len(set(y[te].tolist())) == 2


def test_label_helper_rejects_single_class():
    db = SequenceDatabase(
        (np.asarray([0], dtype=np.int64), np.asarray([0, 1], dtype=np.int64)), 2, "t"
    )
    with pytest.raises(ValueError, match="single class"):
        make_binary_labels(db, positive_items={0})
    with pytest.raises(ValueError, match="non-empty"):
        make_binary_labels(db, positive_items=set())


def test_split_rejects_degenerate_test_size():
    y = np.array([0, 1, 0, 1, 0, 1])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="test_size"):
            stratified_split(y, test_size=bad, seed=1)


def test_logistic_accuracy_runs_on_both_backends():
    rng = np.random.default_rng(0)
    x_tr = rng.normal(size=(60, 4))
    y_tr = (x_tr[:, 0] > 0).astype(np.int64)
    x_te = rng.normal(size=(20, 4))
    y_te = (x_te[:, 0] > 0).astype(np.int64)
    acc = logistic_accuracy(x_tr, y_tr, x_te, y_te, seed=7)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.7  # separable


def test_available_sklearn_is_a_bool():
    assert isinstance(available_sklearn(), bool)


# ----------------------------------------------------------------------- CLI
def test_cli_info_exits_zero(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "seqfuse" in out
    assert "min_support" in out


def test_cli_mine_prints_patterns(capsys):
    assert (
        main(
            [
                "mine",
                "--dataset",
                "grouped",
                "--miner",
                "seqfuse",
                "--seed",
                "161",
                "--min-support",
                "0.1",
                "--max-length",
                "3",
                "--top",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "patterns=" in out
    assert "support=" in out


def test_cli_verify_passes_on_small_dataset(capsys):
    code = main(["verify", "--dataset", "grouped", "--seed", "171", "--max-length", "3"])
    captured = capsys.readouterr().out
    assert code == 0
    assert "PASS" in captured


def test_cli_reports_config_error_as_exit_2(capsys):
    code = main(["mine", "--min-support", "-1"])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_cli_env_override(monkeypatch):
    from seqmineforge.core.config import load_config

    monkeypatch.setenv("ENV_SMF_MIN_SUPPORT", "0.25")
    monkeypatch.setenv("ENV_SMF_MAX_PATTERN_LENGTH", "3")
    cfg = load_config()
    assert cfg.min_support == 0.25
    assert cfg.max_pattern_length == 3


def test_cli_kwargs_beat_environment(monkeypatch):
    from seqmineforge.core.config import load_config

    monkeypatch.setenv("ENV_SMF_MIN_SUPPORT", "0.25")
    assert load_config(min_support=0.1).min_support == 0.1


def test_cli_rejects_unknown_config_field():
    from seqmineforge.core.config import load_config
    from seqmineforge.core.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown config field"):
        load_config(nonsense=1)


def test_cli_bad_bool_env(monkeypatch):
    from seqmineforge.core.config import load_config
    from seqmineforge.core.errors import ConfigError

    # No boolean field exists today, but the coercion path must still be typed.
    monkeypatch.setenv("ENV_SMF_PATTERN_TYPE", "sideways")
    with pytest.raises(ConfigError, match="pattern_type"):
        load_config()


def test_cli_bench_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    code = main(["bench", "--out", str(out), "--seed", "181"])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["gates"]["all_passed"] is True
    assert Path(str(out)).exists()
    assert data["efficiency"]["aggregate_speedup"] > 1.0


# --------------------------------------------------------------- determinism
def test_seed_set_all_is_idempotent():
    from seqmineforge.core.seed import set_all

    a = set_all(7)
    x1 = a.random(5)
    b = set_all(7)
    x2 = b.random(5)
    assert x1.tolist() == x2.tolist()


def test_seed_rng_streams_are_independent():
    from seqmineforge.core.seed import rng

    a = rng(5, stream=0).random(5)
    b = rng(5, stream=1).random(5)
    assert a.tolist() != b.tolist()
    assert a.tolist() == rng(5, stream=0).random(5).tolist()


def test_seed_rejects_bad_input():
    from seqmineforge.core.seed import set_all

    with pytest.raises(ValueError, match="non-negative"):
        set_all(-1)
    with pytest.raises(TypeError, match="must be an integer"):
        set_all(1.5)


# ------------------------------------------------------------------- DGP
def test_dgp_lookup_reports_known_names():
    from seqmineforge.data import get_dataset

    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nope", seed=1)


def test_generated_sequences_have_distinct_items():
    """Invariant O1 depends on this; assert it at the source."""
    for name, kwargs in [
        ("markov", {"seed": 191, "n_sequences": 40, "n_items": 8, "min_len": 6, "max_len": 10}),
        (
            "planted",
            {
                "seed": 192,
                "n_sequences": 40,
                "n_items": 8,
                "min_len": 6,
                "max_len": 10,
                "n_planted": 2,
                "planted_len": 2,
            },
        ),
        ("segmented", {"seed": 193, "n_sequences": 40, "n_items": 8, "min_len": 6, "max_len": 10}),
        ("grouped", {"seed": 194, "n_sequences": 40, "n_groups": 3, "group_len": 2, "tail_len": 4}),
    ]:
        db = get_dataset(name, **kwargs)
        for seq in db.sequences:
            assert len(np.unique(seq)) == len(seq), f"{name} produced a repeated item"


def test_generated_items_are_in_range():
    db = get_dataset("grouped", seed=201, n_sequences=40, n_groups=3, group_len=2, tail_len=4)
    for seq in db.sequences:
        assert seq.min() >= 0
        assert seq.max() < db.n_items


def test_dgp_rejects_infeasible_planting():
    from seqmineforge.core.errors import DataGenerationError

    with pytest.raises(DataGenerationError, match="item-disjoint"):
        get_dataset("planted", seed=1, n_sequences=20, n_items=4, n_planted=8, planted_len=2)


def test_dgp_validates_noise_range():
    from seqmineforge.core.errors import DataGenerationError

    with pytest.raises(DataGenerationError, match="noise"):
        get_dataset("markov", seed=1, n_sequences=20, n_items=5, noise=1.5)


def test_dgp_rejects_too_few_sequences():
    from seqmineforge.core.errors import DataGenerationError

    with pytest.raises(DataGenerationError, match="n_sequences"):
        get_dataset("markov", seed=1, n_sequences=1, n_items=5)


# ------------------------------------------------------- pipeline structure
def test_pipeline_report_round_trips(tmp_path):
    from seqmineforge.pipeline import SeqMinePipeline

    pipe = SeqMinePipeline(min_support=0.1, max_pattern_length=3)
    path = tmp_path / "r.json"
    report = pipe.run(output=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["gates"]["all_passed"] is True
    assert set(data) >= {
        "config",
        "backend",
        "datasets",
        "efficiency",
        "verification",
        "ablation",
        "utility",
        "determinism",
        "failure_cases",
        "gates",
    }
    assert report.elapsed_sec > 0


def test_pipeline_determinism_is_bitwise():
    from seqmineforge.pipeline import SeqMinePipeline

    pipe = SeqMinePipeline()
    det = pipe.determinism_check(pipe.cfg.seed)
    assert det["deterministic"] is True
    assert det["max_abs_delta"] == 0.0


def test_failure_cases_are_derived_not_narrative():
    from seqmineforge.pipeline import SeqMinePipeline

    pipe = SeqMinePipeline()
    rows, _ = pipe.run_efficiency(seeds=[pipe.cfg.seed])
    cases = pipe.collect_failure_cases(rows)
    assert len(cases) >= 3
    for case in cases:
        assert "condition" in case and "observed" in case and "cause" in case


def test_pattern_type_enum_is_exported():
    assert PatternType.SUBSEQUENCE.value == "subsequence"
    assert PatternType.CONTIGUOUS.value == "contiguous"


def test_contains_subsequence_is_the_shared_definition():
    """Guard against two matchers drifting apart."""
    db = get_dataset("markov", seed=211, n_sequences=30, n_items=6, min_len=5, max_len=9)
    for key in PATTERNS:
        manual = sum(
            1 for s in db.sequences if contains_subsequence(np.asarray(key, dtype=np.int64), s)
        )
        assert brute_force_support(key, db.sequences) == manual
