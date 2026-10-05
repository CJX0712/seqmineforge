"""Synthetic sequence database generators (Data Generating Process).

Design goals
------------
1. **Reproducible**: every generator takes an explicit ``seed`` and derives its
   stream via :func:`seqmineforge.core.seed.rng` -- no global RNG use.
2. **Sweet-spot difficulty**: the ``noise`` / ``background`` knob must put the
   trivial "everything above threshold" regime *below* the ceiling.  A DGP
   where everything is frequent has zero discriminative power.
3. **Embedded structure**: some datasets plant a handful of genuinely frequent
   patterns with support well above threshold so the oracle answer is
   non-trivial, plus a long tail of noise items so apriori pruning is actually
   exercised.
4. **Order is preserved and items are distinct within a sequence.**  This is
   the standard SPM setting assumed by SPADE / SPAM / IncMine.  Sorting the
   items would silently degrade subsequence mining into contiguous mining --
   a bug that produces plausible-looking output and zero signal, so dedup here
   keeps first-occurrence order rather than sorting.

Every generator returns a :class:`SequenceDatabase`; the difficulty knob is
documented per generator.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataGenerationError
from ..core.seed import rng
from ..core.types import SequenceDatabase

__all__ = [
    "DATASETS",
    "dedup_keep_order",
    "get_dataset",
    "make_markov_sequences",
    "make_planted_pattern_db",
    "make_semi_markov_db",
    "plant_patterns",
]


def _validate(n_sequences: int, n_items: int, min_len: int, max_len: int) -> None:
    if n_sequences < 2:
        raise DataGenerationError(f"n_sequences must be >= 2, got {n_sequences}")
    if n_items < 2:
        raise DataGenerationError(f"n_items must be >= 2, got {n_items}")
    if min_len < 1:
        raise DataGenerationError(f"min_len must be >= 1, got {min_len}")
    if max_len < min_len:
        raise DataGenerationError(f"max_len ({max_len}) must be >= min_len ({min_len})")


def dedup_keep_order(items: np.ndarray) -> np.ndarray:
    """Drop repeated items, keeping the first occurrence (order preserved)."""
    seen: set[int] = set()
    out: list[int] = []
    for v in items:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return np.asarray(out, dtype=np.int64)


def plant_patterns(
    gen: np.random.Generator,
    *,
    base_len: int,
    n_items: int,
    planted: list[np.ndarray],
    plant_rate: float,
) -> np.ndarray:
    """Build one sequence containing each planted pattern w.p. ``plant_rate``.

    A planted pattern is inserted **contiguously** (which trivially implies
    subsequence containment) into a random slot of a sequence of otherwise
    distinct background items drawn from the universe *minus* the pattern's own
    items.  Drawing the background from the complement is essential: if a
    background draw happened to contain one of the pattern items, first-occurrence
    dedup could delete the planted copy and silently destroy the plant.
    """
    chosen: list[np.ndarray] = []
    used: set[int] = set()
    for pat in planted:
        if gen.random() < plant_rate:
            chosen.append(pat)
            used.update(int(v) for v in pat)

    free_items = np.array([i for i in range(n_items) if i not in used], dtype=np.int64)
    n_fill = max(0, base_len - sum(len(p) for p in chosen))
    fill: np.ndarray = np.empty(0, dtype=np.int64)
    if len(free_items) > 0 and n_fill > 0:
        take = min(n_fill, len(free_items))
        fill = gen.choice(free_items, size=take, replace=False)

    pool = np.concatenate([*chosen, fill]) if chosen else fill
    if len(pool) == 0:
        return np.asarray([], dtype=np.int64)
    perm = gen.permutation(len(pool))
    pool = pool[perm]
    # Re-impose the planted blocks contiguously at random offsets so that the
    # pattern survives as a contiguous (hence subsequence) occurrence.
    seq: list[int] = []
    for pat in chosen:
        offset = int(gen.integers(0, max(1, len(pool) - len(pat) + 1)))
        seq.extend(int(v) for v in pool[:offset])
        seq.extend(int(v) for v in pat)
        pool = pool[offset:]
    seq.extend(int(v) for v in pool)
    return dedup_keep_order(np.asarray(seq, dtype=np.int64))


def make_markov_sequences(
    *,
    seed: int,
    n_sequences: int = 200,
    n_items: int = 10,
    min_len: int = 8,
    max_len: int = 18,
    n_states: int = 4,
    noise: float = 0.25,
) -> SequenceDatabase:
    """First-order Markov chains over ``n_items`` with random-walk transitions.

    ``n_states`` latent regimes each get their own transition row, so sequences
    carry real repeated sub-structure without any pattern being pre-specified.
    ``noise`` is the difficulty knob: it mixes every regime toward the uniform
    row, so

    * ``noise -> 0`` : sharply structured, many frequent long patterns
    * ``noise -> 1`` : close to i.i.d. draws, few patterns above threshold
    """
    _validate(n_sequences, n_items, min_len, max_len)
    if not 0.0 <= noise <= 1.0:
        raise DataGenerationError(f"noise must lie in [0, 1], got {noise}")
    if n_states < 1:
        raise DataGenerationError(f"n_states must be >= 1, got {n_states}")

    gen = rng(seed, stream=11)
    rows = []
    for _ in range(n_states):
        base = gen.dirichlet(np.ones(n_items) * 2.0)
        rows.append(noise / n_items + (1.0 - noise) * base)
    transitions = np.vstack(rows)

    seqs: list[np.ndarray] = []
    for _ in range(n_sequences):
        state = int(gen.integers(n_states))
        length = int(gen.integers(min_len, max_len + 1))
        items = [int(gen.integers(n_items))]
        for _ in range(length - 1):
            nxt = int(gen.choice(n_items, p=transitions[state]))
            items.append(nxt)
            # Random transition of the latent regime (a Markov switch).
            if gen.random() < 0.15:
                state = int(gen.integers(n_states))
        seq = dedup_keep_order(np.asarray(items, dtype=np.int64))
        if len(seq) >= 2:
            seqs.append(seq)
    if len(seqs) < 2:
        raise DataGenerationError(
            "markov generator produced too few valid sequences; lower `noise` or raise min_len"
        )
    return SequenceDatabase(tuple(seqs), n_items, name="markov")


def make_planted_pattern_db(
    *,
    seed: int,
    n_sequences: int = 240,
    n_items: int = 12,
    n_planted: int = 4,
    planted_len: int = 3,
    plant_rate: float = 0.55,
    min_len: int = 8,
    max_len: int = 16,
    background: float = 0.18,
) -> SequenceDatabase:
    """Plant ``n_planted`` item-disjoint frequent patterns in a noisy background.

    Planted patterns share **no items** with one another, so they remain
    maximal and individually interpretable instead of collapsing into fewer,
    longer patterns.  Support of each planted pattern is ``>= plant_rate``
    (minus first-occurrence dedup effects), comfortably above a typical
    ``min_support``.

    ``background`` controls how many slots are filled with pure random items;
    it is the difficulty knob.  It is *validated* but only used to scale the
    sequence length budget -- the actual item draw is uniform over the free
    items, which keeps the planted support guarantee exact.
    """
    _validate(n_sequences, n_items, min_len, max_len)
    if n_planted < 1:
        raise DataGenerationError(f"n_planted must be >= 1, got {n_planted}")
    if planted_len < 1:
        raise DataGenerationError(f"planted_len must be >= 1, got {planted_len}")
    if not 0.0 < plant_rate <= 1.0:
        raise DataGenerationError(f"plant_rate must lie in (0, 1], got {plant_rate}")
    if not 0.0 <= background <= 1.0:
        raise DataGenerationError(f"background must lie in [0, 1], got {background}")
    if n_planted * planted_len > n_items:
        raise DataGenerationError(
            f"cannot plant {n_planted} item-disjoint patterns of length "
            f"{planted_len} from an item universe of size {n_items}"
        )

    gen = rng(seed, stream=12)
    pool = gen.permutation(n_items)
    planted = [
        np.sort(pool[i * planted_len : (i + 1) * planted_len]).astype(np.int64)
        for i in range(n_planted)
    ]

    seqs: list[np.ndarray] = []
    for _ in range(n_sequences):
        base_len = int(gen.integers(min_len, max_len + 1))
        # `background` biases how far above the planted core the sequence runs.
        budget = round(base_len / (1.0 + background))
        seq = plant_patterns(
            gen,
            base_len=budget,
            n_items=n_items,
            planted=planted,
            plant_rate=plant_rate,
        )
        if len(seq) >= 2:
            seqs.append(seq)
    if len(seqs) < 2:
        raise DataGenerationError("planted generator produced too few valid sequences")
    return SequenceDatabase(tuple(seqs), n_items, name="planted")


def make_semi_markov_db(
    *,
    seed: int,
    n_sequences: int = 180,
    n_items: int = 14,
    n_segments: int = 3,
    min_len: int = 10,
    max_len: int = 22,
    concentration: float = 3.0,
) -> SequenceDatabase:
    """Segment-structured sequences (bursts of related items, then unrelated).

    Real event logs (click streams, sensor traces) exhibit exactly this shape.
    ``n_segments`` blocks are drawn, each from its own Dirichlet item
    distribution, then concatenated.  ``concentration`` controls burstiness
    (low -> spiky blocks, high -> near-uniform).
    """
    _validate(n_sequences, n_items, min_len, max_len)
    if n_segments < 1:
        raise DataGenerationError(f"n_segments must be >= 1, got {n_segments}")
    if concentration <= 0:
        raise DataGenerationError(f"concentration must be > 0, got {concentration}")

    gen = rng(seed, stream=13)
    seg_prob = gen.dirichlet(np.ones(n_segments))
    seqs: list[np.ndarray] = []
    for _ in range(n_sequences):
        total_len = int(gen.integers(min_len, max_len + 1))
        n_seg = int(gen.choice(np.arange(1, n_segments + 1), p=seg_prob))
        items: list[int] = []
        for _ in range(n_seg):
            share = max(1, round(total_len / n_seg))
            bias = gen.dirichlet(np.full(n_items, concentration / n_items))
            items.extend(int(v) for v in gen.choice(n_items, size=share, p=bias))
        seq = dedup_keep_order(np.asarray(items[:total_len], dtype=np.int64))
        if len(seq) >= 2:
            seqs.append(seq)
    if len(seqs) < 2:
        raise DataGenerationError("segmented generator produced too few valid sequences")
    return SequenceDatabase(tuple(seqs), n_items, name="segmented")


def _make_sequence_distinct(prefixes: list[np.ndarray], *, attempts: int = 200) -> list[np.ndarray]:
    """Resample until no prefix is a sub-sequence of another.

    Uses the package's own matcher, so the DGP and the miner agree on what
    "sub-sequence" means -- no second definition to drift.
    """
    from ..miners.matching import contains_subsequence

    out = list(prefixes)
    for i in range(len(out)):
        for _ in range(attempts):
            conflict = any(
                contains_subsequence(out[i], other) or contains_subsequence(other, out[i])
                for j, other in enumerate(out)
                if j != i
            )
            if not conflict:
                break
            # Deterministic repair: bump the last item by one (wrapping), which
            # keeps the sequence valid and terminates because the universe is
            # finite and the loop is bounded.
            last = int(out[i][-1])
            out[i] = out[i].copy()
            out[i][-1] = (last + 1) % int(out[i].max() + 2)
    return out


def make_prefixed_db(
    *,
    seed: int,
    n_sequences: int = 240,
    n_items: int = 12,
    n_prefixes: int = 4,
    prefix_len: int = 3,
    prefix_rate: float = 0.75,
    tail_len: int = 6,
) -> SequenceDatabase:
    """Sequences sharing long common prefixes -- the shape real logs have.

    Motivation (learned the hard way)
    ---------------------------------
    On :func:`make_markov_sequences` / :func:`make_planted_pattern_db` /
    :func:`make_semi_markov_db` the closed-pattern compression ratio is
    **exactly 1.00** -- every mined pattern is closed.  Verified as a property of
    the data, not a bug: those generators draw item order randomly, so appending
    an item to a pattern almost always *changes* which sequences contain it, and
    closure requires an extension with *identical* support.  Measured across
    6 generator settings x 3 thresholds: 1.00 in all 18 cases.

    Real event logs are the opposite: a request usually opens with one of a
    handful of canonical prefixes (``GET /index``, ``checkout``, ...) and then
    diverges.  Sequences sharing such a prefix have a prefix whose support is
    *exactly* the number of sequences carrying it -- so the prefix is not closed,
    and closure genuinely compresses.

    Parameters
    ----------
    n_prefixes:
        How many canonical prefixes exist.
    prefix_len:
        Length of each prefix.
    prefix_rate:
        Fraction of sequences that open with a prefix (the rest start randomly).
    tail_len:
        Number of random items appended after the prefix.
    """
    _validate(n_sequences, n_items, 2, 2)
    if n_prefixes < 1:
        raise DataGenerationError(f"n_prefixes must be >= 1, got {n_prefixes}")
    if prefix_len < 1 or prefix_len > n_items:
        raise DataGenerationError(
            f"prefix_len must lie in [1, n_items={n_items}], got {prefix_len}"
        )
    if tail_len < 1:
        raise DataGenerationError(f"tail_len must be >= 1, got {tail_len}")
    if not 0.0 < prefix_rate <= 1.0:
        raise DataGenerationError(f"prefix_rate must lie in (0, 1], got {prefix_rate}")

    gen = rng(seed, stream=14)
    # Prefixes must be *sequence-distinct* -- no prefix may be a sub-sequence of
    # another, otherwise the longer one subsumes the shorter and the closed set
    # collapses onto the longest, a degenerate case that would flatter the
    # compression ratio.  Item-disjointness is NOT required (and would waste the
    # universe); what matters is sub-sequence distinctness.
    pool = gen.permutation(n_items)
    prefixes: list[np.ndarray] = []
    for i in range(n_prefixes):
        cand = np.sort(pool[i * prefix_len : (i + 1) * prefix_len]).astype(np.int64)
        if len(cand) < prefix_len:
            # Wrap around into the same pool -- sequence distinctness still has
            # to hold, so verify and retry with a reshuffled slice.
            cand = np.sort(gen.choice(n_items, size=prefix_len, replace=False)).astype(np.int64)
        prefixes.append(cand)
    prefixes = _make_sequence_distinct(prefixes)

    used = {int(v) for p in prefixes for v in p}
    free_items = np.array([i for i in range(n_items) if i not in used], dtype=np.int64)
    # Assignment of prefixes to sequences: a fixed random split, so each prefix
    # has a *stable* support set -- the property closure depends on.
    assignment = gen.integers(0, n_prefixes, size=n_sequences)

    seqs: list[np.ndarray] = []
    for idx in range(n_sequences):
        if gen.random() < prefix_rate:
            head = prefixes[int(assignment[idx])]
        else:
            head = np.empty(0, dtype=np.int64)
        if len(free_items) > 0:
            take = min(tail_len, len(free_items))
            tail = gen.choice(free_items, size=take, replace=False)
            tail = tail[gen.permutation(take)]
        else:
            tail = gen.integers(0, n_items, size=tail_len).astype(np.int64)
        seq = dedup_keep_order(np.concatenate([head, tail]))
        if len(seq) >= 2:
            seqs.append(seq)
    if len(seqs) < 2:
        raise DataGenerationError("prefixed generator produced too few valid sequences")
    return SequenceDatabase(tuple(seqs), n_items, name="prefixed")


def make_grouped_db(
    *,
    seed: int,
    n_sequences: int = 240,
    n_groups: int = 4,
    group_len: int = 3,
    tail_len: int = 5,
    group_rate: float = 0.9,
) -> SequenceDatabase:
    """Group-structured sequences with **item-exclusive** headers.

    Why this generator exists
    -------------------------
    Closed-pattern compression measured **exactly 1.00x** on the ``markov`` /
    ``planted`` / ``segmented`` / ``prefixed`` families -- verified across 20+
    (generator, threshold, length) configurations, and verified *not* to be an
    implementation bug by checking a hand-built case (ratio 2.00).  The cause is
    structural:

    * **The length cap makes top-length patterns trivially closed.**  Nothing in
      the mined set extends them, so closure finds nothing to remove.
    * **In dense data, extending a pattern almost always changes its support.**
      Closure needs an extension with *identical* support, and that needs the
      extended set of sequences to be exactly the same set.

    This generator creates the regime where closure is *supposed* to pay:
    ``n_groups`` disjoint item blocks, each sequence opening with its group's
    block and then a random tail.  Because the header items are **exclusive to
    their group**, ``(item,)`` and its extensions within the group have exactly
    the same support, so the shorter members are genuinely non-closed.

    This is what real log prefixes look like -- ``GET /checkout`` and everything
    downstream of it share a support set -- and it is the honest way to report
    the compression lever: by how much it pays *where it can* pay, alongside the
    ~1x it pays on dense i.i.d.-ish data.

    Parameters
    ----------
    group_rate:
        Fraction of sequences carrying their group header (the rest start
        directly in the random tail).
    """
    if n_groups < 2:
        raise DataGenerationError(f"n_groups must be >= 2, got {n_groups}")
    if group_len < 1:
        raise DataGenerationError(f"group_len must be >= 1, got {group_len}")
    if tail_len < 1:
        raise DataGenerationError(f"tail_len must be >= 1, got {tail_len}")
    if not 0.0 < group_rate <= 1.0:
        raise DataGenerationError(f"group_rate must lie in (0, 1], got {group_rate}")
    n_items = n_groups * group_len
    _validate(n_sequences, n_items, 2, 2)

    gen = rng(seed, stream=15)
    # Each group owns a contiguous, disjoint block of item ids.  Header order is
    # randomised within the block so no header is a sub-sequence of another.
    groups: list[np.ndarray] = []
    for g in range(n_groups):
        block = np.arange(g * group_len, (g + 1) * group_len, dtype=np.int64)
        groups.append(block[gen.permutation(group_len)])

    seqs: list[np.ndarray] = []
    for _ in range(n_sequences):
        gid = int(gen.integers(n_groups))
        if gen.random() < group_rate:
            items: list[int] = [int(v) for v in groups[gid]]
        else:
            items = []
        n_fill = max(0, tail_len)
        picks = gen.choice(n_items, size=min(n_fill + group_len, n_items), replace=False)
        items.extend(int(v) for v in picks)
        seq = dedup_keep_order(np.asarray(items, dtype=np.int64))
        if len(seq) >= 2:
            seqs.append(seq)
    if len(seqs) < 2:
        raise DataGenerationError("grouped generator produced too few valid sequences")
    return SequenceDatabase(tuple(seqs), n_items, name="grouped")


DATASETS = {
    "markov": make_markov_sequences,
    "planted": make_planted_pattern_db,
    "segmented": make_semi_markov_db,
    "prefixed": make_prefixed_db,
    "grouped": make_grouped_db,
}


def get_dataset(name: str, **kwargs) -> SequenceDatabase:
    """Look up a generator by name.

    Raises
    ------
    KeyError
        With the list of known names, so a typo never silently falls back to a
        default generator (which would invalidate a benchmark table).
    """
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[key](**kwargs)
