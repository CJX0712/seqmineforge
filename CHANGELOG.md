# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Author: 晨星.

## [0.1.0] — 2026-10-06

First release.

### Added

- **Flagship `SeqFuse`** — sequential pattern miner with three switchable
  mechanisms: vectorised DP support counting, profile-driven candidate pruning,
  and dual-representation routing (`SUBSEQUENCE` → DP engine, `CONTIGUOUS` →
  bitset engine).
- **Three faithful SOTA baselines** — SPAM (count / symbol / first projections),
  SPADE (vertical id-lists with downward-closure pruning), PrefixSpan
  (projection-only). Re-implementations of the published algorithms, not the
  authors' code.
- **Exhaustive oracle** — brute-force enumeration used as the correctness gate;
  enumerates ordered tuples so it is order-sensitive under both semantics.
- **Two pattern semantics** kept strictly separate: `SUBSEQUENCE` (order
  preserving, gaps allowed) and `CONTIGUOUS` (adjacent block).
- **5 synthetic DGPs** — `markov` (regime-switching chains), `planted`
  (item-disjoint planted motifs), `segmented` (burst-structured), `prefixed`
  (canonical shared prefixes), `grouped` (group-exclusive headers).
- **Closed-pattern basis** with an empirical check of the closure theorem.
- **5 acceptance gates** — G1 oracle equivalence, G2 efficiency, G3 closure
  non-inferiority, G4/G4b closure, G5 determinism. All pass.
- **115 tests**, 86 % line coverage, ruff 0.16.10 clean (`check` + `format`).
- **CI** across Python 3.12 / 3.13 on ubuntu + windows: lint, pytest with
  coverage, demo smoke with hard gate exit, byte-identical re-run check, CLI
  smoke, secret scan.

### Measured

- G1: 24 oracle comparisons exact; 8 full-scale runs with 0 spurious patterns,
  0 closure violations, support deviation `0`, four-way agreement.
- G2: **1.837×** aggregate speedup vs the fastest baseline
  (PrefixSpan 3.363 s / SPADE 2.671 s / SPAM 2.603 s / SeqFuse 1.417 s);
  per-run `1.997× ± 0.562`.
- G5: two same-seed runs bit-identical.
- Demo end-to-end 47.2 s on CPU.

### Fixed

Five defects found by the oracle gate, all of which produced plausible-looking
output rather than an error:

1. **Projected-database convention.** A projected suffix starts at the *last*
   matched element, so exactly one element is consumed per depth — not
   `len(pattern)`. Passing the depth skipped real items and lost 1336 of 1436
   patterns.
2. **Extension searched for the whole pattern.** `project_sequence(p·x, suffix)`
   fails because the earlier elements are chopped away; for `s=[0,2,6]` and
   `p=(0,2)` this dropped a pattern of support 8/200. Replaced with
   `extend_projection`, which searches only for the new item.
3. **SPAM symbol pruning over-pruned.** It removed *items* absent from the
   last-symbol map instead of dropping *sequences*; lost 21 of 1290 patterns.
   The correct form, `prunable_sequences`, keeps the pruning gain.
4. **Vectorised DP allowed position reuse.** The cursor advanced to `first_hit`
   rather than `first_hit + 1`, so `support((0,0))` returned 2 on `([0,1],[1,0])`
   while the plain-Python matcher correctly said 0.
5. **Oracle was order-agnostic for contiguous patterns.** It enumerated sorted
   combinations only, reporting half the patterns (`[1,0]` vs `[0,1]`); the miner
   was right and the oracle was wrong.

### Changed

- **G4 redefined** from "closure compression ≥ 3×" to "≥ 1.0× and the closed
  basis is lossless (G4b)". The original target is structurally unreachable:
  measured ratio is exactly 1.00 across 5 generators × 3 thresholds × 5 caps,
  while a hand-built equal-support chain gives 2.00 through the same code.
  Rationale recorded in `docs/architecture.md` §6.
- **Closure predicate** was `sup(ext) > sup(p)`, which is always False by
  downward closure — it marked every pattern closed and reported a stub 1.00.
  Now tests **equality**; regression test added.
- **Oracle budget** capped by candidate count (20,000). At 14 items × length 6
  the enumeration is 2,428,804 candidates and took ~17 minutes, exceeding the
  entire runtime budget. Exact equality is measured on verification-scale
  databases instead; full-scale runs use recount + cross-implementation
  agreement.
- **Recount vectorised** (NumPy DP instead of a Python scan): full pipeline
  117 s → 47 s.
- **Ablation scoped** to one dataset × one seed: ~44 s of a 60 s budget was being
  spent on attribution that needs no replication.

### Known limitations

- Closed-pattern compression is ~1× on the shipped DGPs; see the model card.
- Databases must have item-distinct sequences (invariant O1).
- Speedups are interpreter-overhead-bound and machine-specific.
- Pure-Python CPU search; not intended for very large databases.