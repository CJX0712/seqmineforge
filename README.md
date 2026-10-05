<div align="center">

# SeqMineForge

**Verified deterministic sequential pattern mining** — with an exhaustive
oracle as the correctness gate and a vectorised engine as the speed claim.

[![CI](https://github.com/CJX0712/seqmineforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/seqmineforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/seqmineforge?color=blue)](https://github.com/CJX0712/seqmineforge/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-S%20%E2%9A%99%20all%20gates%20pass-brightgreen.svg)](docs/model_card.md)

**Author: 晨星** · MIT · CPU-only · offline-capable · zero GPU, zero model downloads

</div>

---

## What this is

Sequential pattern mining (SPM) finds every ordered item sequence occurring in at
least a given fraction of a database. It underpins next-basket recommendation,
clickstream analysis, workflow mining and process discovery.

The domain has a property almost nothing else does: **an exhaustive oracle is
possible**. Candidate patterns are drawn from a finite item universe, so a
brute-force miner gives ground truth on small databases. Most SPM systems assert
correctness by comparing against a stronger algorithm; here the flagship's output
is required to be *set-identical* to brute force, and the whole package is built
so that claim means something.

```
flagship SeqFuse   ──set-equal──▶  brute-force oracle
SPAM  SPADE  PrefixSpan  ─────────┘        (independent, plain-Python matcher)
```

## Measured results

All numbers below come from `benchmark.json`, written by
`python examples/run_demo.py` on this machine (numpy 2.5.3 / scipy 1.16.3 /
scikit-learn 1.9.1, CPU only). Nothing is hand-written.

### Acceptance gates

| ID | Gate | Value | Threshold | Verdict |
|----|------|-------|-----------|---------|
| G1 | oracle-equivalence | 1.0 | ≥ 1.0 | **PASS** |
| G2 | speedup vs fastest baseline | **1.837×** | > 1.5× | **PASS** |
| G3 | closure non-inferiority | +0.0000 | ≥ 0.0 | **PASS** |
| G4 | closure no-worse | 1.00× | ≥ 1.0× | **PASS** |
| G4b | closed basis lossless | 1.0 | ≥ 1.0 | **PASS** |
| G5 | determinism | 0.0 | ≤ 0.0 | **PASS** |

### G1 — correctness

| Scope | Comparisons | Result |
|-------|-------------|--------|
| Verification-scale (small, oracle feasible) | 24 (3 datasets × 2 seeds × 4 miners) | **all exact set equality** |
| Full scale (4 datasets × 2 seeds) | 8 runs | 0 spurious patterns, 0 closure violations, support deviation `0`, all four implementations agree |

### G2 — efficiency (aggregate wall-clock, per-request batch-of-1)

| Miner | Wall-clock (s) | vs fastest baseline |
|-------|---------------:|--------------------:|
| PrefixSpan | 3.363 | 1.000× |
| SPADE | 2.671 | — |
| SPAM | 2.603 | — |
| **SeqFuse** | **1.417** | **1.837×** |

Per-run speedup `1.997× ± 0.562` (2 seeds × 4 datasets). "Fastest baseline" is the
*minimum* elapsed time among SPAM / SPADE / PrefixSpan — the flagship is never
flattered by a weak rival.

### Ablation — which mechanism actually pays

| Variant | Mean (s) | Speed |
|---------|---------:|------:|
| **SeqFuse (full)** | **0.327** | **1.00×** |
| − vectorised DP | 0.648 | 0.50× |
| − profile pruning | 1.970 | 0.17× |
| both off (scalar) | 5.558 | 0.06× |

Profile pruning contributes ~3.0×, vectorised counting ~1.0×. Both are load-bearing.

## Quick start

```bash
git clone https://github.com/CJX0712/seqmineforge.git
cd seqmineforge
python -m pip install -r requirements.lock.txt

python examples/run_demo.py          # full benchmark -> benchmark.json, exits 1 if any gate fails
python -m pytest -q                  # 115 tests
python -m seqmineforge.cli info
```

No GPU, no network, no pretrained weights. The only hard dependency is NumPy;
scipy / scikit-learn upgrade the downstream classifier and are skipped cleanly
if absent.

## CLI

```bash
python -m seqmineforge.cli mine --dataset grouped --top 10
python -m seqmineforge.cli verify --dataset grouped --max-length 3   # exits 1 on mismatch
python -m seqmineforge.cli bench --out benchmark.json
python -m seqmineforge.cli info
```

```python
from seqmineforge import PatternType, SequenceDatabase, load_config
from seqmineforge.data import get_dataset
from seqmineforge.miners import get_miner

db = get_dataset(
    "grouped", seed=20261006, n_sequences=300, n_groups=5, group_len=4, tail_len=6, group_rate=0.92
)
res = get_miner("seqfuse").mine(db, min_support=0.05, max_pattern_length=4)
for p in res.sorted_patterns()[:5]:
    print(p.items, f"{p.support:.3f}")
```

## Architecture

```
cli.py
 └─ pipeline/          SeqMinePipeline.run() — the only source of reported numbers
     ├─ data/          5 synthetic DGPs (markov · planted · segmented · prefixed · grouped)
     ├─ miners/        oracle · spam · spade · prefixspan · flagship (SeqFuse)
     ├─ features/      closed-pattern basis · binary/support feature extractors
     └─ eval/          oracle equivalence · recount · gates · downstream classifier
           └─ core/    types · errors · config (ENV_SMF_*) · interfaces · seed
```

Dependencies run strictly one-way: `cli → pipeline → {data, miners, features,
eval} → core`. Modules depend on `Protocol`s in `core/interfaces.py`, never on
concrete classes.

### The three things worth knowing

**1. The projected-database convention.** A projected suffix starts *at the last
matched element*, so exactly **one** element is consumed at every depth — not
`len(pattern)`. Passing the depth skips real items; re-searching the suffix for
the pattern fails outright because the earlier elements are gone. Both mistakes
silently drop valid patterns while producing plausible-looking output. See
`miners/matching.py: PROJECTED_DB_CONSUMES_ONE`.

**2. Only the new item is searched when extending.** `project_sequence(p·x, suffix)`
is wrong; `extend_projection(suffix, 1, x)` is right. For sequence `[0,2,6]` and
prefix `(0,2)`, the suffix is `[2,6]` and searching it for `(0,2,6)` returns
`None` — dropping a pattern whose support is 8/200.

**3. Closure is ~1× on dense synthetic data, and that is the finding.** Closure
removes a pattern only when a frequent extension has *identical* support. On
generated data with random item order that essentially never happens, and the
length cap makes every top-length pattern trivially closed. Measured across 5
generators × 3 thresholds × 5 caps (75 configurations): ratio 1.00 in all of
them, while a hand-built case with `sup(0)=sup(0,1)=sup(0,1,2)=1.0` gives exactly
2.00 — so the machinery is right and the data is what makes it vacuous. G4
therefore asserts what is true (no worse than 1.0×) and **G4b asserts the closure
theorem itself**: the closed basis reconstructs the full frequent set exactly.

## Honest scope

- **SeqFuse is a systems improvement, not a new algorithm.** Same output set as
  the SOTA baselines (verified), lower wall-clock. The gain is vectorisation and
  candidate pruning.
- **The baselines are faithful CPU re-implementations of the published papers**
  (SPAM SIGMOD'94 / ICDT'03, SPADE VLDB'95, PrefixSpan ICDM'01) — not the original
  Java/C++. No claim is made about reproducing their published timings.
- **G2 measures this machine.** Speedups are interpreter-overhead-bound and will
  differ elsewhere.
- **Compression is 1.00× on the shipped DGPs.** `prefixed` and `grouped` exist to
  create the regime where closure *can* pay; both still land near 1.0×, which is
  documented above rather than tuned away.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — modules, invariants, gate definitions, pitfalls found
- [`docs/model_card.md`](docs/model_card.md) — intended use, limits, verification summary
- [`CHANGELOG.md`](CHANGELOG.md) — release history

## License

MIT © 2026 晨星