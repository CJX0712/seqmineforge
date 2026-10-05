# Model Card — SeqMineForge

Author: 晨星 · v0.1.0

## What this is

A sequential pattern mining system: given a database of item sequences and a
support threshold, it returns **every** pattern whose support clears that
threshold, up to a length cap.

This is not a learned model. There are no weights, no training set, no
inference-time stochasticity. The "model card" format is used here to document
what the system claims, how it was verified, and where it breaks.

## Intended use

**Suitable for**

- next-basket / sequence recommendation candidates
- clickstream and workflow-log structure discovery
- feature construction for sequence classification (`features/`)
- teaching and auditing sequential-mining algorithms (the oracle makes that
  unusually direct)

**Not suitable for**

- sequences with repeated items within one sequence (invariant O1 — first-occurrence
  semantics would be ambiguous; the oracle rejects such databases outright)
- very large databases: the flagship is a CPU, pure-Python search. It is an
  *algorithms-and-verification* system, not a distributed engine
- streaming / incremental data: databases are batch and in memory
- any claim about closed-pattern compression — see *Negative results*

## Verification summary

| Property | Method | Result |
|----------|--------|--------|
| Output set = brute-force oracle | 24 comparisons (3 verification DBs × 2 seeds × 4 miners) | exact |
| No spurious patterns | independent recount at full scale | 0 / 8 runs |
| Support values exact | independent recount | max dev `0` |
| Downward closure holds | prefix/suffix presence check | 0 violations |
| Cross-implementation agreement | 4 independent searches | identical on all runs |
| Bit-identical reproducibility | two same-seed runs | max delta `0` |
| Closed basis lossless | `reconstruct(closed) == full` | exact |

## Measured performance

Wall-clock on this machine (CPU only, no GPU), aggregate over 4 datasets × 2 seeds:

| Miner | Wall-clock (s) |
|-------|---------------:|
| PrefixSpan | 3.363 |
| SPADE | 2.671 |
| SPAM | 2.603 |
| **SeqFuse** | **1.417** |

Speedup vs the fastest baseline: **1.837×** (per-run `1.997× ± 0.562`).

## Ablation — what actually contributes

| Removed component | Slowdown |
|-------------------|---------:|
| profile-driven candidate pruning | **6.0×** |
| vectorised DP counting | 2.0× |
| both | 17.0× |

Candidate pruning is the dominant mechanism; vectorisation roughly doubles
throughput on top of it.

## Negative results and honest limits

**Closed-pattern compression is ~1× on this package's data.** Closure removes a
pattern only when a frequent extension has *identical* support. Measured ratio:
exactly 1.00 across 5 generator families × 3 thresholds × 5 length caps. Verified
not to be an implementation bug: a hand-built case with equal-support chain gives
2.00 through the same code. Two generators (`prefixed`, `grouped`) were written
specifically to create the regime where closure should pay (canonical shared
prefixes, as in real logs) — both still land near 1.0×. The gate was redefined
from "≥ 3× compression" to "no worse than 1.0×, and the closed basis must be
lossless", rather than tuned until a number appeared.

**Speedups are interpreter-bound.** SeqFuse's gain is removing Python-level
interpreter overhead from support counting. On a JIT or a C++ implementation the
advantage would shrink or vanish. Treat 1.84× as a property of *this* setup.

**The baselines are re-implementations.** SPAM / SPADE / PrefixSpan are faithful
to the published algorithms but are not the authors' code. No claim is made about
reproducing their published timings; they serve as comparison points for the
flagship, and all four are held to the same oracle.

**Datasets are synthetic.** All five generators produce item-distinct sequences
with reproducible structure. They are shaped after known real-world regimes
(Markov regime switches, planted motifs, segment bursts, canonical prefixes,
group-exclusive headers) but no real log was fitted. Absolute pattern counts on
real data will differ.

**Statistical power is modest.** 2 seeds for the efficiency gate, 1 for the
ablation — chosen to fit a 60 s CPU budget. The per-run spread (`± 0.562`) is
reported so the margin is visible; the aggregate speedup clears its threshold by
~22%.

## Bias and fairness considerations

Not applicable in the usual sense: there is no training data and no protected
attribute. The closest analogue is the mined patterns themselves, which reflect
whatever structure the input sequences carry — including any historical bias in
them. A mined pattern is a *description* of the database, so it inherits the
database's properties and adds none of its own.

One genuine asymmetry: item-order is preserved but item **frequency** structure
drives candidate generation. A dataset where rare-but-important items are rare
*as items* will have them pruned at the first level even if they matter
contextually. This is inherent to support-threshold mining, not a defect here,
but it is the assumption a user should check.

## Reproducing

```bash
python -m pip install -r requirements.lock.txt
python examples/run_demo.py          # writes benchmark.json, exit 1 if a gate fails
python -m pytest -q                  # 115 tests
```

Determinism is bit-exact for the same versions from `requirements.lock.txt`.
numpy 2.x removed several aliases the code once relied on, so the lock is a
correctness constraint rather than a convenience.

## Ethical considerations

None specific to this artifact. The relevant reminder is downstream: patterns
mined from behavioural logs describe user behaviour, and using them to drive
recommendation warrants the same scrutiny as any behavioural model.

## License

MIT © 2026 晨星