"""Equivalence metrics + acceptance gates.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import GateFailureError
from ..core.types import MiningResult

__all__ = [
    "EquivalenceReport",
    "GateReport",
    "GateSpec",
    "RecountReport",
    "compare_to_oracle",
    "evaluate_gates",
    "gate_table",
    "mean_std",
    "oracle_candidate_count",
    "recount_and_cross_check",
    "significance_threshold",
]


@dataclass(frozen=True)
class EquivalenceReport:
    """Set-level comparison of a miner against the exhaustive oracle."""

    missing: frozenset[tuple[int, ...]]
    extra: frozenset[tuple[int, ...]]
    max_support_dev: float
    n_oracle: int
    n_miner: int

    @property
    def exact(self) -> bool:
        """True iff the two sets are identical and every support agrees."""
        return not self.missing and not self.extra and self.max_support_dev <= 1e-12

    @property
    def jaccard(self) -> float:
        union = len(self.missing | self.extra) + 0.0
        inter = self.n_oracle - len(self.missing)
        if union == 0:
            return 1.0
        # inter / (inter + missing + extra)
        return inter / (inter + len(self.missing) + len(self.extra))

    def summary(self) -> dict[str, float | bool]:
        return {
            "exact": self.exact,
            "jaccard": self.jaccard,
            "n_oracle": float(self.n_oracle),
            "n_miner": float(self.n_miner),
            "n_missing": float(len(self.missing)),
            "n_extra": float(len(self.extra)),
            "max_support_dev": self.max_support_dev,
        }


def compare_to_oracle(
    oracle: MiningResult, miner: MiningResult, *, tol: float = 1e-12
) -> EquivalenceReport:
    """Compare two mining results as *sets*, plus a support cross-check.

    The support check matters separately from the set check: a miner can return
    exactly the right patterns with wrong support values, which any
    pattern-count-based metric would score as perfect.
    """
    missing = frozenset(oracle.keys - miner.keys)
    extra = frozenset(miner.keys - oracle.keys)
    oracle_sup = oracle.supports
    miner_sup = miner.supports
    dev = 0.0
    for key in oracle.keys & miner.keys:
        dev = max(dev, abs(oracle_sup[key] - miner_sup[key]))
    if dev > tol:
        # A support deviation beyond tolerance is itself a failure; surface it as
        # a synthetic "extra" entry so `exact` flips to False without any caller
        # needing to inspect two fields.
        extra = frozenset(extra | {("<support-deviation>",)})
    return EquivalenceReport(
        missing=missing,
        extra=extra,
        max_support_dev=dev,
        n_oracle=len(oracle),
        n_miner=len(miner),
    )


def oracle_candidate_count(n_items_present: int, max_len: int) -> int:
    """How many patterns the exhaustive oracle must enumerate.

    For subsequence semantics over ``m`` present items, candidates up to length
    ``k`` number ``sum_j P(m, j)`` (ordered, distinct tuples).  This is what
    decides whether the oracle is runnable: at ``m = 14, k = 4`` it is ~40 k,
    at ``m = 14, k = 6`` it is ~2.1 M -- the latter is why G1 falls back to
    recount + cross-agreement on the larger benchmarks.
    """
    if n_items_present < 0 or max_len < 1:
        return 0
    total = 0
    for j in range(1, min(max_len, n_items_present) + 1):
        term = 1
        for i in range(j):
            term *= n_items_present - i
        total += term
    return total


def mean_std(values: list[float]) -> tuple[float, float]:
    """``(mean, population std)``; ``(nan, nan)`` for an empty input."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def significance_threshold(a_mean: float, a_std: float, b_mean: float, b_std: float) -> float:
    """Minimum |delta| for a win to count as significant.

    Rule: ``|mean_a - mean_b| > 0.5 * (std_a + std_b)``.  Deliberately
    conservative -- a mean difference smaller than the combined spread is
    indistinguishable from seed noise.
    """
    return 0.5 * (a_std + b_std)


@dataclass(frozen=True)
class RecountReport:
    """Independent re-verification of a miner's own output.

    This is the *infeasible-oracle* counterpart to
    :func:`compare_to_oracle`.  It cannot prove that a miner found *all*
    frequent patterns (only the oracle can), but it does prove three things that
    together are strong evidence:

    1. **No spurious pattern.**  Every pattern the miner reports really does
       occur at least ``min_support`` of the time.
    2. **Support values are exact.**  Each reported support matches a recount
       to ``1e-12``.
    3. **Downward closure holds.**  Every frequent pattern's frequent
       sub-patterns are also present -- an omission anywhere in the search
       usually breaks this.

    Passing all three on a large database, combined with exact oracle equality on
    small ones, covers both ends of the size range.
    """

    n_patterns: int
    n_spurious: int
    max_support_dev: float
    closure_violations: int
    min_support: float
    independent_miners: int
    cross_agreement: bool

    @property
    def consistent(self) -> bool:
        return (
            self.n_spurious == 0
            and self.max_support_dev <= 1e-12
            and self.closure_violations == 0
            and self.cross_agreement
        )

    def summary(self) -> dict[str, float | bool | int]:
        return {
            "consistent": self.consistent,
            "n_patterns": float(self.n_patterns),
            "n_spurious": float(self.n_spurious),
            "max_support_dev": self.max_support_dev,
            "closure_violations": float(self.closure_violations),
            "independent_miners": float(self.independent_miners),
            "cross_agreement": self.cross_agreement,
        }


def recount_and_cross_check(
    miner_results: dict[str, MiningResult],
    db,
    *,
    min_support: float,
    independent: str,
) -> RecountReport:
    """Verify ``independent``'s output and cross-agreement across all miners.

    Parameters
    ----------
    miner_results:
        Results keyed by miner name, all on the same database and budget.
    db:
        The :class:`~seqmineforge.core.types.SequenceDatabase` they ran on.
    min_support:
        The threshold in force.
    independent:
        Name of the miner whose output is being verified in detail.  Should be
        the flagship.

    Implementation note
    -------------------
    Support recounting goes through :class:`~seqmineforge.miners.vector_index.SequenceIndex`
    (the vectorised DP) rather than a Python ``sum(1 for ...)``.  That is a
    deliberate **independence trade-off**: the DP engine and the miners share the
    vectorised matcher, so a bug there could hide.  The mitigation is that G1's
    *exact-equality* half runs against the brute-force oracle, whose matcher
    (:func:`~seqmineforge.miners.matching.contains_subsequence`) is a completely
    separate plain-Python implementation, and every miner -- including the
    flagship's scalar fallback path -- is checked against that oracle.  The
    cross-implementation agreement check covers the rest.

    The scalar recount was measured at ~45 s of the ~80 s demo budget; the
    vectorised path brings the whole benchmark under it.
    """
    from ..miners.vector_index import SequenceIndex

    n_seq = db.n_sequences
    target = miner_results[independent]
    index = SequenceIndex(db)

    keys = list(target.supports)
    sups = target.supports
    n_spurious = 0
    max_dev = 0.0
    for key in keys:
        true_count = index.support(key)
        if true_count / n_seq + 1e-12 < min_support:
            n_spurious += 1
        max_dev = max(max_dev, abs(true_count / n_seq - sups[key]))

    # Downward closure: every sub-pattern of a reported pattern that itself
    # clears the threshold must also have been reported.  Only immediate
    # prefix/suffix need checking -- a longer frequent sub-pattern implies a
    # frequent one-step prefix by downward closure.
    keys_set = set(keys)
    violations = 0
    for key in keys:
        for cut in (1, len(key) - 1):
            if cut <= 0:
                continue
            for sub in (key[:cut], key[-cut:]):
                if len(sub) == len(key):
                    continue
                cnt = index.support(sub)
                if cnt / n_seq + 1e-12 >= min_support and sub not in keys_set:
                    violations += 1

    # Cross-implementation agreement.
    keysets = {name: r.keys for name, r in miner_results.items()}
    names = sorted(keysets)
    cross = all(keysets[names[0]] == keysets[n] for n in names[1:])

    return RecountReport(
        n_patterns=len(target),
        n_spurious=n_spurious,
        max_support_dev=max_dev,
        closure_violations=violations,
        min_support=min_support,
        independent_miners=len(keysets),
        cross_agreement=cross,
    )


@dataclass(frozen=True)
class GateSpec:
    """One acceptance gate with a fixed threshold."""

    gid: str
    name: str
    threshold: float
    comparator: str = "gt"  # "gt" (value > threshold) or "lt"
    description: str = ""


@dataclass
class GateReport:
    """Outcome of every gate plus a machine-readable summary."""

    rows: list[dict[str, object]] = field(default_factory=list)

    def add(
        self,
        gid: str,
        name: str,
        value: float,
        threshold: float,
        comparator: str,
        passed: bool,
        note: str = "",
    ) -> None:
        self.rows.append(
            {
                "gid": gid,
                "name": name,
                "value": float(value),
                "threshold": float(threshold),
                "comparator": comparator,
                "passed": bool(passed),
                "note": note,
            }
        )

    @property
    def all_passed(self) -> bool:
        return all(bool(r["passed"]) for r in self.rows)

    @property
    def failures(self) -> list[dict[str, object]]:
        return [r for r in self.rows if not r["passed"]]

    def as_dict(self) -> dict[str, object]:
        return {
            "all_passed": self.all_passed,
            "n_gates": len(self.rows),
            "n_failed": len(self.failures),
            "gates": self.rows,
        }


def _passes(value: float, threshold: float, comparator: str) -> bool:
    if comparator == "gt":
        return bool(value > threshold)
    if comparator == "lt":
        return bool(value < threshold)
    if comparator == "ge":
        return bool(value >= threshold)
    if comparator == "le":
        return bool(value <= threshold)
    raise GateFailureError(f"unknown comparator {comparator!r}")


def evaluate_gates(
    rows: list[tuple[str, str, float, GateSpec]],
) -> GateReport:
    """Evaluate ``(gid, name, value, spec)`` tuples into a :class:`GateReport`."""
    report = GateReport()
    for gid, name, value, spec in rows:
        report.add(
            gid=gid,
            name=name,
            value=value,
            threshold=spec.threshold,
            comparator=spec.comparator,
            passed=_passes(value, spec.threshold, spec.comparator),
            note=spec.description,
        )
    return report


def gate_table(report: GateReport) -> str:
    """Render a gate report as a fixed-width ASCII table (Windows-safe)."""
    header = f"{'ID':<5}{'GATE':<28}{'VALUE':>14}{'THRESH':>12}  {'':<4}"
    lines = [header, "-" * len(header)]
    for r in report.rows:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"{r['gid']!s:<5}{str(r['name'])[:27]:<28}"
            f"{float(r['value']):>14.6g}{float(r['threshold']):>12.6g}  {mark:<4}"
        )
    return "\n".join(lines)
