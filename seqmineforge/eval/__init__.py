"""Evaluation: equivalence, gates, downstream utility."""

from __future__ import annotations

from .downstream import (
    DatasetSplits,
    available_sklearn,
    logistic_accuracy,
    make_binary_labels,
    stratified_split,
)
from .gates import (
    EquivalenceReport,
    GateReport,
    GateSpec,
    RecountReport,
    compare_to_oracle,
    evaluate_gates,
    gate_table,
    mean_std,
    oracle_candidate_count,
    recount_and_cross_check,
    significance_threshold,
)

__all__ = [
    "DatasetSplits",
    "EquivalenceReport",
    "GateReport",
    "GateSpec",
    "RecountReport",
    "available_sklearn",
    "compare_to_oracle",
    "evaluate_gates",
    "gate_table",
    "logistic_accuracy",
    "make_binary_labels",
    "mean_std",
    "oracle_candidate_count",
    "recount_and_cross_check",
    "significance_threshold",
    "stratified_split",
]
