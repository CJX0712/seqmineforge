"""Command-line interface.

Author: 晨星
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from .core.config import ENV_PREFIX, load_config
from .core.errors import SeqMineError
from .core.types import PatternType, SequenceDatabase
from .data import get_dataset
from .miners import MINERS, get_miner
from .pipeline import DATASET_SPECS, SeqMinePipeline

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seqmineforge",
        description="Verified deterministic sequential pattern mining (author: 晨星).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- mine ---------------------------------------------------------
    p_mine = sub.add_parser("mine", help="mine patterns from a dataset")
    p_mine.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="planted")
    p_mine.add_argument("--miner", choices=sorted(MINERS), default="seqfuse")
    p_mine.add_argument("--seed", type=int, default=20261006)
    p_mine.add_argument("--min-support", type=float, default=None)
    p_mine.add_argument("--max-length", type=int, default=None)
    p_mine.add_argument("--top", type=int, default=10, help="how many patterns to print")
    p_mine.add_argument(
        "--pattern-type", choices=("subsequence", "contiguous"), default="subsequence"
    )

    # ---- verify -------------------------------------------------------
    p_ver = sub.add_parser("verify", help="check a miner against the exhaustive oracle (G1)")
    p_ver.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="planted")
    p_ver.add_argument("--miner", choices=sorted(MINERS), default="seqfuse")
    p_ver.add_argument("--seed", type=int, default=20261006)
    p_ver.add_argument("--min-support", type=float, default=None)
    p_ver.add_argument("--max-length", type=int, default=None)

    # ---- bench --------------------------------------------------------
    p_bench = sub.add_parser("bench", help="run the full benchmark and write JSON")
    p_bench.add_argument("--out", default="benchmark.json", help="output path for the JSON report")
    p_bench.add_argument("--seed", type=int, default=None, help="override the global seed")
    p_bench.add_argument("--min-support", type=float, default=None)
    p_bench.add_argument("--max-length", type=int, default=None)
    p_bench.add_argument("--no-oracle", action="store_true", help="skip the oracle check")

    # ---- info ---------------------------------------------------------
    sub.add_parser("info", help="print the effective configuration and registry")
    return parser


def _cfg(args: argparse.Namespace):
    return load_config(
        seed=getattr(args, "seed", None),
        min_support=getattr(args, "min_support", None),
        max_pattern_length=getattr(args, "max_length", None),
        pattern_type=getattr(args, "pattern_type", None),
    )


def _cmd_mine(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    spec = DATASET_SPECS[args.dataset]
    db = get_dataset(args.dataset, seed=cfg.seed, **spec)
    ptype = PatternType.SUBSEQUENCE if cfg.pattern_type == "subsequence" else PatternType.CONTIGUOUS
    res = get_miner(args.miner).mine(
        db,
        min_support=cfg.min_support,
        max_pattern_length=cfg.max_pattern_length,
        pattern_type=ptype,
    )
    print(
        f"dataset={args.dataset} miner={args.miner} seed={cfg.seed} "
        f"n_seq={db.n_sequences} min_support={cfg.min_support} "
        f"max_len={cfg.max_pattern_length}"
    )
    print(f"patterns={len(res)} elapsed={res.elapsed_sec:.4f}s")
    print("-" * 62)
    for pat in res.sorted_patterns()[: args.top]:
        items = ",".join(str(i) for i in pat.items)
        print(
            f"  [{items:<16}] support={pat.support:.4f}  ({pat.support * db.n_sequences:.0f} seqs)"
        )
    if len(res) > args.top:
        print(f"  ... {len(res) - args.top} more")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    spec = DATASET_SPECS[args.dataset]
    db = get_dataset(args.dataset, seed=cfg.seed, **spec)
    kw = {"min_support": cfg.min_support, "max_pattern_length": cfg.max_pattern_length}
    oracle = get_miner("oracle").mine(db, **kw)
    res = get_miner(args.miner).mine(db, **kw)

    from .eval import compare_to_oracle

    rep = compare_to_oracle(oracle, res)
    ok = rep.exact
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {args.dataset}/{args.miner} vs oracle")
    print(
        f"  oracle={rep.n_oracle} miner={rep.n_miner} missing={len(rep.missing)} "
        f"extra={len(rep.extra)} max_support_dev={rep.max_support_dev:.3g}"
    )
    if rep.missing:
        print(f"  missing e.g. {sorted(rep.missing)[:5]}")
    if rep.extra:
        print(f"  extra   e.g. {sorted(rep.extra)[:5]}")
    return 0 if ok else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    pipe = SeqMinePipeline(cfg)
    report = pipe.run(output=args.out)
    data = report.to_json()
    print("SeqMineForge benchmark")
    print("=" * 62)
    eff = data["efficiency"]
    print(f"datasets: {', '.join(sorted(DATASET_SPECS))}   seeds: {eff['seeds']}")
    print("aggregate wall-clock (sum over datasets of mean per-seed time):")
    for name, secs in sorted(eff["aggregate_wallclock_sec"].items(), key=lambda kv: kv[1]):
        print(f"  {name:<12} {secs:8.4f}s")
    print(f"aggregate speedup vs fastest baseline: {eff['aggregate_speedup']:.3f}x")
    print(f"all miners exact vs oracle: {eff['all_miners_exact_vs_oracle']}")
    print("-" * 62)
    print(pipe.gate_text(report))
    print("-" * 62)
    util = data["utility"]
    print(
        f"utility: acc_full={util['acc_full']:.4f} acc_closed={util['acc_closed']:.4f} "
        f"delta={util['acc_delta']:+.4f} compression={util['compression_ratio']:.2f}x "
        f"({util['n_patterns_full']} -> {util['n_patterns_closed']} features)"
    )
    print(f"determinism: max|delta| = {data['determinism']['max_abs_delta']}")
    print(f"report written to: {Path(args.out).resolve()}")
    return 0 if data["gates"]["all_passed"] else 1


def _cmd_info(args: argparse.Namespace) -> int:
    cfg = load_config()
    print("SeqMineForge configuration")
    print("-" * 62)
    for key, value in sorted(cfg.to_dict().items()):
        print(f"  {key:<26} {value}")
    print(f"  env prefix                  {ENV_PREFIX}")
    print("registered miners:")
    for name in sorted(MINERS):
        print(f"  - {name}")
    print("datasets:")
    for name in sorted(DATASET_SPECS):
        print(f"  - {name}: {DATASET_SPECS[name]}")
    return 0


_COMMANDS = {
    "mine": _cmd_mine,
    "verify": _cmd_verify,
    "bench": _cmd_bench,
    "info": _cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    with contextlib.suppress(AttributeError, ValueError):
        # Not every stream is reconfigurable (e.g. a captured one under pytest).
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except SeqMineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ``SequenceDatabase`` re-exported for tests that build databases inline.
__test_exports__ = (SequenceDatabase,)
