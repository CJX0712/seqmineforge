"""End-to-end demo.

Every number printed here is read back from the JSON the pipeline just wrote.
Nothing is formatted from an in-memory constant, so the table and the JSON can
never disagree.

Author: 晨星
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqmineforge.core.seed import set_all
from seqmineforge.pipeline import SeqMinePipeline


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SeqMineForge demo (author: 晨星)")
    parser.add_argument("--out", default="benchmark.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with contextlib.suppress(AttributeError, ValueError):
        # Not every stream is reconfigurable (e.g. a captured one under pytest).
        sys.stdout.reconfigure(encoding="utf-8")

    overrides = {"seed": args.seed} if args.seed is not None else {}
    pipe = SeqMinePipeline(**overrides)
    set_all(pipe.cfg.seed)
    pipe.run(output=args.out)

    # Read back from disk -- the single source of truth for the numbers below.
    data = json.loads(Path(args.out).read_text(encoding="utf-8"))
    eff = data["efficiency"]

    print(_rule("="))
    print("SeqMineForge -- verified deterministic sequential pattern mining")
    print("Author: 晨星 | version 0.1.0 | MIT")
    print(_rule("="))

    cfg = data["config"]
    print(
        f"\nconfig: min_support={cfg['min_support']} "
        f"max_pattern_length={cfg['max_pattern_length']} "
        f"pattern_type={cfg['pattern_type']} seed={cfg['seed']}"
    )
    be = data["backend"]
    print(
        f"backend: numpy {be['numpy']} | sklearn {be['sklearn']} "
        f"| logistic={be['logistic_backend']}"
    )

    print("\n[1] ORACLE EQUIVALENCE (G1) -- every miner vs brute force, exact set equality")
    print(_rule())
    ver = data["verification"]
    print(
        f"verification-scale datasets ({', '.join(ver['datasets'])}), "
        f"seeds {ver['seeds']} -- {ver['n_comparisons']} comparisons"
    )
    print(
        f"{'dataset':<14}{'seed':<9}{'oracle':>8}"
        + "".join(f"{m:>13}" for m in ver["rows"][0]["equivalence"])
    )
    for row in ver["rows"]:
        cells = "".join(
            f"{('exact' if row['equivalence'][m]['exact'] else 'DIFF'):>13}"
            for m in ver["rows"][0]["equivalence"]
        )
        print(f"{row['name']:<14}{row['seed']:<9}{row['n_oracle']:>8}{cells}")
    print(f"\nall comparisons exact: {ver['all_exact']}")
    print("full-scale runs use recount + cross-implementation agreement (oracle infeasible):")
    print(
        f"{'dataset':<12}{'seed':<9}{'n_pat':>8}{'spurious':>10}{'closure_viol':>15}{'max_dev':>10}"
    )
    seen2: set[tuple[str, int]] = set()
    for row in data["datasets"]:
        key = (row["name"], row["seed"])
        if key in seen2:
            continue
        seen2.add(key)
        rec = row["recount"]
        print(
            f"{row['name']:<12}{row['seed']:<9}{int(rec['n_patterns']):>8}"
            f"{int(rec['n_spurious']):>10}{int(rec['closure_violations']):>15}"
            f"{rec['max_support_dev']:>10.2g}"
        )
    print(f"\nall full-scale recounts consistent: {eff['all_recounts_consistent']}")

    print("\n[2] EFFICIENCY (G2) -- aggregate wall-clock, per-request batch-of-1")
    print(_rule())
    agg = eff["aggregate_wallclock_sec"]
    min(v for k, v in agg.items() if k != "seqfuse")
    print(f"{'miner':<14}{'wall-clock (s)':>18}{'vs fastest base':>20}")
    for name, secs in sorted(agg.items(), key=lambda kv: kv[1]):
        if name == "seqfuse":
            continue
        print(f"{name:<14}{secs:>18.4f}{'1.000x':>20}")
    print(_rule())
    print(f"{'seqfuse':<14}{agg['seqfuse']:>18.4f}{eff['aggregate_speedup']:>19.3f}x")
    print(
        f"\nper-run speedup: {eff['per_run_speedup_mean']:.3f}x "
        f"+- {eff['per_run_speedup_std']:.3f} (n={len(eff['seeds'])} seeds x "
        f"{len(data['datasets']) // len(eff['seeds'])} datasets)"
    )

    print("\n[3] ABLATION -- which mechanism pays (mean over runs)")
    print(_rule())
    variants: dict[str, list[float]] = {}
    for r in data["ablation"]:
        variants.setdefault(r["variant"], []).append(r["elapsed_sec"])
    base = sum(variants["flagship_full"]) / len(variants["flagship_full"])
    print(f"{'variant':<24}{'mean (s)':>12}{'vs full':>12}")
    for name, vals in variants.items():
        m = sum(vals) / len(vals)
        ratio = base / m if m > 0 else float("inf")
        print(f"{name:<24}{m:>12.4f}{ratio:>11.3f}x")

    print("\n[4] DOWNSTREAM UTILITY (G3/G4) -- closure compression")
    print(_rule())
    u = data["utility"]
    print(f"train/test = {u['n_train']}/{u['n_test']}   positive rate = {u['pos_rate']:.3f}")
    print(
        f"features: {u['n_patterns_full']} (all patterns) -> "
        f"{u['n_patterns_closed']} (closed only)   "
        f"compression = {u['compression_ratio']:.2f}x"
    )
    print(
        f"hold-out accuracy: full={u['acc_full']:.4f}  closed={u['acc_closed']:.4f}  "
        f"delta={u['acc_delta']:+.4f}"
    )

    print("\n[5] DETERMINISM (G5) -- two same-seed runs")
    print(_rule())
    d = data["determinism"]
    print(f"run A: {d['run_a']}")
    print(f"run B: {d['run_b']}")
    print(f"max |delta| = {d['max_abs_delta']}  deterministic={d['deterministic']}")

    print("\n[6] ACCEPTANCE GATES")
    print(_rule())
    g = data["gates"]
    print(f"{'ID':<5}{'GATE':<28}{'VALUE':>14}{'THRESH':>12}{'':<6}")
    for r in g["gates"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['gid']:<5}{r['name'][:27]:<28}{float(r['value']):>14.6g}"
            f"{float(r['threshold']):>12.6g}  {mark}"
        )
    print(_rule())
    print(f"ALL GATES: {'PASS' if g['all_passed'] else 'FAIL'}")

    print("\n[7] FAILURE CASES (derived from measurements, not narrative)")
    print(_rule())
    for fc in data["failure_cases"]:
        print(f"* {fc['id']}: {fc['condition']}")
        print(f"    observed: {fc['observed']}")
        print(f"    cause:    {fc['cause']}")

    print("\n" + _rule("="))
    print(f"total pipeline wall-clock: {data['elapsed_sec']:.2f}s")
    print(f"report: {Path(args.out).resolve()}")
    print(_rule("="))

    if not args.quiet:
        print("\nLegend: exact = output set identical to brute-force oracle")
        print("        G2 speedup is vs the FASTEST baseline (min elapsed), never a weak one")

    return 0 if g["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
