from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from eval.report import (
    compare_named_reports,
    compare_reports,
    load_eval_csv,
    reorder_with_baseline,
    resolve_named_inputs,
    single_report,
)

DEFAULT_KS = [1, 5, 10, 100]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Eval report: geometry metrics + MRR + NDCG")
    ap.add_argument("--csv", action="append", default=[], help="Eval CSV path or run directory. Repeatable.")
    ap.add_argument("--label", action="append", default=[], help="Optional label for each --csv. Repeatable.")
    ap.add_argument("--baseline", type=str, default=None, help="For 2-run legacy mode: baseline eval_hits.csv or directory. For multi-run mode: baseline label.")
    ap.add_argument("--rerank", type=str, default=None, help="Legacy 2-run comparison: rerank eval_hits.csv or directory.")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="K values for Geom@K/NDCG@K")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save the output CSV.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    ks = list(args.ks)

    # Legacy 2-run mode preserved for compatibility.
    if args.baseline and args.rerank and not args.csv:
        dfb = load_eval_csv(Path(args.baseline))
        dfr = load_eval_csv(Path(args.rerank))
        out = compare_reports(dfb, dfr, ks=ks)
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(save_path)
            print(f"\nSaved report to: {save_path}")
        return

    # Single-run or multi-run modern mode.
    if args.csv:
        named = resolve_named_inputs(args.csv, args.label)
        named = reorder_with_baseline(named, args.baseline)

        if len(named) == 1:
            _, df = next(iter(named.items()))
            out = single_report(df, ks=ks)
        elif len(named) == 2 and not args.label and args.baseline is None:
            # Keep a familiar 2-run comparison shape when the user just passes two CSVs.
            dfs = list(named.values())
            out = compare_reports(dfs[0], dfs[1], ks=ks)
        else:
            out = compare_named_reports(named, ks=ks)

        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(save_path)
            print(f"\nSaved report to: {save_path}")
        return

    raise SystemExit("Provide either --csv (one or more), or both --baseline and --rerank")


if __name__ == "__main__":
    main()
