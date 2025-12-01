# scripts/run_report.py
from __future__ import annotations

import argparse
from pathlib import Path

from eval.report import run_report, DEFAULT_KS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse eval_hits.csv and compute top-k metrics."
    )
    p.add_argument(
        "--csv",
        type=str,
        default="eval_out/eval_hits.csv",
        help="Path to eval_hits.csv produced by run_eval.",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=DEFAULT_KS,
        help=f"Top-K values to analyse (default: {DEFAULT_KS}).",
    )
    p.add_argument(
        "--no-save-summary",
        action="store_true",
        default=False,
        help="Do not write eval_summary.csv alongside eval_hits.csv.",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_report(
        csv_path=Path(a.csv),
        ks=a.ks,
        save_summary=not a.no_save_summary,
    )


if __name__ == "__main__":
    main()
