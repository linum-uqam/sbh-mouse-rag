from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from eval.report import compare_reports, single_report

DEFAULT_KS = [1, 5, 10, 100]

def main() -> None:
    ap = argparse.ArgumentParser(description="Eval report: geometry metrics + MRR + NDCG")
    ap.add_argument("--csv", type=str, default=None, help="Single eval_hits.csv to summarize")
    ap.add_argument("--baseline", type=str, default=None, help="Baseline eval_hits.csv")
    ap.add_argument("--rerank", type=str, default=None, help="Rerank eval_hits.csv")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="K values for Geom@K/NDCG@K")
    args = ap.parse_args()

    ks = list(args.ks)

    if args.baseline and args.rerank:
        dfb = pd.read_csv(Path(args.baseline))
        dfr = pd.read_csv(Path(args.rerank))
        compare_reports(dfb, dfr, ks=ks)
        return

    if args.csv:
        df = pd.read_csv(Path(args.csv))
        single_report(df, ks=ks)
        return

    raise SystemExit("Provide either --csv or both --baseline and --rerank")


if __name__ == "__main__":
    main()