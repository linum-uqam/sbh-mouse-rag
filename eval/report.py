# eval/report.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, List

import pandas as pd


# Include 100; will be ignored if max rank < 100.
DEFAULT_KS = [1, 2, 5, 10, 100]


def _safe_available_ks(df: pd.DataFrame, ks: Iterable[int]) -> List[int]:
    max_rank = int(df["rank"].max()) if not df.empty else 0
    return [k for k in ks if k <= max_rank]


def summarize_topk_metrics(
    df: pd.DataFrame,
    ks: Iterable[int] = DEFAULT_KS,
) -> Dict[int, Dict[str, Any]]:
    """
    Per-rank metrics: look only at hits with rank == k.

    Metrics are split by source ("allen", "real"):

      n_rows_total, n_hits_total
      n_rows_allen, n_hits_allen, score_mean_allen, spatial_dist_vox_mean_allen, ...
      n_rows_real,  n_hits_real,  score_mean_real,  spatial_dist_vox_mean_real,  ...
    """
    out: Dict[int, Dict[str, Any]] = {}
    ks = _safe_available_ks(df, ks)

    for k in ks:
        dfk = df[df["rank"] == k]
        if dfk.empty:
            continue

        stats: Dict[str, Any] = {}

        # Totals across all sources
        stats["n_rows"] = int(dfk["row_idx"].nunique())
        stats["n_hits"] = int(len(dfk))

        # Per-source metrics
        for src in sorted(dfk["source"].unique()):
            dfks = dfk[dfk["source"] == src]
            if dfks.empty:
                continue

            prefix = src  # "allen" or "real"
            stats[f"n_rows_{prefix}"] = int(dfks["row_idx"].nunique())
            stats[f"n_hits_{prefix}"] = int(len(dfks))

            # Means; pandas .mean() ignores NaNs.
            score_mean = float(dfks["score"].mean())
            spatial_vox_mean = float(dfks["spatial_dist_vox"].mean())
            spatial_um_mean = float(dfks["spatial_dist_um"].mean())
            region_mse_mean = float(dfks["region_mse"].mean())
            region_mse_mean_pct = 100.0 * region_mse_mean

            stats[f"score_mean_{prefix}"] = score_mean
            stats[f"spatial_dist_vox_mean_{prefix}"] = spatial_vox_mean
            stats[f"spatial_dist_um_mean_{prefix}"] = spatial_um_mean
            stats[f"region_mse_mean_{prefix}"] = region_mse_mean
            stats[f"region_mse_mean_pct_{prefix}"] = region_mse_mean_pct

        out[k] = stats

    return out


def summarize_oracle_topk(
    df: pd.DataFrame,
    ks: Iterable[int] = DEFAULT_KS,
) -> Dict[int, Dict[str, Any]]:
    """
    Oracle@K metrics:

      For each row_idx and source, keep the best (minimum) error across ranks <= K,
      then average over rows.

    This answers:
      "If a human can pick any of the top-K from this source, how good can we get?"

    For each K and source we compute:

      - n_rows_<src>
      - score_best_mean_<src>            (max score among ranks <= K)
      - spatial_dist_vox_best_mean_<src> (min spatial_dist_vox among ranks <= K)
      - spatial_dist_um_best_mean_<src>
      - region_mse_best_mean_<src>
      - best_rank_mean_<src>             (avg rank position of best spatial hit)
    """
    out: Dict[int, Dict[str, Any]] = {}
    ks = _safe_available_ks(df, ks)

    for k in ks:
        dsub = df[df["rank"] <= k]
        if dsub.empty:
            continue

        stats: Dict[str, Any] = {}
        # Total distinct rows across all sources (for reference)
        stats["n_rows"] = int(dsub["row_idx"].nunique())

        for src in sorted(dsub["source"].unique()):
            dsrc = dsub[dsub["source"] == src]
            if dsrc.empty:
                continue

            prefix = src  # "allen" or "real"
            grp = dsrc.groupby("row_idx", sort=False)

            # Best errors (per row, for this source)
            best_spatial_vox = grp["spatial_dist_vox"].min()
            best_spatial_um = grp["spatial_dist_um"].min()
            best_region = grp["region_mse"].min()
            best_score = grp["score"].max()  # score: higher is better

            # Rank of the best spatial hit (per row)
            idx_best = grp["spatial_dist_vox"].idxmin()
            best_ranks = dsrc.loc[idx_best, "rank"]

            stats[f"n_rows_{prefix}"] = int(len(best_spatial_vox))
            stats[f"score_best_mean_{prefix}"] = float(best_score.mean())
            stats[f"spatial_dist_vox_best_mean_{prefix}"] = float(best_spatial_vox.mean())
            stats[f"spatial_dist_um_best_mean_{prefix}"] = float(best_spatial_um.mean())
            stats[f"region_mse_best_mean_{prefix}"] = float(best_region.mean())
            stats[f"region_mse_best_mean_pct_{prefix}"] = 100.0 * float(best_region.mean())
            stats[f"best_rank_mean_{prefix}"] = float(best_ranks.mean())

        out[k] = stats

    return out


def summarize_crop_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Error breakdown for full slices vs crops and crop-size buckets, on Top-1 hits.

    Uses the new metrics:
      - spatial_dist_vox
      - spatial_dist_um
      - region_mse (and region_mse_pct)
    """
    df_top1 = df[df["rank"] == 1].copy()
    if df_top1.empty:
        return pd.DataFrame()

    # Precompute percent for readability
    df_top1["region_mse_pct"] = 100.0 * df_top1["region_mse"]

    out_blocks: List[pd.DataFrame] = []

    # Full slice vs crop
    if "q_is_crop" in df_top1.columns:
        by_crop = df_top1.groupby("q_is_crop").agg(
            spatial_dist_vox=("spatial_dist_vox", "mean"),
            spatial_dist_um=("spatial_dist_um", "mean"),
            region_mse=("region_mse", "mean"),
            region_mse_pct=("region_mse_pct", "mean"),
            score=("score", "mean"),
            count=("score", "size"),
        )
        by_crop.index = by_crop.index.map(lambda x: "crop" if x else "full_slice")
        out_blocks.append(by_crop)

    # Crop-size buckets (area fraction)
    if "q_crop_area_frac" in df_top1.columns:
        df_top1 = df_top1.copy()
        df_top1["crop_bucket"] = pd.cut(
            df_top1["q_crop_area_frac"].fillna(1.0),
            bins=[0.0, 0.1, 0.25, 0.5, 0.75, 1.01],
            labels=["<=10%", "10–25%", "25–50%", "50–75%", ">75%"],
        )
        by_bucket = df_top1.groupby("crop_bucket").agg(
            spatial_dist_vox=("spatial_dist_vox", "mean"),
            spatial_dist_um=("spatial_dist_um", "mean"),
            region_mse=("region_mse", "mean"),
            region_mse_pct=("region_mse_pct", "mean"),
            score=("score", "mean"),
            count=("score", "size"),
        )
        out_blocks.append(by_bucket)

    if not out_blocks:
        return pd.DataFrame()
    return pd.concat(out_blocks, keys=["by_crop", "by_bucket"], names=["kind", "group"])


def run_report(
    csv_path: str | Path,
    ks: Iterable[int] = DEFAULT_KS,
    save_summary: bool = True,
) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"No rows in {csv_path}")
        return

    print(f"Loaded eval hits from: {csv_path} (rows={len(df)})")

    # ---- per-rank ----
    per_rank = summarize_topk_metrics(df, ks)
    print("\n=== Per-rank metrics (rank == k) ===")
    for k, stats in per_rank.items():
        print(f"\nRank {k}:")
        for k2, v in stats.items():
            print(f"  {k2:35s} = {v}")

    # ---- oracle@k ----
    oracle = summarize_oracle_topk(df, ks)
    print("\n=== Oracle@K metrics (best among ranks <= K) ===")
    for k, stats in oracle.items():
        print(f"\nK = {k}:")
        for k2, v in stats.items():
            print(f"  {k2:35s} = {v}")

    # ---- crop buckets ----
    crop_summary = summarize_crop_buckets(df)
    if not crop_summary.empty:
        print("\n=== Crop / size breakdown (Top-1 only) ===")
        print(crop_summary)

    if save_summary:
        out_path = csv_path.with_name("eval_summary.csv")
        # Flatten per-rank + oracle into a single DataFrame
        rows = []
        for k, stats in per_rank.items():
            rows.append({"mode": "per_rank", "K": k, **stats})
        for k, stats in oracle.items():
            rows.append({"mode": "oracle", "K": k, **stats})
        if rows:
            df_sum = pd.DataFrame(rows)
            df_sum.to_csv(out_path, index=False)
            print(f"\nSaved summary stats to: {out_path}")
