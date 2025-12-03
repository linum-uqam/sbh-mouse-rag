# eval/report.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, List

import pandas as pd

# Include 100; will be ignored if max rank < 100.
DEFAULT_KS = [1, 2, 5, 10, 100]


def _safe_available_ks(df: pd.DataFrame, ks: Iterable[int]) -> List[int]:
    """Return only those K that are <= max rank present in df."""
    max_rank = int(df["rank"].max()) if not df.empty else 0
    return [k for k in ks if k <= max_rank]


def summarize_topk_metrics(
    df: pd.DataFrame,
    ks: Iterable[int] = DEFAULT_KS,
) -> Dict[int, Dict[str, Any]]:
    """
    Per-rank metrics: look only at hits with rank == k.

    For each K and source ("allen", "real") we compute:

      - n_rows                      (total distinct rows across all sources)
      - score_mean_<src>            (mean score for rank == K)
      - spatial_dist_vox_mean_<src> (mean spatial distance at rank == K)
      - region_l1_error_mean_<src>  (mean region mismatch fraction [0,1])
    """
    out: Dict[int, Dict[str, Any]] = {}
    ks = _safe_available_ks(df, ks)

    for k in ks:
        dfk = df[df["rank"] == k]
        if dfk.empty:
            continue

        stats: Dict[str, Any] = {}
        # Total distinct rows across all sources
        stats["n_rows"] = int(dfk["row_idx"].nunique())

        # Per-source metrics
        for src in sorted(dfk["source"].unique()):
            dfks = dfk[dfk["source"] == src]
            if dfks.empty:
                continue

            prefix = src  # "allen" or "real"
            # Means; pandas .mean() ignores NaNs.
            score_mean = float(dfks["score"].mean())
            spatial_vox_mean = float(dfks["spatial_dist_vox"].mean())
            region_l1_mean = float(dfks["region_l1_error"].mean())

            stats[f"score_mean_{prefix}"] = score_mean
            stats[f"spatial_dist_vox_mean_{prefix}"] = spatial_vox_mean
            stats[f"region_l1_error_mean_{prefix}"] = region_l1_mean

        out[k] = stats

    return out


def summarize_oracle_topk(
    df: pd.DataFrame,
    ks: Iterable[int] = DEFAULT_KS,
    spatial_ranges: Dict[str, tuple[float, float]] | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Oracle@K metrics:

      For each row_idx and source, keep the best (minimum) error across ranks <= K,
      then average over rows.

    This answers:
      "If a human can pick any of the top-K from this source, how good can we get?"

    For each K and source we compute:

      - oracle_score_mean_<src>                  (max score among ranks <= K)
      - oracle_spatial_dist_vox_mean_<src>       (min spatial_dist_vox among ranks <= K)
      - oracle_region_l1_error_mean_<src>        (min region_l1_error among ranks <= K)
      - oracle_best_rank_spatial_mean_<src>      (avg rank of best spatial hit)
      - oracle_best_rank_region_mean_<src>       (avg rank of best region hit)
      - oracle_best_rank_mean_<src>              (alias for spatial version)

      Combined metric (spatial + region):

        spatial_norm = (spatial_dist_vox - spatial_min_src) / (spatial_max_src - spatial_min_src)
        combined_error = 0.5 * spatial_norm + 0.5 * region_l1_error

      And then:

      - oracle_combined_error_mean_<src>         (mean of min combined_error over ranks <= K)
      - oracle_best_rank_combined_mean_<src>     (avg rank of best-combined hit)
    """
    out: Dict[int, Dict[str, Any]] = {}
    ks = _safe_available_ks(df, ks)

    for k in ks:
        dsub = df[df["rank"] <= k]
        if dsub.empty:
            continue

        stats: Dict[str, Any] = {}

        for src in sorted(dsub["source"].unique()):
            dsrc = dsub[dsub["source"] == src]
            if dsrc.empty:
                continue

            prefix = src  # "allen" or "real"
            grp = dsrc.groupby("row_idx", sort=False)

            # Spatial normalization range for this source
            if spatial_ranges and prefix in spatial_ranges:
                spatial_min, spatial_max = spatial_ranges[prefix]
            else:
                spatial_min = float(dsrc["spatial_dist_vox"].min())
                spatial_max = float(dsrc["spatial_dist_vox"].max())
            spatial_scale = max(spatial_max - spatial_min, 1e-6)

            # Best errors (per row, for this source)
            best_spatial_vox = grp["spatial_dist_vox"].min()
            best_region_l1 = grp["region_l1_error"].min()
            best_score = grp["score"].max()  # score: higher is better

            # Rank of the best spatial hit (per row)
            idx_best_spatial = grp["spatial_dist_vox"].idxmin()
            best_ranks_spatial = dsrc.loc[idx_best_spatial, "rank"]

            # Rank of the best region hit (per row)
            idx_best_region = grp["region_l1_error"].idxmin()
            best_ranks_region = dsrc.loc[idx_best_region, "rank"]

            # Combined metric per row: 0.5 * spatial_norm + 0.5 * region_l1_error
            def _best_combined_idx(group: pd.DataFrame) -> int:
                spatial = group["spatial_dist_vox"].astype(float)
                spatial_norm = (spatial - spatial_min) / spatial_scale
                combined = 0.5 * spatial_norm + 0.5 * group["region_l1_error"].astype(float)
                return combined.idxmin()

            idx_best_combined = grp.apply(_best_combined_idx)
            best_ranks_combined = dsrc.loc[idx_best_combined, "rank"]

            # Also record the combined error value at that best rank
            # (use the same normalization as above)
            spatial_at_best = dsrc.loc[idx_best_combined, "spatial_dist_vox"].astype(float)
            region_at_best = dsrc.loc[idx_best_combined, "region_l1_error"].astype(float)
            spatial_norm_at_best = (spatial_at_best - spatial_min) / spatial_scale
            combined_at_best = 0.5 * spatial_norm_at_best + 0.5 * region_at_best

            stats[f"oracle_score_mean_{prefix}"] = float(best_score.mean())
            stats[f"oracle_spatial_dist_vox_mean_{prefix}"] = float(
                best_spatial_vox.mean()
            )
            stats[f"oracle_region_l1_error_mean_{prefix}"] = float(
                best_region_l1.mean()
            )

            # Best ranks (spatial, region, combined)
            stats[f"oracle_best_rank_spatial_mean_{prefix}"] = float(
                best_ranks_spatial.mean()
            )
            stats[f"oracle_best_rank_region_mean_{prefix}"] = float(
                best_ranks_region.mean()
            )
            # Combined metric stats
            stats[f"oracle_combined_error_mean_{prefix}"] = float(
                combined_at_best.mean()
            )
            stats[f"oracle_best_rank_combined_mean_{prefix}"] = float(
                best_ranks_combined.mean()
            )

            # Backward-compat alias (spatial-based)
            stats[f"oracle_best_rank_mean_{prefix}"] = float(
                best_ranks_spatial.mean()
            )

        out[k] = stats

    return out


def summarize_crop_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Error breakdown for full slices vs crops and crop-size buckets, on Top-1 hits.

    Uses the metrics:
      - spatial_dist_vox
      - spatial_dist_um
      - region_l1_error (and region_l1_error_pct)
    """
    df_top1 = df[df["rank"] == 1].copy()
    if df_top1.empty:
        return pd.DataFrame()

    # Precompute percent for readability
    df_top1["region_l1_error_pct"] = 100.0 * df_top1["region_l1_error"]

    out_blocks: List[pd.DataFrame] = []

    # Full slice vs crop
    if "q_is_crop" in df_top1.columns:
        by_crop = df_top1.groupby("q_is_crop").agg(
            spatial_dist_vox=("spatial_dist_vox", "mean"),
            spatial_dist_um=("spatial_dist_um", "mean"),
            region_l1_error=("region_l1_error", "mean"),
            region_l1_error_pct=("region_l1_error_pct", "mean"),
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
            region_l1_error=("region_l1_error", "mean"),
            region_l1_error_pct=("region_l1_error_pct", "mean"),
            score=("score", "mean"),
            count=("score", "size"),
        )
        out_blocks.append(by_bucket)

    if not out_blocks:
        return pd.DataFrame()
    return pd.concat(out_blocks, keys=["by_crop", "by_bucket"], names=["kind", "group"])


def _print_per_rank_with_oracle(
    per_rank: Dict[int, Dict[str, Any]],
    oracle: Dict[int, Dict[str, Any]],
    sources: List[str],
) -> None:
    """
    Pretty-print combined per-rank and oracle stats in one block per K.
    """
    print("\n=== Per-rank metrics with Oracle@K ===")
    for k in sorted(per_rank.keys()):
        stats_rank = per_rank[k]
        stats_oracle = oracle.get(k, {})

        print(f"\nRank {k}:")
        # n_rows across all sources at this rank
        n_rows = stats_rank.get("n_rows", None)
        if n_rows is not None:
            print(f"  {'n_rows':35s} = {n_rows}")

        for src in sources:
            prefix = src

            # Per-rank metrics
            score_mean = stats_rank.get(f"score_mean_{prefix}", None)
            spatial_mean = stats_rank.get(f"spatial_dist_vox_mean_{prefix}", None)
            region_mean = stats_rank.get(f"region_l1_error_mean_{prefix}", None)

            # Oracle metrics
            o_score_mean = stats_oracle.get(f"oracle_score_mean_{prefix}", None)
            o_spatial_mean = stats_oracle.get(
                f"oracle_spatial_dist_vox_mean_{prefix}", None
            )
            o_region_mean = stats_oracle.get(
                f"oracle_region_l1_error_mean_{prefix}", None
            )
            o_best_rank_spatial = stats_oracle.get(
                f"oracle_best_rank_spatial_mean_{prefix}", None
            )
            o_best_rank_region = stats_oracle.get(
                f"oracle_best_rank_region_mean_{prefix}", None
            )
            o_combined_mean = stats_oracle.get(
                f"oracle_combined_error_mean_{prefix}", None
            )
            o_best_rank_combined = stats_oracle.get(
                f"oracle_best_rank_combined_mean_{prefix}", None
            )

            # Only print this source if we have at least one metric
            if not any(
                v is not None
                for v in [
                    score_mean,
                    spatial_mean,
                    region_mean,
                    o_score_mean,
                    o_spatial_mean,
                    o_region_mean,
                    o_best_rank_spatial,
                    o_best_rank_region,
                    o_combined_mean,
                    o_best_rank_combined,
                ]
            ):
                continue

            print(f"  --- source = {prefix} ---")

            if score_mean is not None:
                print(
                    f"  {f'score_mean_{prefix}':35s} = {score_mean}"
                )
            if spatial_mean is not None:
                print(
                    f"  {f'spatial_dist_vox_mean_{prefix}':35s} = {spatial_mean}"
                )
            if region_mean is not None:
                print(
                    f"  {f'region_l1_error_mean_{prefix}':35s} = {region_mean}"
                )

            if o_score_mean is not None:
                print(
                    f"  {f'oracle_score_mean_{prefix}':35s} = {o_score_mean}"
                )
            if o_spatial_mean is not None:
                print(
                    f"  {f'oracle_spatial_dist_vox_mean_{prefix}':35s} = {o_spatial_mean}"
                )
            if o_region_mean is not None:
                print(
                    f"  {f'oracle_region_l1_error_mean_{prefix}':35s} = {o_region_mean}"
                )
            if o_best_rank_spatial is not None:
                print(
                    f"  {f'oracle_best_rank_spatial_mean_{prefix}':35s} = {o_best_rank_spatial}"
                )
            if o_best_rank_region is not None:
                print(
                    f"  {f'oracle_best_rank_region_mean_{prefix}':35s} = {o_best_rank_region}"
                )
            if o_combined_mean is not None:
                print(
                    f"  {f'oracle_combined_error_mean_{prefix}':35s} = {o_combined_mean}"
                )
            if o_best_rank_combined is not None:
                print(
                    f"  {f'oracle_best_rank_combined_mean_{prefix}':35s} = {o_best_rank_combined}"
                )


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

    # Backward-compat: if old column name exists, alias it.
    if "region_l1_error" not in df.columns and "region_mse" in df.columns:
        df["region_l1_error"] = df["region_mse"]

    print(f"Loaded eval hits from: {csv_path} (rows={len(df)})")

    sources = sorted(df["source"].unique())

    # Precompute spatial min/max per source for normalization of combined metric
    spatial_ranges = {}
    for src in sources:
        dsrc = df[df["source"] == src]
        if dsrc.empty:
            continue
        mn = float(dsrc["spatial_dist_vox"].min())
        mx = float(dsrc["spatial_dist_vox"].max())
        spatial_ranges[src] = (mn, mx)

    # ---- per-rank & oracle ----
    per_rank = summarize_topk_metrics(df, ks)
    oracle = summarize_oracle_topk(df, ks, spatial_ranges=spatial_ranges)
    _print_per_rank_with_oracle(per_rank, oracle, sources)

    # ---- crop buckets ----
    crop_summary = summarize_crop_buckets(df)
    if not crop_summary.empty:
        print("\n=== Crop / size breakdown (Top-1 only) ===")
        print(crop_summary)

    # ---- save summary CSV ----
    if save_summary:
        out_path = csv_path.with_name("eval_summary.csv")
        rows = []
        for k, stats in per_rank.items():
            rows.append({"mode": "per_rank", "K": k, **stats})
        for k, stats in oracle.items():
            rows.append({"mode": "oracle", "K": k, **stats})
        if rows:
            df_sum = pd.DataFrame(rows)
            df_sum.to_csv(out_path, index=False)
            print(f"\nSaved summary stats to: {out_path}")
