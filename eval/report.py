from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd


# ----------------------------
# Core per-query extraction
# ----------------------------

def _iter_queries(df: pd.DataFrame) -> Iterable[Tuple[Tuple[int, str], pd.DataFrame]]:
    """
    Yields ((row_idx, source), dfq_sorted_by_rank)
    """
    g = df.groupby(["row_idx", "source"], sort=False)
    for key, dfq in g:
        dfq = dfq.sort_values("rank", ascending=True)
        yield (int(key[0]), str(key[1])), dfq


def _prefix_min(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    cur = np.inf
    for i, v in enumerate(x):
        if np.isfinite(v) and v < cur:
            cur = v
        out[i] = cur
    return out


def _ndcg_from_dist(d: np.ndarray, tau: float, K: int) -> float:
    """
    NDCG@K with relevance rel = exp(-d/tau).
    Uses ideal ordering = ascending distance (descending rel).

    Returns NaN if no finite distances.
    """
    if d.size == 0:
        return float("nan")

    finite = np.isfinite(d)
    if not finite.any():
        return float("nan")

    tau = float(max(tau, 1e-6))
    dK = d[:K]
    finiteK = np.isfinite(dK)
    if not finiteK.any():
        return float("nan")

    rel = np.zeros_like(dK, dtype=np.float64)
    rel[finiteK] = np.exp(-dK[finiteK] / tau)

    discounts = 1.0 / np.log2(np.arange(2, 2 + dK.size))
    dcg = float((rel * discounts).sum())

    # sort by distance ascending within available candidates
    d_sorted = np.sort(d[np.isfinite(d)])
    d_sorted = d_sorted[:K]
    rel_id = np.exp(-d_sorted / tau)
    discounts_id = 1.0 / np.log2(np.arange(2, 2 + d_sorted.size))
    idcg = float((rel_id * discounts_id).sum())

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def compute_metrics(
    df: pd.DataFrame,
    ks: List[int],
    thresholds: List[float],
) -> pd.DataFrame:
    """
    Returns a tidy table with metrics aggregated over queries,
    computed separately for each source and overall.
    """
    rows = []

    # Ensure needed cols
    required = {"row_idx", "source", "rank", "geom_dist_vox"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_tau = "gt_tau_vox" in df.columns
    has_crop = "q_is_crop" in df.columns

    def run_block(df_block: pd.DataFrame, label: str) -> None:
        per_query = []

        for (row_idx, src), dfq in _iter_queries(df_block):
            d = dfq["geom_dist_vox"].to_numpy(dtype=np.float64)

            # K-limited prefix min for Geom@K
            pm = _prefix_min(d)

            # best geometry rank (over full list)
            finite = np.isfinite(d)
            if finite.any():
                best_rank = int(np.argmin(np.where(finite, d, np.inf)) + 1)
            else:
                best_rank = None

            tau_q = float(dfq["gt_tau_vox"].iloc[0]) if has_tau else float(np.nan)

            per_query.append((src, d, pm, best_rank, tau_q))

        if not per_query:
            return

        # Aggregate overall and per-source
        for src_name in sorted({x[0] for x in per_query} | {"ALL"}):
            subset = [x for x in per_query if (src_name == "ALL" or x[0] == src_name)]
            if not subset:
                continue

            n_q = len(subset)

            # Geom@K
            for K in ks:
                geomK = []
                for _, d, pm, _, _ in subset:
                    kk = min(K, pm.size)
                    v = pm[kk - 1] if kk >= 1 else np.nan
                    geomK.append(v)
                geomK = np.asarray(geomK, dtype=np.float64)

                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"Geom@{K}_mean",
                    "value": float(np.nanmean(geomK)),
                    "n_queries": n_q,
                })
                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"Geom@{K}_median",
                    "value": float(np.nanmedian(geomK)),
                    "n_queries": n_q,
                })

            # RankBestGeom
            ranks = np.asarray([r if r is not None else np.nan for *_, r, __ in subset], dtype=np.float64)
            rows.append({
                "group": label,
                "source": src_name,
                "metric": "RankBestGeom_mean",
                "value": float(np.nanmean(ranks)),
                "n_queries": n_q,
            })
            rows.append({
                "group": label,
                "source": src_name,
                "metric": "RankBestGeom_median",
                "value": float(np.nanmedian(ranks)),
                "n_queries": n_q,
            })

            # SR@K(thr) and MRR_geom(thr)
            for thr in thresholds:
                thr = float(thr)

                # SR@K at K=10 and K=100 are the usual; keep both
                for K in [10, 100]:
                    sr = []
                    for _, d, _, _, _ in subset:
                        kk = min(K, d.size)
                        if kk <= 0:
                            sr.append(0.0)
                            continue
                        dd = d[:kk]
                        ok = np.isfinite(dd) & (dd <= thr)
                        sr.append(1.0 if ok.any() else 0.0)
                    rows.append({
                        "group": label,
                        "source": src_name,
                        "metric": f"SR@{K}(thr={thr:.2f})",
                        "value": float(np.mean(sr)) if sr else float("nan"),
                        "n_queries": n_q,
                    })

                # MRR over full list (cap at 100)
                mrr = []
                for _, d, _, _, _ in subset:
                    kk = min(100, d.size)
                    if kk <= 0:
                        mrr.append(0.0)
                        continue
                    dd = d[:kk]
                    ok = np.isfinite(dd) & (dd <= thr)
                    if not ok.any():
                        mrr.append(0.0)
                    else:
                        r = int(np.argmax(ok) + 1)  # first True
                        mrr.append(1.0 / float(r))
                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"MRR_geom(thr={thr:.2f})",
                    "value": float(np.mean(mrr)) if mrr else float("nan"),
                    "n_queries": n_q,
                })

            # NDCG@K (use tau_query if present, else use global scale=median Geom@1)
            for K in [10, 100]:
                ndcgs = []
                for _, d, _, __, tau_q in subset:
                    if not np.isfinite(tau_q):
                        # fallback scale: median finite distance in this query list
                        finite = d[np.isfinite(d)]
                        tau_use = float(np.median(finite)) if finite.size else float("nan")
                    else:
                        tau_use = tau_q

                    ndcgs.append(_ndcg_from_dist(d, tau=tau_use, K=min(K, d.size)))

                ndcgs = np.asarray(ndcgs, dtype=np.float64)
                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"NDCG@{K}_mean",
                    "value": float(np.nanmean(ndcgs)),
                    "n_queries": n_q,
                })
                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"NDCG@{K}_median",
                    "value": float(np.nanmedian(ndcgs)),
                    "n_queries": n_q,
                })

    # main blocks: overall + (optional) crop vs full
    run_block(df, "ALL")

    if has_crop:
        run_block(df[df["q_is_crop"] == False], "FULL")
        run_block(df[df["q_is_crop"] == True], "CROP")

    return pd.DataFrame(rows)


def choose_thresholds_from_baseline(df_base: pd.DataFrame) -> List[float]:
    """
    Thresholds in vox derived from baseline Geom@1 distribution (P25/P50/P75).
    """
    # Extract top-1 per query (rank==1) distances
    df1 = df_base[df_base["rank"] == 1]
    d1 = df1["geom_dist_vox"].to_numpy(dtype=np.float64)
    d1 = d1[np.isfinite(d1)]
    if d1.size == 0:
        return [50.0, 100.0, 200.0]
    p25, p50, p75 = np.quantile(d1, [0.25, 0.50, 0.75]).tolist()
    # Ensure strictly positive, readable
    return [float(max(1e-6, p25)), float(max(1e-6, p50)), float(max(1e-6, p75))]


def _pivot(metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Pivot to: rows=(group,source,metric) cols=value
    """
    return metrics.pivot_table(
        index=["group", "source", "metric"],
        values=value_col,
        aggfunc="first",
    ).sort_index()


def compare_reports(df_base: pd.DataFrame, df_rerank: pd.DataFrame, ks: List[int]) -> None:
    thr = choose_thresholds_from_baseline(df_base)
    print(f"\nThresholds (vox) from baseline Geom@1 percentiles: {', '.join(f'{t:.2f}' for t in thr)}")

    m_base = compute_metrics(df_base, ks=ks, thresholds=thr)
    m_rer  = compute_metrics(df_rerank, ks=ks, thresholds=thr)

    p_base = _pivot(m_base, "value").rename(columns={"value": "baseline"})
    p_rer  = _pivot(m_rer, "value").rename(columns={"value": "rerank"})

    joined = p_base.join(p_rer, how="outer")
    joined["delta"] = joined["rerank"] - joined["baseline"]

    with pd.option_context(
        "display.max_rows", 500,
        "display.max_columns", 10,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:10.4f}",
    ):
        print("\n=== Baseline vs Rerank (rerank - baseline) ===")
        print(joined)


def single_report(df: pd.DataFrame, ks: List[int]) -> None:
    # thresholds based on that file itself
    thr = choose_thresholds_from_baseline(df)
    print(f"\nThresholds (vox) from Geom@1 percentiles: {', '.join(f'{t:.2f}' for t in thr)}")

    m = compute_metrics(df, ks=ks, thresholds=thr)
    p = _pivot(m, "value")

    with pd.option_context(
        "display.max_rows", 500,
        "display.max_columns", 10,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:10.4f}",
    ):
        print("\n=== Metrics ===")
        print(p)


