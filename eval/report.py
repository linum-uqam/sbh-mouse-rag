from __future__ import annotations

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

    required = {"row_idx", "source", "rank", "geom_dist_vox"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_tau = "gt_tau_vox" in df.columns
    has_crop = "q_is_crop" in df.columns
    has_crop_kind = "q_crop_kind" in df.columns
    has_search_mode = "search_mode" in df.columns

    def run_block(df_block: pd.DataFrame, label: str) -> None:
        per_query = []

        for (_, src), dfq in _iter_queries(df_block):
            d = dfq["geom_dist_vox"].to_numpy(dtype=np.float64)
            pm = _prefix_min(d)

            finite = np.isfinite(d)
            if finite.any():
                best_rank = int(np.argmin(np.where(finite, d, np.inf)) + 1)
            else:
                best_rank = None

            tau_q = float(dfq["gt_tau_vox"].iloc[0]) if has_tau else float(np.nan)
            per_query.append((src, d, pm, best_rank, tau_q))

        if not per_query:
            return

        for src_name in sorted({x[0] for x in per_query} | {"ALL"}):
            subset = [x for x in per_query if (src_name == "ALL" or x[0] == src_name)]
            if not subset:
                continue

            n_q = len(subset)

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

            for thr in thresholds:
                thr = float(thr)

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
                        r = int(np.argmax(ok) + 1)
                        mrr.append(1.0 / float(r))
                rows.append({
                    "group": label,
                    "source": src_name,
                    "metric": f"MRR_geom(thr={thr:.2f})",
                    "value": float(np.mean(mrr)) if mrr else float("nan"),
                    "n_queries": n_q,
                })

            for K in [10, 100]:
                ndcgs = []
                for _, d, _, __, tau_q in subset:
                    if not np.isfinite(tau_q):
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

    run_block(df, "ALL")

    if has_crop:
        run_block(df[df["q_is_crop"] == False], "FULL")
        run_block(df[df["q_is_crop"] == True], "CROP")

    if has_crop and has_crop_kind:
        df_crop = df[df["q_is_crop"] == True]
        for kind in sorted(df_crop["q_crop_kind"].dropna().astype(str).unique()):
            run_block(df_crop[df_crop["q_crop_kind"].astype(str) == kind], f"CROP_KIND={kind}")

    if has_search_mode:
        for mode in sorted(df["search_mode"].dropna().astype(str).unique()):
            run_block(df[df["search_mode"].astype(str) == mode], f"SEARCH_MODE={mode}")

    return pd.DataFrame(rows)


def choose_thresholds_from_baseline(df_base: pd.DataFrame) -> List[float]:
    df1 = df_base[df_base["rank"] == 1]
    d1 = df1["geom_dist_vox"].to_numpy(dtype=np.float64)
    d1 = d1[np.isfinite(d1)]
    if d1.size == 0:
        return [50.0, 100.0, 200.0]
    p25, p50, p75 = np.quantile(d1, [0.25, 0.50, 0.75]).tolist()
    return [float(max(1e-6, p25)), float(max(1e-6, p50)), float(max(1e-6, p75))]


def _pivot(metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return metrics.pivot_table(
        index=["group", "source", "metric"],
        values=value_col,
        aggfunc="first",
    ).sort_index()


def load_eval_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "eval_hits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find eval CSV: {path}")
    return pd.read_csv(path)




# ----------------------------
# Report display helpers
# ----------------------------

def _metric_prefers_lower(metric_name: str) -> bool:
    return metric_name.startswith("Geom@") or metric_name.startswith("RankBestGeom")

def _decorate_best_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a display-only copy where the best numeric run values in each row are suffixed with a star.
    Delta columns are left undecorated.
    """
    out = df.copy().astype(object)

    run_cols = [c for c in df.columns if not str(c).startswith("delta")]
    for idx in df.index:
        metric = str(idx[2]) if isinstance(idx, tuple) and len(idx) >= 3 else ""
        vals = pd.to_numeric(df.loc[idx, run_cols], errors="coerce")
        finite = vals[np.isfinite(vals.to_numpy(dtype=float))]
        # format all run columns as strings first
        for c in run_cols:
            v = df.loc[idx, c]
            out.at[idx, c] = "" if pd.isna(v) else f"{float(v):.4f}"
        # format delta columns too
        for c in [c for c in df.columns if str(c).startswith("delta")]:
            v = df.loc[idx, c]
            out.at[idx, c] = "" if pd.isna(v) else f"{float(v):+.4f}"
        if finite.empty:
            continue
        best = float(finite.min() if _metric_prefers_lower(metric) else finite.max())
        for c in run_cols:
            v = pd.to_numeric(pd.Series([df.loc[idx, c]]), errors="coerce").iloc[0]
            if pd.isna(v):
                continue
            if np.isclose(float(v), best, rtol=1e-12, atol=1e-12):
                out.at[idx, c] = f"{float(v):.4f} ★"
    return out

def print_metric_legend() -> None:
    print("\nMetric guide:")
    print("  - Geom@K_mean / Geom@K_median: best geometric distance found within the top-K. Lower is better.")
    print("  - RankBestGeom_mean / RankBestGeom_median: rank position of the geometrically best candidate. Lower is better.")
    print("  - SR@K(thr=T): success rate; fraction of queries with at least one candidate within distance threshold T in top-K. Higher is better.")
    print("  - MRR_geom(thr=T): reciprocal rank of the first candidate within threshold T. Higher is better.")
    print("  - NDCG@K_mean / NDCG@K_median: ranking quality over the top-K using geometry-derived relevance. Higher is better.")
    print("  - ★ marks the best run value in that row.")

def print_metrics_table(df: pd.DataFrame, title: str, *, decorate_best: bool = False) -> None:
    with pd.option_context(
        "display.max_rows", 1000,
        "display.max_columns", 100,
        "display.width", 260,
        "display.float_format", lambda x: f"{x:10.4f}",
    ):
        print(f"\n=== {title} ===")
        display_df = _decorate_best_values(df) if decorate_best else df
        print(display_df)


def single_report(df: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    thr = choose_thresholds_from_baseline(df)
    print(f"\nThresholds (vox) from Geom@1 percentiles: {', '.join(f'{t:.2f}' for t in thr)}")

    m = compute_metrics(df, ks=ks, thresholds=thr)
    p = _pivot(m, "value")
    print_metrics_table(p, "Metrics", decorate_best=True)
    print_metric_legend()
    return p


def compare_reports(df_base: pd.DataFrame, df_rerank: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    thr = choose_thresholds_from_baseline(df_base)
    print(f"\nThresholds (vox) from baseline Geom@1 percentiles: {', '.join(f'{t:.2f}' for t in thr)}")

    m_base = compute_metrics(df_base, ks=ks, thresholds=thr)
    m_rer = compute_metrics(df_rerank, ks=ks, thresholds=thr)

    p_base = _pivot(m_base, "value").rename(columns={"value": "baseline"})
    p_rer = _pivot(m_rer, "value").rename(columns={"value": "rerank"})

    joined = p_base.join(p_rer, how="outer")
    joined["delta"] = joined["rerank"] - joined["baseline"]
    print_metrics_table(joined, "Baseline vs Rerank (rerank - baseline)", decorate_best=True)
    print_metric_legend()
    return joined


def compare_named_reports(named_dfs: Dict[str, pd.DataFrame], ks: List[int]) -> pd.DataFrame:
    if not named_dfs:
        raise ValueError("No runs provided")

    first_name = next(iter(named_dfs.keys()))
    thr = choose_thresholds_from_baseline(named_dfs[first_name])
    print(f"\nThresholds (vox) from baseline '{first_name}' Geom@1 percentiles: {', '.join(f'{t:.2f}' for t in thr)}")

    metric_tables: Dict[str, pd.DataFrame] = {}
    for name, df in named_dfs.items():
        m = compute_metrics(df, ks=ks, thresholds=thr)
        metric_tables[name] = _pivot(m, "value").rename(columns={"value": name})

    joined: Optional[pd.DataFrame] = None
    for name in named_dfs.keys():
        joined = metric_tables[name] if joined is None else joined.join(metric_tables[name], how="outer")

    assert joined is not None
    baseline_col = first_name
    for name in list(named_dfs.keys())[1:]:
        joined[f"delta_vs_{baseline_col}__{name}"] = joined[name] - joined[baseline_col]

    print_metrics_table(joined, f"Multi-run comparison (baseline={baseline_col})", decorate_best=True)
    print_metric_legend()
    return joined


def resolve_named_inputs(paths: List[str], labels: List[str]) -> Dict[str, pd.DataFrame]:
    if not paths:
        raise ValueError("Provide at least one csv path")

    if labels and len(labels) != len(paths):
        raise ValueError("If provided, labels must match the number of csv paths")

    if labels:
        names = labels
    else:
        names = []
        for p in paths:
            path = Path(p)
            names.append(path.name if path.is_dir() else path.stem)

        seen: Dict[str, int] = {}
        deduped: List[str] = []
        for n in names:
            c = seen.get(n, 0)
            seen[n] = c + 1
            deduped.append(n if c == 0 else f"{n}_{c+1}")
        names = deduped

    return {name: load_eval_csv(path) for name, path in zip(names, paths)}


def reorder_with_baseline(named: Dict[str, pd.DataFrame], baseline: Optional[str]) -> Dict[str, pd.DataFrame]:
    if not named or baseline is None:
        return named
    if baseline not in named:
        raise ValueError(f"Baseline label '{baseline}' not found in runs: {list(named.keys())}")
    ordered = {baseline: named[baseline]}
    for k, v in named.items():
        if k != baseline:
            ordered[k] = v
    return ordered
