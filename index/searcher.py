from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Optional, TypedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import faiss

from .model.dino import model
from .store import IndexStore, unpack_token_id
from .utils import log, l2norm_rows

__all__ = ["SliceSearcher"]


class Candidate(TypedDict, total=False):
    slice_id: int
    base_score: float
    best_base_angle: float
    col_score: float
    best_col_angle: float
    best_scale: int
    col_heat: np.ndarray  # or List[float]


class SearchParams(TypedDict):
    angles: Iterable[float]
    scales: Iterable[int]
    mode: str
    base_topk_for_col: int
    base_per_rotation: int
    token_topM: int
    token_scales: Iterable[int]
    use_token_ann: bool
    rerank_final_k: int
    debug: bool


class SliceSearcher:
    """
    Stage-1 (hybrid):
      - Global (per rotation) using coarse.faiss
      - Per-scale (k in SLICE_SCALES, per rotation) using coarse.scale{k}.faiss
      - Token ANN shortlist (IVF-PQ) using query tokens at selected scales

    Stage-2:
      - ColBERT-style Windowed MaxSim rerank on candidate slices

    Output: top-K hits with pose/grid metadata.
    """

    def __init__(self, store: IndexStore):
        self.store = store
        self.grid_hw: int = int(self.store.manifest.iloc[0].grid_h)

    # ---------- Public API ----------

    def search_image(self, img_np: np.ndarray, **kwargs) -> List[Dict]:
        # Pack params to keep signature tidy
        params: SearchParams = {
            "angles": kwargs.get("angles", (0, 90, 180, 270)),
            "scales": kwargs.get("scales", (1, 2, 4, 8, 14)),
            "mode": kwargs.get("mode", "base"),
            "base_topk_for_col": kwargs.get("base_topk_for_col", 200),
            "base_per_rotation": kwargs.get("base_per_rotation", 200),
            "token_topM": kwargs.get("token_topM", 32),
            "token_scales": kwargs.get("token_scales", (4, 8, 14)),
            "use_token_ann": kwargs.get("use_token_ann", True),
            "rerank_final_k": kwargs.get("rerank_final_k", 10),
            "debug": kwargs.get("debug", False),
        }

        # 1) Query embeddings (angles × scales)
        q_vars = model.embed_query_augmentations(
            img_np, angles=params["angles"], scales=params["scales"]
        )

        # 2) Stage-1 (hybrid) coarse shortlist
        agg_g = self._coarse_global(q_vars, per_rot=params["base_per_rotation"])
        agg_s = self._coarse_per_scale(q_vars, per_rot=params["base_per_rotation"])
        agg_t = (
            self._coarse_token_ann(
                q_vars, token_topM=params["token_topM"], token_scales=set(params["token_scales"])
            )
            if params["use_token_ann"]
            else {}
        )

        candidates = self._merge_candidates(
            agg_g, agg_s, agg_t, max_total=params["base_topk_for_col"], debug=params["debug"]
        )
        if not candidates:
            return []

        # 3) Either return base-ranking or run Stage-2 rerank
        if params["mode"] == "base":
            base_sorted = sorted(candidates, key=lambda x: x["base_score"], reverse=True)[: params["rerank_final_k"]]
            return self._enrich_and_format(base_sorted, use_col=False)

        reranked = self._rerank_colbert(candidates, q_vars, scales=params["scales"])
        valid = [r for r in reranked if r.get("col_score") is not None and np.isfinite(r["col_score"])]
        valid.sort(key=lambda x: x["col_score"], reverse=True)
        return self._enrich_and_format(valid[: params["rerank_final_k"]], use_col=True)

    # ---------- Stage 1: hybrid coarse retrieval ----------

    def _coarse_global(self, q_vars, per_rot: int) -> Dict[int, Candidate]:
        agg: Dict[int, Candidate] = {}
        Qg = np.stack([q["global"] for q in q_vars], axis=0).astype(np.float32)
        faiss.normalize_L2(Qg)
        for aidx in range(Qg.shape[0]):
            D, I = self.store.search(self.store.coarse, Qg[aidx : aidx + 1], per_rot)
            angle = float(q_vars[aidx]["angle"])
            for sid, sc in zip(I[0].tolist(), D[0].tolist()):
                if sid != -1:
                    self._best_update(agg, int(sid), float(sc), angle)
        return agg

    def _coarse_per_scale(self, q_vars, per_rot: int) -> Dict[int, Candidate]:
        agg: Dict[int, Candidate] = {}
        for q in q_vars:
            angle = float(q["angle"])
            for s in q["scales"]:
                k = int(s["k"])
                if k == 1:
                    continue  # 1×1 is the global vector; handled above
                idx = self.store.coarse_scale(k)
                # Pool query tokens at this scale to a single vector (simple mean)
                v = np.mean(s["Qt"].astype(np.float32), axis=0, keepdims=True)
                faiss.normalize_L2(v)
                D, I = self.store.search(idx, v, per_rot)
                for sid, sc in zip(I[0].tolist(), D[0].tolist()):
                    if sid != -1:
                        self._best_update(agg, int(sid), float(sc), angle)
        return agg

    def _coarse_token_ann(self, q_vars, token_topM: int, token_scales: set) -> Dict[int, float]:
        if self.store.token_ivfpq is None:
            return {}
        token_scores: Dict[int, float] = {}
        for q in q_vars:
            for s in q["scales"]:
                k = int(s["k"])
                if k not in token_scales:
                    continue
                Qt = l2norm_rows(s["Qt"].astype(np.float32))  # cosine/IP friendly
                D_all, I_all = self.store.search_tokens(Qt, token_topM)
                # For each query token, keep best match per slice; then average over tokens
                per_slice_best: Dict[int, List[float]] = {}
                for r in range(I_all.shape[0]):
                    ids_row = I_all[r]
                    ds_row = D_all[r]
                    local_best: Dict[int, float] = {}
                    for packed, dval in zip(ids_row.tolist(), ds_row.tolist()):
                        if packed == -1:
                            continue
                        sid, _ = unpack_token_id(int(packed))
                        cur = local_best.get(sid)
                        if (cur is None) or (dval > cur):
                            local_best[sid] = float(dval)
                    for sid, dval in local_best.items():
                        arr = per_slice_best.setdefault(sid, [])
                        arr.append(dval)
                for sid, vals in per_slice_best.items():
                    s = float(np.mean(vals))
                    prev = token_scores.get(sid)
                    if (prev is None) or (s > prev):
                        token_scores[sid] = s
        return token_scores

    def _merge_candidates(
        self, agg_g: Dict[int, Candidate], agg_s: Dict[int, Candidate], token_scores: Dict[int, float], *, max_total: int, debug: bool
    ) -> List[Candidate]:
        agg: Dict[int, Candidate] = {}

        def upd(c: Dict[int, Candidate]) -> None:
            for sid, row in c.items():
                prev = agg.get(sid)
                if (prev is None) or (row["base_score"] > prev["base_score"]):
                    agg[sid] = row

        upd(agg_g)
        upd(agg_s)

        for sid, s in token_scores.items():
            prev = agg.get(sid)
            if (prev is None) or (s > prev["base_score"]):
                agg[sid] = {"slice_id": sid, "base_score": float(s), "best_base_angle": 0.0}

        out = list(agg.values())
        out.sort(key=lambda x: x["base_score"], reverse=True)
        out = out[:max_total]

        if debug:
            log(
                "search",
                [
                    f"stage-1 merged={len(out)} (global={len(agg_g)}, per-scale={len(agg_s)}, token={len(token_scores)})",
                    "peek: " + ", ".join(f"({r['slice_id']},{r['base_score']:.4f})" for r in out[:5]),
                ],
            )
        return out

    # ---------- Stage 2: ColBERT rerank ----------

    def _rerank_colbert(self, base: List[Dict], q_vars: List[Dict], *, scales: Iterable[int]) -> List[Dict]:
        """Rerank Stage-1 candidates with windowed MaxSim over (angle × scale)."""
        out: List[Dict] = []
        for row in base:
            sid = int(row["slice_id"])
            Ct, cm = self._load_candidate_tokens_and_mask(sid)
            if Ct is None:
                out.append(row)
                continue
            best_s, best_a, best_k, best_heat = self._best_score_windowed(q_vars, Ct, cm, scales=scales)
            row["col_score"] = best_s
            row["best_col_angle"] = best_a
            row["best_scale"] = best_k
            row["col_heat"] = best_heat
            out.append(row)
        return out

    def _load_candidate_tokens_and_mask(self, slice_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load candidate tokens (L2-row-normalized) and mask (bool)."""
        tpath = self.store.token_path(slice_id)
        if not tpath:
            return None, None
        Ct = np.load(tpath, mmap_mode="r").astype(np.float32)
        Ct = l2norm_rows(Ct)

        mpath = self.store.mask_path(slice_id)
        if mpath and Path(mpath).exists():
            cm = np.load(mpath, mmap_mode="r").astype(bool)
        else:
            cm = np.ones((Ct.shape[0],), dtype=bool)
        return Ct, cm

    def _best_score_windowed(
        self, q_vars: List[Dict], Ct: np.ndarray, cm: np.ndarray, *, scales: Iterable[int]
    ) -> Tuple[float, float, Optional[int], Optional[np.ndarray]]:
        """
        Evaluate all (angle × scale) query variants with windowed MaxSim.
        Returns (best_score, best_angle, best_scale, best_heat_196).
        """
        ghw = self.grid_hw
        want = set(scales)
        best_s, best_a, best_k, best_heat = -1.0, 0.0, None, None

        for q in q_vars:
            for s_item in q["scales"]:
                k = int(s_item["k"])
                if k not in want:
                    continue
                Qt = s_item["Qt"]  # (k*k, D), assumed L2-row-normalized upstream
                qm = s_item["qm"]  # (k*k,) bool

                s, heat = self._windowed_colbert_score(Qt, qm, Ct, cm, grid_hw=ghw, k=k)
                if s > best_s:
                    best_s, best_a, best_k, best_heat = s, float(q["angle"]), k, heat

        return best_s, best_a, best_k, best_heat
    @staticmethod
    def _best_update(agg: Dict[int, "Candidate"], sid: int, score: float, angle: float) -> None:
        """Keep the best (highest) base_score per slice_id; update angle alongside."""
        prev = agg.get(sid)
        if (prev is None) or (score > prev["base_score"]):
            agg[sid] = {
                "slice_id": sid,
                "base_score": float(score),
                "best_base_angle": float(angle),
            }
            
    @staticmethod
    @lru_cache(maxsize=256)
    def _window_indices(grid_hw: int, k: int) -> Tuple[np.ndarray, ...]:
        idx_grid = np.arange(grid_hw * grid_hw, dtype=np.int32).reshape(grid_hw, grid_hw)
        wins: List[np.ndarray] = []
        limit = grid_hw - k + 1
        for i in range(limit):
            for j in range(limit):
                wins.append(idx_grid[i : i + k, j : j + k].reshape(-1))
        return tuple(wins)

    @staticmethod
    def _windowed_colbert_score(
        Qt: np.ndarray,
        qm: np.ndarray,
        Ct14: np.ndarray,
        cm14: np.ndarray,
        *,
        grid_hw: int,
        k: int,
        min_fg_ratio: float = 0.05,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Windowed MaxSim:
            Score_k = max_W ( mean_i max_{j in W∩FG} Qt[i]·Ct[j] )
        Returns (best_score, heat_196) where heat is nonzero only on the chosen window.
        """
        if Qt.size == 0 or Ct14.size == 0:
            return -1.0, None
        if qm.sum() == 0 or cm14.sum() == 0:
            return -1.0, None

        # Similarity matrix once (Tq x 196)
        S = Qt @ Ct14.T

        best_s, best_heat = -1.0, None
        for W in SliceSearcher._window_indices(grid_hw, k):
            cmW = cm14[W]
            if cmW.mean() < min_fg_ratio:
                continue
            cols = W[cmW]
            if cols.size == 0:
                continue

            per_q_max = S[:, cols].max(axis=1)  # (Tq,)
            s = float(per_q_max[qm].mean()) if qm.any() else float(per_q_max.mean())
            if s > best_s:
                best_s = s
                heat = np.zeros((grid_hw * grid_hw,), dtype=np.float32)
                heat[cols] = S[:, cols].max(axis=0).astype(np.float32)
                best_heat = heat
        return best_s, best_heat

    # ---------- Output formatting ----------

    def _enrich_and_format(self, rows: List[Dict], use_col: bool) -> List[Dict]:
        M = self.store.manifest
        out: List[Dict] = []
        for r in rows:
            sid = int(r["slice_id"])
            row = M.loc[sid]
            score, best_angle = (
                (float(r["col_score"]), float(r["best_col_angle"])) if (use_col and r.get("col_score") is not None)
                else (float(r["base_score"]), float(r["best_base_angle"]))
            )

            item = {
                "slice_id": sid,
                "score": score,
                "best_angle": best_angle,
                "base_score": float(r["base_score"]),
            }
            if r.get("col_score") is not None:
                item["col_score"] = float(r["col_score"])
            if r.get("col_heat") is not None:
                item["col_heat"] = [float(x) for x in np.asarray(r["col_heat"]).ravel().tolist()]
            if use_col and r.get("best_scale") is not None:
                item["best_scale"] = int(r["best_scale"])

            item.update(
                {
                    "token_path": row.token_path,
                    "normal_idx": int(row.normal_idx),
                    "depth_idx": int(row.depth_idx),
                    "rot_idx": int(row.rot_idx),
                    "normal": (float(row.normal_x), float(row.normal_y), float(row.normal_z)),
                    "depth_vox": float(row.depth_vox),
                    "rotation_deg": float(row.rotation_deg),
                    "grid_h": int(row.grid_h),
                    "grid_w": int(row.grid_w),
                    "feat_dim": int(row.feat_dim),
                    "size_px": int(row.size_px),
                    "resolution_um": int(row.resolution_um),
                    "linear_interp": bool(row.linear_interp),
                }
            )
            out.append(item)
        return out
