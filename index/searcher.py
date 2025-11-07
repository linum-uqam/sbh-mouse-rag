# index/searcher.py
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Optional
from functools import lru_cache
from pathlib import Path

import numpy as np
import faiss

from .model.dino import model
from .store import IndexStore

__all__ = ["SliceSearcher"]

_EPS = 1e-12


def _l2norm_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / (n + _EPS)


class SliceSearcher:
    """
    Orchestrates:
      - Query embedding (angle × scale) using model.embed_query_augmentations
      - Base retrieval via FAISS (global vectors)
      - ColBERT-style Windowed MaxSim rerank (masked)
      - Output formatting w/ metadata from manifest
    """

    def __init__(self, store: IndexStore):
        self.store = store
        # Cache grid size (e.g., 14) once; manifest is indexed by 'id'
        # Assume constant grid size across rows.
        self.grid_hw: int = int(self.store.manifest.iloc[0].grid_h)

    # ---------- Public API ----------

    def search_image(
        self,
        img_np: np.ndarray,
        *,
        angles: Iterable[float] = (0.0, 90.0, 180.0, 270.0),
        scales: Iterable[int] = (1, 2, 4, 8, 14),
        mode: str = "base",               # "base" or "col"
        base_topk_for_col: int = 200,     # how many base hits to consider for col rerank
        base_per_rotation: int = 200,     # base hits per rotation
        rerank_final_k: int = 10,         # number of final results
    ) -> List[Dict]:
        """
        Returns a list of hits. Each hit contains:
          - slice_id
          - score (base or col), best_angle, base_score, (optional) col_score, best_scale, col_heat
          - pose + token grid metadata from manifest
        """
        # 1) Query embedding (angle × scale)
        q_vars = self._embed_query(img_np, angles=angles, scales=scales)

        # 2) Base/global retrieval (one FAISS search per rotation)
        Qg = np.stack([q["global"] for q in q_vars], axis=0).astype(np.float32)
        faiss.normalize_L2(Qg)
        base = self._base_retrieve(Qg, q_vars, per_rot=base_per_rotation, topk=base_topk_for_col)
        if not base:
            return []

        # Base-only mode
        if mode == "base":
            base_sorted = sorted(base, key=lambda x: x["base_score"], reverse=True)[:rerank_final_k]
            return self._enrich_and_format(base_sorted, use_col=False)

        # 3) Rerank with Windowed MaxSim
        reranked = self._rerank_colbert(base, q_vars, scales=scales)

        if not reranked or all(r.get("col_score") is None for r in reranked):
            # Fallback to base if no token info
            base_sorted = sorted(base, key=lambda x: x["base_score"], reverse=True)[:rerank_final_k]
            return self._enrich_and_format(base_sorted, use_col=False)

        # Top-K by col score
        reranked = [r for r in reranked if (r["col_score"] is not None and np.isfinite(r["col_score"])) ]
        reranked.sort(key=lambda x: x["col_score"], reverse=True)
        reranked = reranked[:rerank_final_k]
        return self._enrich_and_format(reranked, use_col=True)

    # ---------- Stage 1: Query Embedding ----------

    @staticmethod
    def _embed_query(img_np: np.ndarray, *, angles: Iterable[float], scales: Iterable[int]) -> List[Dict]:
        """
        Produces a list over rotations; each item contains:
          {
            "angle": float,
            "global": (D,),
            "tokens14": (196,D),
            "qmask14": (196,) bool,
            "scales": [{"k": k, "Qt": (k*k,D), "qm": (k*k,) bool}, ...]
          }
        """
        return model.embed_query_augmentations(img_np, angles=angles, scales=scales)

    # ---------- Stage 2: Base Retrieval ----------

    def _base_retrieve(
        self,
        Qg: np.ndarray,
        q_vars: List[Dict],
        *,
        per_rot: int,
        topk: int,
    ) -> List[Dict]:
        """
        Searches FAISS once per rotation; aggregates the best base score per slice.
        """
        agg: Dict[int, Dict] = {}

        for aidx in range(Qg.shape[0]):
            D, I = self.store.search(self.store.coarse, Qg[aidx:aidx+1], per_rot)
            scores = D[0]
            ids = I[0]
            angle = float(q_vars[aidx]["angle"])

            for sid, sc in zip(ids.tolist(), scores.tolist()):
                if sid == -1:
                    continue
                sid = int(sid)
                prev = agg.get(sid)
                if (prev is None) or (sc > prev["base_score"]):
                    agg[sid] = {
                        "slice_id": sid,
                        "base_score": float(sc),
                        "best_base_angle": angle,
                        # placeholders to be filled by reranker (if any)
                        "col_score": None,
                        "best_col_angle": None,
                        "best_scale": None,
                        "col_heat": None,
                    }

        uniq = list(agg.values())
        uniq.sort(key=lambda x: x["base_score"], reverse=True)
        return uniq[:topk]

    # ---------- Stage 3: ColBERT Rerank (Windowed MaxSim) ----------

    def _rerank_colbert(
        self,
        base: List[Dict],
        q_vars: List[Dict],
        *,
        scales: Iterable[int],
    ) -> List[Dict]:
        """
        For each base candidate:
          - Load candidate tokens + mask
          - Score all (angle × scale) query variants using Windowed MaxSim
          - Keep best score/angle/scale and a 196-d heatmap for overlay
        """
        out: List[Dict] = []
        for row in base:
            sid = int(row["slice_id"])
            Ct, cm = self._load_candidate_tokens_and_mask(sid)
            if Ct is None:
                # no tokens available; carry forward base data
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
        """
        Loads candidate token matrix (float32, L2-row-normalized) and mask (bool).
        Returns (Ct, cm) or (None, None) if token path missing.
        """
        tpath = self.store.token_path(slice_id)
        if not tpath:
            return None, None

        Ct = np.load(tpath, mmap_mode="r").astype(np.float32)  # (Tc,D)
        Ct = _l2norm_rows(Ct)
        mpath = self.store.mask_path(slice_id)
        if mpath and Path(mpath).exists():
            cm = np.load(mpath).astype(bool)  # (Tc,)
        else:
            cm = np.ones((Ct.shape[0],), dtype=bool)
        return Ct, cm

    def _best_score_windowed(
        self,
        q_vars: List[Dict],
        Ct: np.ndarray,
        cm: np.ndarray,
        *,
        scales: Iterable[int],
    ) -> Tuple[float, float, Optional[int], Optional[np.ndarray]]:
        """
        Windowed MaxSim across all rotations and scales.
        Returns (best_score, best_angle, best_scale, best_heat_196)
        """
        best_s, best_a, best_k, best_heat = -1.0, 0.0, None, None
        ghw = self.grid_hw  # e.g., 14

        for q in q_vars:
            for s_item in q["scales"]:
                k = int(s_item["k"])
                # Skip scales not requested (if caller provided a subset)
                if k not in set(scales):
                    continue

                Qt = s_item["Qt"]  # (k*k,D), row-normalized
                qm = s_item["qm"]  # (k*k,) bool

                s, heat = self._windowed_colbert_score(Qt, qm, Ct, cm, grid_hw=ghw, k=k)
                if s > best_s:
                    best_s, best_a, best_k, best_heat = s, float(q["angle"]), k, heat

        return best_s, best_a, best_k, best_heat

    @staticmethod
    @lru_cache(maxsize=256)
    def _window_indices(grid_hw: int, k: int) -> Tuple[np.ndarray, ...]:
        """
        Cached enumerator of flat-index windows (each window is a np.ndarray of shape (k*k,))
        on a grid_hw x grid_hw lattice.
        """
        idx_grid = np.arange(grid_hw * grid_hw, dtype=np.int32).reshape(grid_hw, grid_hw)
        wins: List[np.ndarray] = []
        limit = grid_hw - k + 1
        for i in range(limit):
            for j in range(limit):
                wins.append(idx_grid[i:i + k, j:j + k].reshape(-1))
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
        Returns (best_score, heat_196) where heat is nonzero only on the chosen window (max over query).
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

            per_q_max = S[:, cols].max(axis=1)       # (Tq,)
            s = float(per_q_max[qm].mean()) if qm.any() else float(per_q_max.mean())
            if s > best_s:
                best_s = s
                heat = np.zeros((grid_hw * grid_hw,), dtype=np.float32)
                heat[cols] = S[:, cols].max(axis=0).astype(np.float32)
                best_heat = heat
        return best_s, best_heat

    # ---------- Stage 4: Output Formatting ----------

    def _enrich_and_format(self, rows: List[Dict], use_col: bool) -> List[Dict]:
        """
        Join per-slice metadata from manifest and normalize the shape of each output row.
        """
        M = self.store.manifest
        out: List[Dict] = []

        for r in rows:
            sid = int(r["slice_id"])
            row = M.loc[sid]

            # Select final score/angle
            score, best_angle = (
                (float(r["col_score"]), float(r["best_col_angle"]))
                if (use_col and r.get("col_score") is not None)
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

            # Pose & grid metadata from manifest
            item.update({
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
            })
            out.append(item)

        return out
