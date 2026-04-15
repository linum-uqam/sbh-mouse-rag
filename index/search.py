from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image

from index.store import IndexStore
from index.model.dino import model
from index.utils import log
from index.reranker.runtime import RerankerRuntimeConfig, RerankerService


RerankQueryMode = Literal["best_angle", "max_over_angles", "per_hit_angle"]
BlendNormMode = Literal["zscore", "minmax"]
LocalSearchMode = Literal["off", "auto", "force"]
LocalScoreMode = Literal["max", "top2_mean"]


@dataclass
class SearchConfig:
    """
    Configuration for image → patch search.
    """
    angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

    # Query augmentations
    flip_x: bool = False   # horizontal mirror (left-right)
    flip_y: bool = False   # vertical mirror (top-bottom)
    pad_to_square: bool = True

    k_per_angle: int = 200
    crop_foreground: bool = True
    bg_threshold: float = 0.05
    min_fg_ratio: float = 0.05
    verbose: bool = True

    # Optional restriction on indexed patch scales.
    # None = search across all scales.
    allowed_scales: Tuple[int, ...] | None = None

    # --- local crop search (v2 auto / v3 force) ---
    local_search_mode: LocalSearchMode = "auto"   # off | auto | force
    local_k_per_view: int | None = None            # defaults to k_per_angle
    local_score_mode: LocalScoreMode = "top2_mean"
    local_weight: float = 0.35
    global_weight: float = 1.00

    # v2 trigger
    auto_local_aspect_threshold: float = 1.35

    # sliding square windows for rectangular queries
    local_crop_overlap: float = 0.50
    local_crop_min_side_px: int = 64
    auto_max_local_crops: int = 3
    force_max_local_crops: int = 8

    # extra square-query crops for v3 / force mode
    force_square_scales: Tuple[int, ...] = (2,)

    # --- optional neural reranker (embedding-only) ---
    use_reranker: bool = False
    rerank_topk: int = 100
    rerank_query_mode: RerankQueryMode = "max_over_angles"

    reranker_model_path: Path = Path("out/reranker/reranker_listwise.pt")
    reranker_device: str = "cuda"
    reranker_batch_size: int = 256

    # should generally match what you trained with
    reranker_normalize_embeddings: bool = True
    reranker_use_fp16: bool = True
    reranker_compile: bool = False

    # --- blend reranker + cosine ---
    # final_score = (1 - rerank_alpha) * cosine + rerank_alpha * reranker
    rerank_alpha: float = 1.0  # 1.0 = pure reranker (previous behavior), 0.5 = half/half
    blend_normalize: bool = True
    blend_norm_mode: BlendNormMode = "zscore"  # "zscore" or "minmax"


@dataclass
class SearchResult:
    """
    One retrieved patch.
    """
    patch_id: int
    score: float
    angle: float  # query rotation angle that produced this coarse score
    meta: Dict[str, Any]
    rerank_score: float | None = None

    # optional metadata for augmented query provenance
    flip_x: bool = False
    flip_y: bool = False
    query_variant_idx: int = -1
    query_branch: str = "global"
    query_crop_id: int = -1


class SliceSearcher:
    """
    High-level searcher over the patch index.

    Typical usage:
        store = IndexStore().load_all()
        searcher = SliceSearcher(store)
        hits, q = searcher.search_image(img_np, k=10)
    """

    def __init__(self, store: IndexStore, cfg: SearchConfig | None = None):
        self.store = store
        self.cfg = cfg or SearchConfig()

        self.reranker: Optional[RerankerService] = None
        if self.cfg.use_reranker:
            rr_cfg = RerankerRuntimeConfig(
                model_path=self.cfg.reranker_model_path,
                device=self.cfg.reranker_device,
                batch_size=self.cfg.reranker_batch_size,
                normalize_embeddings=self.cfg.reranker_normalize_embeddings,
                use_fp16=self.cfg.reranker_use_fp16,
                compile_model=self.cfg.reranker_compile,
            )
            self.reranker = RerankerService(rr_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_key_from_row(row: pd.Series) -> tuple[int, int, int]:
        """
        Unique key for the indexed source image/slice.
        Many patch_ids can belong to the same source slice.
        """
        return (
            int(row["normal_idx"]),
            int(row["depth_idx"]),
            int(row["rot_idx"]),
        )

    def _normalized_allowed_scales(self) -> set[int] | None:
        vals = self.cfg.allowed_scales
        if vals is None:
            return None
        out = {int(v) for v in vals}
        return out if out else None

    def search_image(self, img_np: np.ndarray, k: int = 10) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Search using a query image and return:
          - hits (with meta)
          - q: the preprocessed query image used for embedding (H,W) float32 in [0,1]
        """
        # 1) normalize to grayscale [0,1] and optionally crop
        q, q_pil = self._prepare_query(img_np)

        # 2) build global + optional local query views
        angles = tuple(float(a) for a in self.cfg.angles)
        query_views = self._build_search_views(q, q_pil, angles)
        if not query_views:
            raise ValueError("No query variants were generated. Check angles/flip settings.")

        # 3) embed all query views in a single batch -> (A,D)
        pils = [item["pil"] for item in query_views]
        G = model.embed_pil_batch(pils).astype(np.float32, copy=False)  # (A, D)
        if G.ndim != 2:
            raise ValueError(f"Expected embeddings (A,D), got {G.shape}")

        per_view_k = [int(v.get("k", self.cfg.k_per_angle)) for v in query_views]
        base_k_search = int(max(per_view_k)) if per_view_k else int(self.cfg.k_per_angle)

        allowed_scales = self._normalized_allowed_scales()

        current_k = min(max(base_k_search, int(k)), int(self.store.size))
        if allowed_scales is not None:
            current_k = min(max(current_k, int(k) * 4), int(self.store.size))

        final_results: List[SearchResult] = []
        final_best_slice_count = 0
        final_total_candidates = 0

        while True:
            D_all, I_all = self.store.search(G, current_k)
            results, best_slice_count, total_candidates = self._collect_results_from_search(
                G=G,
                angles=angles,
                query_views=query_views,
                D_all=D_all,
                I_all=I_all,
                k=k,
                allowed_scales=allowed_scales,
            )

            final_results = results
            final_best_slice_count = best_slice_count
            final_total_candidates = total_candidates

            enough = len(results) >= int(k)
            exhausted = current_k >= int(self.store.size)
            no_filter = allowed_scales is None

            if no_filter or enough or exhausted:
                break

            next_k = min(int(self.store.size), int(current_k * 2))
            if next_k <= current_k:
                break
            current_k = next_k

        if self.cfg.verbose:
            n_global = sum(1 for v in query_views if v["branch"] == "global")
            n_local = sum(1 for v in query_views if v["branch"] == "local")
            log("search", [
                f"Angles                        : {len(angles)}",
                f"flip_x                        : {bool(self.cfg.flip_x)}",
                f"flip_y                        : {bool(self.cfg.flip_y)}",
                f"Allowed scales                : {sorted(allowed_scales) if allowed_scales is not None else 'all'}",
                f"Global query variants         : {n_global}",
                f"Local query variants          : {n_local}",
                f"Total query variants          : {len(query_views)}",
                f"FAISS k_search                : {current_k}",
                f"Candidates before patch dedup : {final_total_candidates}",
                f"Unique indexed slices         : {final_best_slice_count}",
                f"Returned results              : {len(final_results)}",
            ])

        # 9) Optional reranking
        if self.cfg.use_reranker and self.reranker is not None and final_results:
            self._apply_reranker(final_results, G)

        return final_results, q

    def _collect_results_from_search(
        self,
        *,
        G: np.ndarray,
        angles: Tuple[float, ...],
        query_views: List[Dict[str, Any]],
        D_all: np.ndarray,
        I_all: np.ndarray,
        k: int,
        allowed_scales: set[int] | None,
    ) -> Tuple[List[SearchResult], int, int]:
        """
        Convert raw FAISS outputs into final SearchResult objects, with optional
        scale filtering applied at manifest-row time.

        Returns:
          - results
          - number of unique indexed slices after filtering/dedup
          - total candidate count examined before patch dedup
        """
        k_search = I_all.shape[1] if I_all.ndim == 2 else int(self.cfg.k_per_angle)

        # 1) aggregate evidence per patch_id across global/local views
        best_patch: Dict[int, Dict[str, Any]] = {}
        total_candidates = 0

        for aug_idx, view in enumerate(query_views):
            angle = float(view["angle"])
            flip_x = bool(view["flip_x"])
            flip_y = bool(view["flip_y"])
            branch = str(view["branch"])
            crop_id = int(view.get("crop_id", -1))
            k_view = min(int(view.get("k", k_search)), k_search)

            d_row = D_all[aug_idx][:k_view]
            i_row = I_all[aug_idx][:k_view]
            total_candidates += len(i_row)

            for score, pid in zip(d_row, i_row):
                pid = int(pid)
                if pid < 0:
                    continue

                sc = float(score)
                rec = best_patch.get(pid)
                if rec is None:
                    rec = {
                        "global_best": None,
                        "global_aug_idx": -1,
                        "global_angle": 0.0,
                        "global_flip_x": False,
                        "global_flip_y": False,
                        "global_crop_id": -1,
                        "local_scores": [],
                        "local_best": None,
                        "local_aug_idx": -1,
                        "local_angle": 0.0,
                        "local_flip_x": False,
                        "local_flip_y": False,
                        "local_crop_id": -1,
                    }
                    best_patch[pid] = rec

                if branch == "global":
                    prev = rec["global_best"]
                    if prev is None or sc > prev:
                        rec["global_best"] = sc
                        rec["global_aug_idx"] = aug_idx
                        rec["global_angle"] = angle
                        rec["global_flip_x"] = flip_x
                        rec["global_flip_y"] = flip_y
                        rec["global_crop_id"] = crop_id
                else:
                    rec["local_scores"].append(sc)
                    prev = rec["local_best"]
                    if prev is None or sc > prev:
                        rec["local_best"] = sc
                        rec["local_aug_idx"] = aug_idx
                        rec["local_angle"] = angle
                        rec["local_flip_x"] = flip_x
                        rec["local_flip_y"] = flip_y
                        rec["local_crop_id"] = crop_id

        if not best_patch:
            return [], 0, total_candidates

        # 2) collapse global/local evidence into a single coarse score per patch
        fused_patch: Dict[int, Tuple[float, float, bool, bool, int, str, int]] = {}
        # patch_id -> (score, angle, flip_x, flip_y, aug_idx, branch, crop_id)
        for pid, rec in best_patch.items():
            g = rec["global_best"]
            l = self._aggregate_local_scores(rec["local_scores"])

            fused = 0.0
            if g is not None:
                fused += float(self.cfg.global_weight) * float(g)
            if l is not None:
                fused += float(self.cfg.local_weight) * float(l)

            if g is None and l is None:
                continue

            global_contrib = float(self.cfg.global_weight) * float(g) if g is not None else -np.inf
            local_contrib = float(self.cfg.local_weight) * float(rec["local_best"]) if rec["local_best"] is not None else -np.inf

            if global_contrib >= local_contrib:
                fused_patch[pid] = (
                    float(fused),
                    float(rec["global_angle"]),
                    bool(rec["global_flip_x"]),
                    bool(rec["global_flip_y"]),
                    int(rec["global_aug_idx"]),
                    "global",
                    int(rec["global_crop_id"]),
                )
            else:
                fused_patch[pid] = (
                    float(fused),
                    float(rec["local_angle"]),
                    bool(rec["local_flip_x"]),
                    bool(rec["local_flip_y"]),
                    int(rec["local_aug_idx"]),
                    "local",
                    int(rec["local_crop_id"]),
                )

        if not fused_patch:
            return [], 0, total_candidates

        # 3) dedup by indexed source slice/image, with optional scale filtering
        candidate_ids = list(fused_patch.keys())
        df_rows = self.store.rows_for_ids(candidate_ids)

        best_slice: Dict[tuple[int, int, int], Tuple[int, float, float, bool, bool, int, str, int]] = {}
        # slice_key -> (patch_id, score, angle, flip_x, flip_y, aug_idx, branch, crop_id)

        for pid in candidate_ids:
            row = df_rows.loc[int(pid)]
            if row.isnull().all():
                continue

            if allowed_scales is not None:
                row_scale = row.get("scale", None)
                if row_scale is None or int(row_scale) not in allowed_scales:
                    continue

            slice_key = self._slice_key_from_row(row)
            score, angle, flip_x, flip_y, aug_idx, branch, crop_id = fused_patch[int(pid)]

            prev = best_slice.get(slice_key)
            if prev is None or score > prev[1]:
                best_slice[slice_key] = (
                    int(pid),
                    float(score),
                    float(angle),
                    bool(flip_x),
                    bool(flip_y),
                    int(aug_idx),
                    str(branch),
                    int(crop_id),
                )

        if not best_slice:
            return [], 0, total_candidates
        

        # 4) sort unique slices by score desc and keep top-k
        sorted_hits = sorted(best_slice.values(), key=lambda t: t[1], reverse=True)
        top = sorted_hits[: int(k)]

        ids = [pid for pid, *_rest in top]
        df_top = self.store.rows_for_ids(ids)

        results: List[SearchResult] = []
        for pid, score, angle, flip_x, flip_y, aug_idx, branch, crop_id in top:
            row = df_top.loc[int(pid)]
            results.append(
                SearchResult(
                    patch_id=int(pid),
                    score=float(score),
                    angle=float(angle),
                    flip_x=bool(flip_x),
                    flip_y=bool(flip_y),
                    query_variant_idx=int(aug_idx),
                    query_branch=str(branch),
                    query_crop_id=int(crop_id),
                    meta=row.to_dict(),
                )
            )

        return results, len(best_slice), total_candidates

    def to_dataframe(self, hits: List[SearchResult]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for h in hits:
            row = dict(h.meta)
            row["patch_id"] = h.patch_id
            row["score"] = h.score
            row["query_angle_deg"] = h.angle
            row["query_flip_x"] = h.flip_x
            row["query_flip_y"] = h.flip_y
            row["query_branch"] = h.query_branch
            row["query_crop_id"] = h.query_crop_id
            if h.rerank_score is not None:
                row["rerank_score"] = h.rerank_score
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Query view building
    # ------------------------------------------------------------------

    def _build_search_views(self, q: np.ndarray, q_pil: Image.Image, angles: Tuple[float, ...]) -> List[Dict[str, Any]]:
        views: List[Dict[str, Any]] = []

        global_pil = self._pad_pil_to_square(q_pil) if self.cfg.pad_to_square else q_pil
        views.extend(self._build_query_variants(global_pil, angles, branch="global", crop_id=-1, k=self.cfg.k_per_angle))

        local_enabled, reason = self._should_enable_local_search(q)
        local_crops = self._build_local_crops(q, force=(self.cfg.local_search_mode == "force")) if local_enabled else []
        local_k = int(self.cfg.local_k_per_view or self.cfg.k_per_angle)

        for crop in local_crops:
            crop_img = crop["image"]
            crop_pil = Image.fromarray((np.clip(crop_img, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            crop_pil = self._pad_pil_to_square(crop_pil) if self.cfg.pad_to_square else crop_pil
            views.extend(
                self._build_query_variants(
                    crop_pil,
                    angles,
                    branch="local",
                    crop_id=int(crop["crop_id"]),
                    k=local_k,
                )
            )

        if self.cfg.verbose:
            h, w = q.shape
            ratio = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
            log("search", [
                f"Prepared query size           : {h}x{w}",
                f"Aspect ratio                  : {ratio:.3f}",
                f"Local search mode             : {self.cfg.local_search_mode}",
                f"Local search active           : {local_enabled}",
                f"Local search reason           : {reason}",
                f"Local crop count              : {len(local_crops)}",
            ])

        return views

    def _build_query_variants(
        self,
        q_pil: Image.Image,
        angles: Tuple[float, ...],
        *,
        branch: str,
        crop_id: int,
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Build all query augmentation variants.

        For each rotation angle, optionally include:
          - original
          - flip_x
          - flip_y
          - flip_x + flip_y
        """
        variants: List[Dict[str, Any]] = []

        flip_states_x = [False, True] if self.cfg.flip_x else [False]
        flip_states_y = [False, True] if self.cfg.flip_y else [False]

        for angle in angles:
            base = q_pil.rotate(float(angle), resample=Image.BILINEAR, expand=False)

            for flip_x in flip_states_x:
                for flip_y in flip_states_y:
                    img = base
                    if flip_x:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if flip_y:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)

                    variants.append(
                        {
                            "pil": img,
                            "angle": float(angle),
                            "flip_x": bool(flip_x),
                            "flip_y": bool(flip_y),
                            "branch": str(branch),
                            "crop_id": int(crop_id),
                            "k": int(k),
                        }
                    )

        return variants

    def _should_enable_local_search(self, q: np.ndarray) -> Tuple[bool, str]:
        mode = str(self.cfg.local_search_mode)
        if mode == "off":
            return False, "disabled"
        if mode == "force":
            return True, "forced"

        h, w = q.shape
        short_side = min(h, w)
        if short_side < int(self.cfg.local_crop_min_side_px):
            return False, f"short_side<{int(self.cfg.local_crop_min_side_px)}"

        ratio = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
        if ratio >= float(self.cfg.auto_local_aspect_threshold):
            return True, f"aspect_ratio>={float(self.cfg.auto_local_aspect_threshold):.2f}"

        return False, "ratio_not_triggered"

    def _build_local_crops(self, q: np.ndarray, *, force: bool) -> List[Dict[str, Any]]:
        h, w = q.shape
        crops: List[Dict[str, Any]] = []
        crop_id = 0
        seen_boxes: set[Tuple[int, int, int, int]] = set()

        def add_crop(y0: int, y1: int, x0: int, x1: int) -> None:
            nonlocal crop_id
            box = (int(y0), int(y1), int(x0), int(x1))
            if box in seen_boxes:
                return
            seen_boxes.add(box)
            crop = q[y0:y1, x0:x1]
            if crop.size == 0:
                return
            ch, cw = crop.shape
            if min(ch, cw) < int(self.cfg.local_crop_min_side_px):
                return
            crops.append({
                "crop_id": crop_id,
                "box": box,
                "image": crop,
            })
            crop_id += 1

        # Rectangular queries: slide square windows along the long axis.
        if h != w:
            side = min(h, w)
            if h > w:
                starts = self._sliding_starts(h, side, float(self.cfg.local_crop_overlap))
                starts = self._reduce_positions(starts, max_count=self._max_local_crops(force))
                for y0 in starts:
                    add_crop(y0, y0 + side, 0, side)
            else:
                starts = self._sliding_starts(w, side, float(self.cfg.local_crop_overlap))
                starts = self._reduce_positions(starts, max_count=self._max_local_crops(force))
                for x0 in starts:
                    add_crop(0, side, x0, x0 + side)

        # Forced mode: also create square-grid local crops on square or rectangular queries.
        if force:
            side = min(h, w)
            q_square = q[:side, :side]
            qh, qw = q_square.shape
            for scale in self.cfg.force_square_scales:
                scale = int(scale)
                if scale <= 1:
                    continue
                win = max(1, int(round(side / float(scale))))
                ys = self._sliding_starts(qh, win, float(self.cfg.local_crop_overlap))
                xs = self._sliding_starts(qw, win, float(self.cfg.local_crop_overlap))
                for y0 in ys:
                    for x0 in xs:
                        add_crop(y0, y0 + win, x0, x0 + win)
                        if len(crops) >= self._max_local_crops(force):
                            break
                    if len(crops) >= self._max_local_crops(force):
                        break
                if len(crops) >= self._max_local_crops(force):
                    break

        return crops[: self._max_local_crops(force)]

    def _max_local_crops(self, force: bool) -> int:
        return int(self.cfg.force_max_local_crops if force else self.cfg.auto_max_local_crops)

    @staticmethod
    def _sliding_starts(length: int, window: int, overlap: float) -> List[int]:
        if window >= length:
            return [0]
        overlap = float(np.clip(overlap, 0.0, 0.95))
        step = max(1, int(round(window * (1.0 - overlap))))
        starts = list(range(0, max(length - window, 0) + 1, step))
        last = length - window
        if not starts or starts[-1] != last:
            starts.append(last)
        return sorted(set(int(s) for s in starts))

    @staticmethod
    def _reduce_positions(starts: List[int], max_count: int) -> List[int]:
        starts = sorted(set(int(s) for s in starts))
        if len(starts) <= max_count:
            return starts
        if max_count <= 1:
            return [starts[len(starts) // 2]]

        idxs = np.linspace(0, len(starts) - 1, num=max_count)
        picked = sorted(set(starts[int(round(i))] for i in idxs))
        return picked

    @staticmethod
    def _pad_pil_to_square(pil: Image.Image) -> Image.Image:
        w, h = pil.size
        if w == h:
            return pil
        side = max(w, h)
        out = Image.new(pil.mode, (side, side), color=0)
        x = (side - w) // 2
        y = (side - h) // 2
        out.paste(pil, (x, y))
        return out

    def _aggregate_local_scores(self, scores: List[float]) -> float | None:
        if not scores:
            return None
        vals = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
        if vals.size == 0:
            return None

        mode = str(self.cfg.local_score_mode)
        if mode == "max" or vals.size == 1:
            return float(vals[0])
        if mode == "top2_mean":
            return float(vals[: min(2, vals.size)].mean())
        raise ValueError(f"Unknown local_score_mode={mode!r}")

    # ------------------------------------------------------------------
    # Reranker integration
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(x: np.ndarray, mode: str = "zscore") -> np.ndarray:
        """
        Normalize a 1D score vector for blending.
        - zscore: (x - mean) / std
        - minmax: (x - min) / (max - min)
        If the vector is (near) constant, returns zeros.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x

        if mode == "zscore":
            mu = float(x.mean())
            sd = float(x.std())
            if sd < 1e-6:
                return np.zeros_like(x)
            return (x - mu) / sd

        if mode == "minmax":
            lo = float(x.min())
            hi = float(x.max())
            if (hi - lo) < 1e-6:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        raise ValueError(f"Unknown normalize mode: {mode!r}")

    def _apply_reranker(self, results: List[SearchResult], G: np.ndarray) -> None:
        """
        Rerank only the top `rerank_topk` results, then keep the remainder in coarse order.

        Modes:
          - best_angle: use query embedding of the best coarse hit's exact augmentation (single listwise call)
          - max_over_angles: score listwise per augmented query, take max score per candidate
          - per_hit_angle: score each hit using its own best coarse augmentation (uses score_pairs)

        Blending:
          final_score = (1 - alpha) * cosine + alpha * reranker
          with optional per-query normalization before blending.
        """
        assert self.reranker is not None

        N = len(results)
        topk = int(max(1, min(self.cfg.rerank_topk, N)))

        top_hits = results[:topk]
        rest_hits = results[topk:]

        ids_for_rerank = [h.patch_id for h in top_hits]
        cand_embs = self.store.vectors_for_ids(ids_for_rerank).astype(np.float32, copy=False)  # (topk,D)

        mode = self.cfg.rerank_query_mode

        if mode == "best_angle":
            q_idx = int(top_hits[0].query_variant_idx)
            q_emb = G[q_idx]
            scores = self.reranker.score_list(q_emb, cand_embs)  # (topk,)

        elif mode == "max_over_angles":
            all_scores = []
            for aug_idx in range(G.shape[0]):
                q_emb = G[aug_idx]
                s = self.reranker.score_list(q_emb, cand_embs)  # (topk,)
                all_scores.append(s)
            scores = np.max(np.stack(all_scores, axis=0), axis=0).astype(np.float32, copy=False)

        elif mode == "per_hit_angle":
            q_embs = np.stack([G[int(h.query_variant_idx)] for h in top_hits], axis=0)  # (topk,D)
            scores = self.reranker.score_pairs(q_embs, cand_embs)  # (topk,)

        else:
            raise ValueError(f"Unknown rerank_query_mode={mode!r}")

        scores = np.asarray(scores, dtype=np.float32)

        alpha = float(self.cfg.rerank_alpha)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        coarse = np.asarray([h.score for h in top_hits], dtype=np.float32)

        if bool(self.cfg.blend_normalize):
            mode_n = str(self.cfg.blend_norm_mode)
            coarse_n = self._normalize_scores(coarse, mode=mode_n)
            rerank_n = self._normalize_scores(scores, mode=mode_n)
        else:
            coarse_n = coarse
            rerank_n = scores

        blended = (1.0 - alpha) * coarse_n + alpha * rerank_n

        for h, s_raw, s_blend in zip(top_hits, scores, blended):
            h.rerank_score = float(s_raw)
            h.score = float(s_blend)

        top_hits_sorted = sorted(
            top_hits,
            key=lambda h: (h.score, h.rerank_score if h.rerank_score is not None else -1e9),
            reverse=True,
        )

        results[:] = top_hits_sorted + rest_hits

    # ------------------------------------------------------------------
    # Query preprocessing
    # ------------------------------------------------------------------

    def _prepare_query(self, img_np: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
        x = np.asarray(img_np)

        if x.ndim == 3 and x.shape[2] in (1, 3):
            if x.shape[2] == 3:
                x = x.mean(axis=2)
            else:
                x = x[..., 0]

        if x.ndim != 2:
            raise ValueError(f"Expected (H,W) or (H,W,1/3) array, got shape {x.shape}")

        x = x.astype(np.float32, copy=False)
        if x.max() > 1.0:
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0)

        if self.cfg.crop_foreground:
            x = self._auto_crop_foreground(x)

        x8 = (x * 255.0).astype(np.uint8, copy=False)
        pil = Image.fromarray(x8, mode="L")
        return x, pil

    def _auto_crop_foreground(self, img: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        mask = img > float(cfg.bg_threshold)
        fg_ratio = float(mask.sum()) / float(mask.size)

        if fg_ratio < cfg.min_fg_ratio:
            return img

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        cropped = img[y0:y1, x0:x1]

        if cfg.verbose:
            H, W = img.shape
            ch, cw = cropped.shape
            log("search", [f"Auto-crop foreground: ({H}x{W}) -> ({ch}x{cw})"])

        return cropped
