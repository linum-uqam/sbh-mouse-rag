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


@dataclass
class SearchConfig:
    """
    Configuration for image → patch search.
    """
    angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

    # Query augmentations
    flip_x: bool = False   # horizontal mirror (left-right)
    flip_y: bool = False   # vertical mirror (top-bottom)

    k_per_angle: int = 200
    crop_foreground: bool = True
    bg_threshold: float = 0.05
    min_fg_ratio: float = 0.05
    verbose: bool = True

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

    def search_image(self, img_np: np.ndarray, k: int = 10) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Search using a query image and return:
          - hits (with meta)
          - q: the preprocessed query image used for embedding (H,W) float32 in [0,1]
        """
        # 1) normalize to grayscale [0,1] and optionally crop
        q, q_pil = self._prepare_query(img_np)

        # 2) build augmented query views as PIL grayscale
        angles = tuple(float(a) for a in self.cfg.angles)
        aug_views = self._build_query_variants(q_pil, angles)

        if not aug_views:
            raise ValueError("No query variants were generated. Check angles/flip settings.")

        pils = [item["pil"] for item in aug_views]

        # 3) embed all augmented views in a single batch -> (A,D)
        G = model.embed_pil_batch(pils).astype(np.float32, copy=False)  # (A, D)
        if G.ndim != 2:
            raise ValueError(f"Expected embeddings (A,D), got {G.shape}")

        # 4) single FAISS call for all augmented views at once
        D_all, I_all = self.store.search(G, self.cfg.k_per_angle)  # each (A, k_per_angle)

        # 5) first dedup: keep best score per patch_id across all query augmentations
        # patch_id -> (score, best_query_angle, best_flip_x, best_flip_y, aug_idx)
        best_patch: Dict[int, Tuple[float, float, bool, bool, int]] = {}

        for aug_idx, aug in enumerate(aug_views):
            angle = float(aug["angle"])
            flip_x = bool(aug["flip_x"])
            flip_y = bool(aug["flip_y"])

            d_row = D_all[aug_idx]
            i_row = I_all[aug_idx]

            for score, pid in zip(d_row, i_row):
                pid = int(pid)
                if pid < 0:
                    continue

                sc = float(score)
                prev = best_patch.get(pid)
                if prev is None or sc > prev[0]:
                    best_patch[pid] = (sc, angle, flip_x, flip_y, aug_idx)

        if not best_patch:
            return [], q

        # 6) second dedup: keep only one best patch per indexed source slice/image
        candidate_ids = list(best_patch.keys())
        df_rows = self.store.rows_for_ids(candidate_ids)

        # slice_key -> (patch_id, score, angle, flip_x, flip_y, aug_idx)
        best_slice: Dict[tuple[int, int, int], Tuple[int, float, float, bool, bool, int]] = {}

        for pid in candidate_ids:
            row = df_rows.loc[int(pid)]

            # Defensive guard for missing manifest rows
            if row.isnull().all():
                continue

            slice_key = self._slice_key_from_row(row)
            score, angle, flip_x, flip_y, aug_idx = best_patch[int(pid)]

            prev = best_slice.get(slice_key)
            if prev is None or score > prev[1]:
                best_slice[slice_key] = (int(pid), float(score), float(angle), bool(flip_x), bool(flip_y), int(aug_idx))

        if not best_slice:
            return [], q

        if self.cfg.verbose:
            log("search", [
                f"Angles                        : {len(angles)}",
                f"flip_x                        : {bool(self.cfg.flip_x)}",
                f"flip_y                        : {bool(self.cfg.flip_y)}",
                f"Total query variants          : {len(aug_views)}",
                f"Candidates before patch dedup : {sum(len(I_all[a]) for a in range(len(aug_views)))}",
                f"Unique patch ids              : {len(best_patch)}",
                f"Unique indexed slices         : {len(best_slice)}",
            ])

        # 7) sort unique slices by score desc and keep top-k
        sorted_hits = sorted(best_slice.values(), key=lambda t: t[1], reverse=True)
        top = sorted_hits[: int(k)]

        ids = [pid for pid, _, _, _, _, _ in top]
        df_top = self.store.rows_for_ids(ids)

        results: List[SearchResult] = []
        for pid, score, angle, flip_x, flip_y, _aug_idx in top:
            row = df_top.loc[int(pid)]
            results.append(
                SearchResult(
                    patch_id=int(pid),
                    score=float(score),
                    angle=float(angle),
                    flip_x=bool(flip_x),
                    flip_y=bool(flip_y),
                    meta=row.to_dict(),
                )
            )

        # 8) Optional reranking
        if self.cfg.use_reranker and self.reranker is not None and results:
            self._apply_reranker(results, G, aug_views)

        return results, q

    def to_dataframe(self, hits: List[SearchResult]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for h in hits:
            row = dict(h.meta)
            row["patch_id"] = h.patch_id
            row["score"] = h.score
            row["query_angle_deg"] = h.angle
            row["query_flip_x"] = h.flip_x
            row["query_flip_y"] = h.flip_y
            if h.rerank_score is not None:
                row["rerank_score"] = h.rerank_score
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Query augmentation
    # ------------------------------------------------------------------

    def _build_query_variants(self, q_pil: Image.Image, angles: Tuple[float, ...]) -> List[Dict[str, Any]]:
        """
        Build all query augmentation variants.

        For each rotation angle, optionally include:
          - original
          - flip_x
          - flip_y
          - flip_x + flip_y

        Returns a list of dicts:
            {
                "pil": PIL.Image,
                "angle": float,
                "flip_x": bool,
                "flip_y": bool,
            }
        """
        variants: List[Dict[str, Any]] = []

        flip_states_x = [False, True] if self.cfg.flip_x else [False]
        flip_states_y = [False, True] if self.cfg.flip_y else [False]

        for angle in angles:
            base = q_pil.rotate(float(angle), resample=Image.BILINEAR, expand=False)

            for flip_x in flip_states_x:
                for flip_y in flip_states_y:
                    img = base

                    # PIL transpose names are from image coordinates:
                    # FLIP_LEFT_RIGHT  = horizontal mirror
                    # FLIP_TOP_BOTTOM  = vertical mirror
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
                        }
                    )

        return variants

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

    def _apply_reranker(self, results: List[SearchResult], G: np.ndarray, aug_views: List[Dict[str, Any]]) -> None:
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

        # --- compute reranker scores (topk,) ---
        if mode == "best_angle":
            q_idx = self._variant_to_idx(
                angle=float(top_hits[0].angle),
                flip_x=bool(top_hits[0].flip_x),
                flip_y=bool(top_hits[0].flip_y),
                aug_views=aug_views,
            )
            q_emb = G[q_idx]
            scores = self.reranker.score_list(q_emb, cand_embs)  # (topk,)

        elif mode == "max_over_angles":
            all_scores = []
            for aug_idx in range(len(aug_views)):
                q_emb = G[aug_idx]
                s = self.reranker.score_list(q_emb, cand_embs)  # (topk,)
                all_scores.append(s)
            scores = np.max(np.stack(all_scores, axis=0), axis=0).astype(np.float32, copy=False)

        elif mode == "per_hit_angle":
            q_embs = np.stack(
                [
                    G[
                        self._variant_to_idx(
                            angle=float(h.angle),
                            flip_x=bool(h.flip_x),
                            flip_y=bool(h.flip_y),
                            aug_views=aug_views,
                        )
                    ]
                    for h in top_hits
                ],
                axis=0,
            )  # (topk,D)
            scores = self.reranker.score_pairs(q_embs, cand_embs)  # (topk,)

        else:
            raise ValueError(f"Unknown rerank_query_mode={mode!r}")

        scores = np.asarray(scores, dtype=np.float32)

        # --- blend reranker + cosine (over the reranked topk) ---
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

        # store raw rerank score for logging/CSV, but use blended for ranking by overwriting h.score
        for h, s_raw, s_blend in zip(top_hits, scores, blended):
            h.rerank_score = float(s_raw)
            h.score = float(s_blend)

        # Sort top_hits by blended score desc; tie-breaker: raw reranker
        top_hits_sorted = sorted(
            top_hits,
            key=lambda h: (h.score, h.rerank_score if h.rerank_score is not None else -1e9),
            reverse=True,
        )

        results[:] = top_hits_sorted + rest_hits

    @staticmethod
    def _variant_to_idx(
        angle: float,
        flip_x: bool,
        flip_y: bool,
        aug_views: List[Dict[str, Any]],
    ) -> int:
        """
        Robust lookup for the query augmentation index that matches
        (angle, flip_x, flip_y).
        """
        best_idx = 0
        best_err = float("inf")

        for i, v in enumerate(aug_views):
            same_fx = bool(v["flip_x"]) == bool(flip_x)
            same_fy = bool(v["flip_y"]) == bool(flip_y)
            if not (same_fx and same_fy):
                continue

            err = abs(float(v["angle"]) - float(angle))
            if err < best_err:
                best_err = err
                best_idx = i

        return int(best_idx)

    # ------------------------------------------------------------------
    # Query preprocessing
    # ------------------------------------------------------------------

    def _prepare_query(self, img_np: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
        x = np.asarray(img_np)

        # Handle channels if present
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